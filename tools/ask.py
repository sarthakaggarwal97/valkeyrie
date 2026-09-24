"""Minimal single-user web UI for asking the deployed Valkeyrie assistant questions.

Testing utility. Runs a local HTTP server that invokes the deployed Lambda with your
own AWS credentials, so nothing is exposed to the internet and no deployment or
authorization change is needed.

    uv run --with boto3==1.40.21 python tools/ask.py
    # then open http://127.0.0.1:8765

The server generates its own request id per question. That id is the idempotency and
lease key, so it is never taken from user input.
"""

from __future__ import annotations

import html
import json
import os
import time
import uuid
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from secrets import compare_digest
from typing import Any, cast

import boto3
from botocore.config import Config

FUNCTION = "valkeyrie-development-application"
# Version 9 runs Claude Fable 5, the candidate that passed qualification, with a 120s
# timeout so the densest questions are not cut off. The corpus is resolved at runtime
# from the active pointer, so this serves the complete generation. Version 8 is Opus,
# which failed the claim-to-evidence gate at 0.743 against a required 1.0.
QUALIFIER = "59"
KNOWLEDGE_BASE_ID = "ONVASJDDNX"
REGION = "us-east-1"
ADDRESS = ("127.0.0.1", 8765)

# Questions about current project state must route live; anything else uses the
# pinned corpus. Keyword choice mirrors routing.py so the UI does not fight the router.
_LIVE_HINTS = ("current", "currently", "latest", "right now", "upcoming", "recent", "status of")

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Ask Valkeyrie</title>
<style>
 body{{font:16px/1.5 system-ui,sans-serif;max-width:52rem;margin:3rem auto;padding:0 1rem}}
 form{{display:flex;gap:.5rem}} input{{flex:1;padding:.6rem;font-size:1rem}}
 button{{padding:.6rem 1.2rem;font-size:1rem;cursor:pointer}}
 .meta{{color:#555;font-size:.85rem;margin:1rem 0}}
 .claim{{margin:.6rem 0;padding:.6rem .9rem;background:#f4f6f8;border-radius:6px}}
 .cite{{font-size:.8rem;color:#444;word-break:break-all}}
 .outcome{{display:inline-block;padding:.1rem .5rem;border-radius:4px;background:#e8eef5}}
</style></head><body>
<h1>Ask Valkeyrie</h1>
<form method="get" action="/">{token_field}
  <input name="q" value="{question}"
         placeholder="How do Valkey replication and failover behave?" autofocus>
  <button type="submit">Ask</button>
</form>
{answer}
</body></html>
"""


def _render(question: str, body: str) -> bytes:
    field = ""
    if Handler.token is not None:
        field = f'<input type="hidden" name="t" value="{html.escape(Handler.token)}">'
    return _PAGE.format(question=html.escape(question), answer=body, token_field=field).encode(
        "utf-8"
    )


def _answer_html(result: dict[str, Any], elapsed_ms: float) -> str:
    outcome = html.escape(str(result.get("outcome")))
    parts = [
        f'<p class="meta"><span class="outcome">{outcome}</span> '
        f"&nbsp;{elapsed_ms:.0f} ms &nbsp;generation "
        f"{html.escape(str(result.get('generation_id')))}</p>"
    ]
    message = result.get("message")
    if message:
        parts.append(f"<p>{html.escape(str(message))}</p>")
    for claim in cast(list[dict[str, Any]], result.get("claims") or []):
        parts.append(f'<div class="claim">{html.escape(str(claim.get("text")))}</div>')
    citations = cast(list[str], result.get("citations") or [])
    if citations:
        parts.append("<p class='meta'>Sources</p><ul>")
        parts.extend(f'<li class="cite">{html.escape(c)}</li>' for c in citations)
        parts.append("</ul>")
    return "\n".join(parts)


class Handler(BaseHTTPRequestHandler):
    lambda_client: Any = None
    token: str | None = None

    def do_GET(self) -> None:  # noqa: N802  (BaseHTTPRequestHandler API)
        if self.path.startswith("/favicon"):
            self.send_error(404)
            return
        question = ""
        supplied = ""
        if "?" in self.path:
            from urllib.parse import parse_qs, urlsplit

            query = parse_qs(urlsplit(self.path).query)
            question = query.get("q", [""])[0].strip()
            supplied = query.get("t", [""])[0]
        # When ASK_TOKEN is set the server is assumed to be reachable by others, so every
        # request must carry it. Each answer costs money, so an unprotected exposed URL
        # would let anyone spend the account's Bedrock budget.
        if Handler.token is not None and not compare_digest(supplied, Handler.token):
            self.send_error(403, "missing or incorrect token")
            return
        body = ""
        if question:
            try:
                body = self._ask(question)
            except Exception as error:  # surface the failure instead of a blank page
                body = f'<p class="claim">Request failed: {html.escape(str(error))}</p>'
        payload = _render(question, body)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _ask(self, question: str) -> str:
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        live = any(hint in question.lower() for hint in _LIVE_HINTS)
        event = {
            "action": "answer",
            # Server-generated: this is the idempotency and lease key, never user input.
            "request_id": f"req_ask-{uuid.uuid4().hex}",
            "question": question,
            "version_requirement": "current_state" if live else "none",
            "requested_version": None,
            "knowledge_base_id": KNOWLEDGE_BASE_ID,
            "owner": "ask-ui",
            "lease_duration_seconds": 300,
            "now": stamp,
            "completed_at": stamp,
        }
        started = time.monotonic_ns()
        response = Handler.lambda_client.invoke(
            FunctionName=FUNCTION, Qualifier=QUALIFIER, Payload=json.dumps(event).encode()
        )
        elapsed_ms = (time.monotonic_ns() - started) / 1e6
        result = json.loads(response["Payload"].read().decode())
        return _answer_html(result, elapsed_ms)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} {format % args}")


def main() -> int:
    Handler.lambda_client = boto3.client(
        "lambda",
        region_name=REGION,
        config=Config(retries={"max_attempts": 0}, read_timeout=300, connect_timeout=20),
    )
    Handler.token = os.environ.get("ASK_TOKEN") or None
    server = ThreadingHTTPServer(ADDRESS, Handler)
    if Handler.token is None:
        print("no ASK_TOKEN set: local use only, do not expose this port")
    else:
        print("ASK_TOKEN required; share URLs as .../?t=<token>&q=<question>")
    print(f"Ask Valkeyrie on http://{ADDRESS[0]}:{ADDRESS[1]}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
