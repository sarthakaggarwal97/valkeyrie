"""Run the question battery against a deployed Valkeyrie version and report reliability.

Testing utility, not part of the request path. It invokes the deployed Lambda with your own AWS
credentials, exactly as tools/ask.py does, so nothing is exposed and no deployment change is needed.

    uv run --with boto3==1.40.21 --with pyyaml==6.0.2 python tools/regression/run.py
    uv run --with boto3==1.40.21 --with pyyaml==6.0.2 python tools/regression/run.py --draws 3
    ... python tools/regression/run.py --id sibling-helm   # one entry

WHY DRAWS. The pipeline is stochastic at two points: the router chooses lookups, and the answer turn
decides whether its evidence is enough. One pass tells you whether a question CAN be answered; three
tell you whether it RELIABLY is. A question that answers two draws in three is not passing, it is
flaking, and a single pass hides exactly the failure that took four draws to see in production.

TWO AXES. A RAG answer can fail in two independent places, and one pass/fail hides which. Axis one
is the answer SHAPE: the outcome the entry expects, with at least one claim for an answer. Axis two,
for an entry that names `cites` (a substring of a citation), is whether the answer CITED a source
matching it. Neither axis judges the claims' truth; a wrong claim citing the right file passes both.
What the second axis buys is attribution: an answer with the right shape that did not cite the
expected source is a RETRIEVAL problem and is marked so, and it FAILS the draw, because the answer
was drawn from somewhere other than where the truth lives.

Exit status is 0 only when every question matched its expectation on every draw, so this is usable
as a gate. Failures print the outcome and the message so the reason is in the output, not in a
follow-up investigation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import boto3
import yaml
from botocore.config import Config

FUNCTION = "valkeyrie-development-application"
KNOWLEDGE_BASE_ID = "ONVASJDDNX"
REGION = "us-east-1"
BATTERY = Path(__file__).with_name("battery.yaml")
# Lambda rejects concurrent invocations of the same version beyond a small burst (TooManyRequests
# was observed at six), and every draw is a full model turn, so two at a time is the ceiling that
# finishes a 28-question battery without failing for the wrong reason.
PARALLELISM = 2
# An answer is an answer; these two are the other legitimate shapes. "partial" is a dependency
# failure (the corpus or GitHub was unreachable), which is neither a pass nor a regression, so it
# is reported verbatim rather than mapped onto one of these.
_EXPECTED = {"answer", "clarification", "refusal"}


def _outcome_matches(expected: str, outcome: str, claims: int) -> bool:
    if expected == "answer":
        return outcome == "answer" and claims >= 1
    if expected == "clarification":
        return outcome == "clarification"
    # A refusal is an abstention. An answer to a question that must be refused is the serious
    # direction of failure, which is why these entries are in the battery at all.
    return outcome == "abstention"


def _load(only: str | None) -> list[dict[str, Any]]:
    cases = cast(list[dict[str, Any]], yaml.safe_load(BATTERY.read_text()))
    for case in cases:
        missing = {"id", "question", "expect", "because"} - set(case)
        if missing:
            raise SystemExit(f"battery entry {case.get('id')!r} is missing {sorted(missing)}")
        if case["expect"] not in _EXPECTED:
            raise SystemExit(f"battery entry {case['id']!r} expects {case['expect']!r}")
    if only is not None:
        cases = [case for case in cases if case["id"] == only]
        if not cases:
            raise SystemExit(f"no battery entry with id {only!r}")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualifier", default=None, help="deployed version (default: the bot's)")
    parser.add_argument("--draws", type=int, default=1, help="runs per question (default 1)")
    parser.add_argument("--id", default=None, help="run one battery entry")
    parser.add_argument("--json", action="store_true", help="write a machine-readable report")
    args = parser.parse_args()

    qualifier = args.qualifier
    if qualifier is None:
        # The version the bot serves is the version worth testing, and it is recorded in one place.
        text = (Path(__file__).parents[1] / "slack_bot.py").read_text()
        qualifier = text.split('QUALIFIER = "', 1)[1].split('"', 1)[0]

    cases = _load(args.id)
    client = boto3.client(
        "lambda", region_name=REGION, config=Config(read_timeout=180, retries={"max_attempts": 0})
    )

    def run(item: tuple[dict[str, Any], int]) -> dict[str, Any]:
        case, draw = item
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        event: dict[str, Any] = {
            "action": "answer",
            # Generated here: this is the idempotency and lease key, so every draw is its own
            # request rather than a replay of the previous one.
            "request_id": f"req_reg-{uuid.uuid4().hex}",
            "question": case["question"],
            "version_requirement": "current_state" if case.get("live") else "none",
            "requested_version": None,
            "knowledge_base_id": KNOWLEDGE_BASE_ID,
            "owner": "regression",
            "lease_duration_seconds": 300,
            "now": now,
            "completed_at": now,
        }
        if case.get("conversation"):
            event["conversation"] = case["conversation"]
        started = time.monotonic()
        try:
            payload = client.invoke(
                FunctionName=FUNCTION, Qualifier=qualifier, Payload=json.dumps(event).encode()
            )["Payload"].read()
            result = cast(dict[str, Any], json.loads(payload.decode()))
        except Exception as error:  # noqa: BLE001 - the report names the failure either way
            result = {"outcome": "invoke_failed", "message": f"{type(error).__name__}: {error}"}
        claims = result.get("claims") or []
        citations = [str(c) for c in (result.get("citations") or [])]
        wanted = case.get("cites")
        cited_ok = None if not wanted else any(str(wanted) in c for c in citations)
        return {
            "id": case["id"],
            "draw": draw,
            "expect": case["expect"],
            "outcome": result.get("outcome", "missing"),
            "claims": len(claims),
            "detail": (claims[0]["text"] if claims else result.get("message")) or "",
            "seconds": round(time.monotonic() - started, 1),
            "cites": wanted,
            "cited_ok": cited_ok,
            "passed": _outcome_matches(case["expect"], str(result.get("outcome")), len(claims))
            and cited_ok is not False,
        }

    work = [(case, draw) for case in cases for draw in range(1, args.draws + 1)]
    print(f"battery: {len(cases)} questions x {args.draws} draw(s) against :{qualifier}\n")
    with ThreadPoolExecutor(max_workers=PARALLELISM) as pool:
        rows = list(pool.map(run, work))

    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(row["id"], []).append(row)

    flaking = 0
    failing = 0
    for case in cases:
        draws = by_id[case["id"]]
        passed = sum(1 for row in draws if row["passed"])
        mark = "pass" if passed == len(draws) else ("FLAKE" if passed else "FAIL")
        if mark == "FLAKE":
            flaking += 1
        elif mark == "FAIL":
            failing += 1
        cited = [row["cited_ok"] for row in draws if row["cited_ok"] is not None]
        axis = ""
        if cited:
            hits = sum(1 for c in cited if c)
            axis = f"  cited {hits}/{len(cited)}"
            if hits < len(cited):
                axis += " RETRIEVAL"
        print(f"{mark:5} {passed}/{len(draws)}  {case['id']:32} expect {case['expect']}{axis}")
        if passed != len(draws):
            print(f"        because: {case['because']}")
            for row in draws:
                if not row["passed"]:
                    print(f"        got {row['outcome']}: {row['detail'][:120]}")

    total = sum(1 for row in rows if row["passed"])
    print(f"\n{total}/{len(rows)} draws matched. {failing} failing, {flaking} flaking.")
    print(
        "outcomes: "
        + ", ".join(f"{k}={v}" for k, v in sorted(Counter(r["outcome"] for r in rows).items()))
    )
    if args.json:
        report = Path("/tmp") / f"valkeyrie-regression-{qualifier}.json"
        report.write_text(json.dumps({"qualifier": qualifier, "rows": rows}, indent=1))
        print(f"report: {report}")
    return 0 if total == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
