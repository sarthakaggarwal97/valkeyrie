"""Minimal credential-free GET transport for bounded public GitHub API reads."""

from __future__ import annotations

import http.client
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Protocol
from urllib.parse import parse_qsl, urlsplit


class GitHubReadError(ValueError):
    """A constrained public GitHub read could not be completed safely."""


@dataclass(frozen=True)
class HttpResponse:
    """Minimal response returned by an injectable public GitHub fetcher."""

    status: int
    headers: Mapping[str, str]
    body: bytes


class GitHubFetcher(Protocol):
    """Fetch one already constrained public GitHub URL."""

    def __call__(self, url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse: ...


_GIT_REFS_PATH = re.compile(r"^/valkey-io/[a-z0-9.][a-z0-9._-]*[.]git/info/refs$")


def fetch_public_github(
    url: str,
    timeout_seconds: float,
    max_bytes: int,
    *,
    elapsed_clock: Callable[[], float] = monotonic,
    token: str | None = None,
    method: str = "GET",
    body_bytes: bytes | None = None,
) -> HttpResponse:
    """Issue one bounded GET without redirects or arbitrary origins.

    ``token`` is optional and read-only in effect. It authenticates the request so GitHub
    applies the account rate limit of 5,000 requests an hour rather than the anonymous
    60 an hour tied to the caller's IP address, which the runtime exhausts quickly. It is
    sent only on api.github.com requests, never on the git-refs path, and every URL, byte
    and time bound below applies unchanged.
    """
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
        or not math.isfinite(timeout_seconds)
        or not isinstance(max_bytes, int)
        or isinstance(max_bytes, bool)
        or max_bytes < 1
    ):
        raise GitHubReadError("invalid bounded public GitHub request")

    parsed = urlsplit(url)
    api_request = parsed.netloc == "api.github.com"
    git_request = (
        parsed.netloc == "github.com"
        and _GIT_REFS_PATH.fullmatch(parsed.path) is not None
        and parse_qsl(parsed.query, keep_blank_values=True) == [("service", "git-upload-pack")]
    )
    if parsed.scheme != "https" or parsed.fragment or not (api_request or git_request):
        raise GitHubReadError("invalid bounded public GitHub request")

    headers = {
        "Accept": (
            "application/vnd.github+json"
            if api_request
            else "application/x-git-upload-pack-advertisement"
        ),
        "User-Agent": "valkeyrie-read/0.1",
    }
    if method not in {"GET", "POST"} or (method == "GET") != (body_bytes is None):
        raise GitHubReadError("GitHub read method and body are inconsistent")
    if body_bytes is not None:
        # A POST is only ever a GraphQL query, and only to api.github.com; it is bounded in size
        # like everything else here, and it still reads, so it is still sent read-only.
        if not api_request or parsed.path != "/graphql" or len(body_bytes) > 64 * 1024:
            raise GitHubReadError("GitHub POST is only permitted for a bounded GraphQL query")
        headers["Content-Type"] = "application/json"
    if api_request:
        headers["X-GitHub-Api-Version"] = "2022-11-28"
        if token:
            headers["Authorization"] = f"Bearer {token}"
    target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
    deadline = elapsed_clock() + float(timeout_seconds)

    def remaining_time() -> float:
        remaining = deadline - elapsed_clock()
        if remaining <= 0:
            raise GitHubReadError("public GitHub read exceeded its time bound")
        return remaining

    connection = http.client.HTTPSConnection(parsed.netloc, timeout=remaining_time())
    try:
        connection.request(method, target, body=body_bytes, headers=headers)
        if connection.sock is not None:
            connection.sock.settimeout(remaining_time())
        response = connection.getresponse()
        response_headers: dict[str, str] = {}
        for name, value in response.getheaders():
            folded_name = name.casefold()
            response_headers[folded_name] = ",".join(
                filter(None, (response_headers.get(folded_name), value))
            )

        body = bytearray()
        while len(body) <= max_bytes:
            if connection.sock is not None:
                connection.sock.settimeout(remaining_time())
            chunk = response.read1(min(64 * 1024, max_bytes + 1 - len(body)))
            remaining_time()
            if not chunk:
                break
            body.extend(chunk)
        return HttpResponse(response.status, response_headers, bytes(body))
    except GitHubReadError:
        raise
    except (OSError, TimeoutError, http.client.HTTPException) as error:
        raise GitHubReadError(f"public GitHub read failed: {error}") from error
    finally:
        connection.close()


def fetch_github_graphql(
    query: str,
    variables: Mapping[str, object],
    timeout_seconds: float,
    max_bytes: int,
    *,
    token: str,
) -> HttpResponse:
    """Execute one GraphQL query. Satisfies live_github.ProjectsGraphQLFetcher.

    Projects (v2) are only reachable through GraphQL and only with a token carrying read:project,
    so this is the one place the runtime speaks GraphQL. The same transport, byte and time bounds
    apply as for every REST read; the token is the same read-only token.
    """
    if not isinstance(query, str) or not query.strip():
        raise GitHubReadError("GraphQL query must be non-blank text")
    body = json.dumps({"query": query, "variables": dict(variables)}, separators=(",", ":"))
    return fetch_public_github(
        "https://api.github.com/graphql",
        timeout_seconds,
        max_bytes,
        token=token,
        method="POST",
        body_bytes=body.encode("utf-8"),
    )
