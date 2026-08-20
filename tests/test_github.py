from __future__ import annotations

from collections.abc import Callable

import pytest

from valkeyrie.github import GitHubReadError, fetch_public_github


class FakeSocket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


class FakeResponse:
    status = 200

    def __init__(
        self,
        chunks: list[bytes],
        *,
        content_type: str,
        after_read: Callable[[], None] | None = None,
    ) -> None:
        self.chunks = chunks
        self.content_type = content_type
        self.after_read = after_read

    def getheaders(self) -> list[tuple[str, str]]:
        return [("Content-Type", self.content_type)]

    def read1(self, amount: int) -> bytes:
        chunk = self.chunks.pop(0) if self.chunks else b""
        if len(chunk) > amount:
            self.chunks.insert(0, chunk[amount:])
            chunk = chunk[:amount]
        if self.after_read is not None:
            self.after_read()
        return chunk


class FakeConnection:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.sock = FakeSocket()
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.closed = False

    def request(self, method: str, target: str, headers: dict[str, str]) -> None:
        self.requests.append((method, target, headers))

    def getresponse(self) -> FakeResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("url", "host", "target", "accept"),
    [
        (
            "https://api.github.com/orgs/valkey-io/repos?page=1",
            "api.github.com",
            "/orgs/valkey-io/repos?page=1",
            "application/vnd.github+json",
        ),
        (
            "https://github.com/valkey-io/valkey.git/info/refs?service=git-upload-pack",
            "github.com",
            "/valkey-io/valkey.git/info/refs?service=git-upload-pack",
            "application/x-git-upload-pack-advertisement",
        ),
    ],
)
def test_transport_is_get_only_credential_free_and_origin_constrained(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    host: str,
    target: str,
    accept: str,
) -> None:
    response = FakeResponse([b"{}", b""], content_type=accept)
    connection = FakeConnection(response)
    opened: list[tuple[str, float]] = []

    def open_connection(request_host: str, timeout: float) -> FakeConnection:
        opened.append((request_host, timeout))
        return connection

    monkeypatch.setattr("valkeyrie.github.http.client.HTTPSConnection", open_connection)

    result = fetch_public_github(url, 2.0, 100, elapsed_clock=lambda: 0.0)

    assert result.body == b"{}"
    assert opened == [(host, 2.0)]
    assert len(connection.requests) == 1
    method, actual_target, headers = connection.requests[0]
    assert method == "GET"
    assert actual_target == target
    assert headers["Accept"] == accept
    assert not any("authorization" in name.casefold() for name in headers)
    assert connection.closed is True
    assert connection.sock.timeouts


@pytest.mark.parametrize(
    ("url", "timeout", "max_bytes"),
    [
        ("http://api.github.com/repos", 1.0, 10),
        ("https://user@api.github.com/repos", 1.0, 10),
        ("https://api.github.com/repos#fragment", 1.0, 10),
        ("https://github.com/other/repo.git/info/refs?service=git-upload-pack", 1.0, 10),
        ("https://github.com/valkey-io/valkey.git/info/refs?service=other", 1.0, 10),
        ("https://api.github.com/repos", 0.0, 10),
        ("https://api.github.com/repos", float("nan"), 10),
        ("https://api.github.com/repos", float("inf"), 10),
        ("https://api.github.com/repos", True, 10),
        ("https://api.github.com/repos", 1.0, 0),
        ("https://api.github.com/repos", 1.0, True),
    ],
)
def test_transport_rejects_unsafe_urls_and_non_finite_or_non_integer_bounds(
    url: str, timeout: float, max_bytes: int
) -> None:
    with pytest.raises(GitHubReadError, match="invalid bounded"):
        fetch_public_github(url, timeout, max_bytes)


def test_transport_enforces_one_deadline_across_slow_body_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    elapsed = [0.0]

    def advance() -> None:
        elapsed[0] += 0.6

    response = FakeResponse(
        [b"a", b"b", b""],
        content_type="application/vnd.github+json",
        after_read=advance,
    )
    connection = FakeConnection(response)
    monkeypatch.setattr(
        "valkeyrie.github.http.client.HTTPSConnection",
        lambda host, timeout: connection,
    )

    with pytest.raises(GitHubReadError, match="time bound"):
        fetch_public_github(
            "https://api.github.com/repos",
            1.0,
            100,
            elapsed_clock=lambda: elapsed[0],
        )

    assert connection.closed is True
