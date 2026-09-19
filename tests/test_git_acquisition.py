from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import cast
from unittest import mock

import pytest

from valkeyrie.git_acquisition import (
    GitAcquisitionError,
    GitObjectAcquirer,
    _run_git,
    create_source_lock,
    load_source_lock,
)
from valkeyrie.revisions import Authority, RefKind, ResolvedRevision
from valkeyrie.sources import load_source_inventory

ROOT = Path(__file__).resolve().parents[1]


def _policy_digest(inventory: Mapping[str, object]) -> str:
    content = json.dumps(
        inventory,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _resolver(inventory: Mapping[str, object], repository_name: str) -> ResolvedRevision:
    repositories = cast(list[dict[str, object]], inventory["repositories"])
    entry = next(item for item in repositories if item["name"] == repository_name)
    requested_ref = cast(str, entry["requested_ref"])
    ref_kind: RefKind = "commit" if len(requested_ref) == 40 else "branch"
    return ResolvedRevision(
        repository=repository_name,
        repository_url=cast(str, entry["url"]),
        requested_ref=requested_ref,
        ref_kind=ref_kind,
        commit=hashlib.sha1(repository_name.encode(), usedforsecurity=False).hexdigest(),
        authority=cast(Authority, entry["authority"]),
        version_scope=cast(str, entry["version_scope"]),
        source_policy_digest=_policy_digest(inventory),
    )


def _locked() -> tuple[dict[str, object], bytes]:
    inventory = load_source_inventory(ROOT / "sources.yaml")
    return inventory, create_source_lock(inventory, resolve=_resolver)


def test_source_lock_is_canonical_complete_deterministic_and_policy_bound() -> None:
    inventory, document = _locked()

    assert document == create_source_lock(inventory, resolve=_resolver)
    locked = load_source_lock(inventory, document)
    reviewed = sorted(
        cast(str, item["name"])
        for item in cast(list[dict[str, object]], inventory["repositories"])
        if item["classification"] in {"curated", "structured_exact"}
    )
    assert len(reviewed) == 43
    assert [item.repository for item in locked.revisions] == reviewed
    assert locked.lock_id.startswith("sha256:")
    assert locked.resolve(inventory, "valkey").repository == "valkey"

    changed = json.loads(json.dumps(inventory))
    cast(dict[str, object], changed["defaults"])["max_file_bytes"] = 1
    with pytest.raises(GitAcquisitionError, match="policy identity"):
        load_source_lock(changed, document)


def test_source_lock_rejects_noncanonical_tampering_and_unknown_resolution() -> None:
    inventory, document = _locked()
    locked = load_source_lock(inventory, document)

    with pytest.raises(GitAcquisitionError, match="canonical JSON"):
        load_source_lock(inventory, document + b"\n")
    with pytest.raises(GitAcquisitionError, match="exactly one revision"):
        locked.resolve(inventory, "not-reviewed")


class FakeGit:
    def __init__(self, *, malformed_batch: bool = False) -> None:
        self.calls: list[tuple[tuple[str, ...], bytes | None, int]] = []
        self.malformed_batch = malformed_batch
        self.contents = {
            "README.md": b"# Valkey\n",
            "src/server.c": b"int main(void) { return 0; }\n",
        }
        self.shas = {path: _git_blob_sha(content) for path, content in self.contents.items()}

    def __call__(
        self,
        arguments: tuple[str, ...],
        *,
        input_bytes: bytes | None,
        maximum_output_bytes: int,
    ) -> bytes:
        self.calls.append((arguments, input_bytes, maximum_output_bytes))
        if arguments[-3:] == ("remote", "get-url", "origin"):
            return b"https://github.com/valkey-io/valkey.git\n"
        if "ls-tree" in arguments:
            # Unsized listing: sizes come from batch-check after the prefetch. Asking ls-tree
            # for sizes on a partial clone resolves every blob individually (measured 557s).
            assert "-l" not in arguments
            rows = []
            for path in sorted(self.contents):
                rows.append(
                    b"100644 blob " + self.shas[path].encode() + b"\t" + path.encode() + b"\0"
                )
            rows.append(b"100644 blob " + b"0" * 40 + b"\tasset.png\0")
            rows.append(b"160000 commit " + b"1" * 40 + b"\tvendor/dependency\0")
            return b"".join(rows)
        if "fetch" in arguments and "origin" in arguments and arguments[-1] != "origin":
            # The one-pack prefetch of exactly the selected blobs, with noop negotiation.
            selected = arguments[arguments.index("origin") + 1 :]
            assert set(selected) == {self.shas[path] for path in sorted(self.contents)}
            assert "fetch.negotiationAlgorithm=noop" in arguments
            return b""
        if arguments[-2:] == ("cat-file", "--batch-check"):
            assert input_bytes == b"".join(
                self.shas[path].encode() + b"\n" for path in sorted(self.contents)
            )
            return b"".join(
                f"{self.shas[path]} blob {len(self.contents[path])}\n".encode()
                for path in sorted(self.contents)
            )
        if arguments[-2:] == ("cat-file", "--batch"):
            assert input_bytes == b"".join(
                self.shas[path].encode() + b"\n" for path in sorted(self.contents)
            )
            rows = []
            for path in sorted(self.contents):
                content = self.contents[path]
                sha = self.shas[path]
                if self.malformed_batch and path == "README.md":
                    sha = "0" * 40
                rows.append(f"{sha} blob {len(content)}\n".encode() + content + b"\n")
            return b"".join(rows)
        if "cat-file" in arguments and "-e" in arguments:
            return b""
        raise AssertionError(f"unexpected Git call: {arguments!r}")


def _git_blob_sha(content: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(content)}\0".encode())
    digest.update(content)
    return digest.hexdigest()


def test_git_object_acquirer_reads_exact_selected_blobs_from_reviewed_cache(
    tmp_path: Path,
) -> None:
    inventory, document = _locked()
    revision = load_source_lock(inventory, document).resolve(inventory, "valkey")
    cache = tmp_path.resolve() / "cache"
    (cache / "valkey.git").mkdir(parents=True)
    git = FakeGit()

    acquired = GitObjectAcquirer(cache, run_git=git).acquire(inventory, revision)

    assert acquired.repository == "valkey"
    assert acquired.commit == revision.commit
    assert [(item.path, item.content) for item in acquired.files] == [
        ("README.md", b"# Valkey\n"),
        ("src/server.c", b"int main(void) { return 0; }\n"),
    ]
    assert acquired.total_bytes == sum(len(item.content) for item in acquired.files)
    assert all("asset.png" not in str(call) for call in git.calls)


def test_git_object_acquirer_fails_closed_on_blob_identity_conflict(
    tmp_path: Path,
) -> None:
    inventory, document = _locked()
    revision = load_source_lock(inventory, document).resolve(inventory, "valkey")
    cache = tmp_path.resolve() / "cache"
    (cache / "valkey.git").mkdir(parents=True)

    with pytest.raises(GitAcquisitionError, match="content conflicts"):
        GitObjectAcquirer(cache, run_git=FakeGit(malformed_batch=True)).acquire(inventory, revision)


def test_git_commands_refuse_replacement_and_untrusted_configuration() -> None:
    """A commit ID must name exactly one content tree, whatever the host or cache contains.

    Git's replacement mechanism makes ls-tree serve substituted content while still reporting the
    locked commit ID, and an existing bare cache can carry refs/replace. Suppressing system
    configuration alone leaves a global config reachable through HOME, which can reintroduce
    replacement, alternates, or transport rewriting.
    """
    captured: dict[str, dict[str, str]] = {}

    def fake_run(arguments: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        captured["env"] = dict(cast(dict[str, str], kwargs["env"]))
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")

    with mock.patch("subprocess.run", fake_run):
        _run_git(("--version",), input_bytes=b"", maximum_output_bytes=1024)

    environment = captured["env"]
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    # Pointed at the null device rather than merely unset, so an inherited HOME cannot supply one.
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_SYSTEM"] == os.devnull
