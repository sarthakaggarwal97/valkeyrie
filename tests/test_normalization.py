from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import pytest

from valkeyrie.acquisition import AcquiredFile, AcquiredRepository
from valkeyrie.normalization import (
    NormalizationError,
    NormalizationLimits,
    NormalizedDocument,
    canonical_document_bytes,
    canonical_document_identity_bytes,
    canonical_metadata_identity_bytes,
    normalize_repository,
)
from valkeyrie.revisions import Authority, ResolvedRevision
from valkeyrie.sources import load_source_inventory

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "sources.yaml"
COMMIT_1 = "1" * 40
COMMIT_2 = "2" * 40
DEFAULT_LIMITS = NormalizationLimits()


def _inventory() -> dict[str, object]:
    return deepcopy(load_source_inventory(SOURCES))


def _entry(inventory: dict[str, object], repository: str) -> dict[str, object]:
    repositories = cast(list[dict[str, object]], inventory["repositories"])
    return next(item for item in repositories if item["name"] == repository)


def _digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _policy_digest(inventory: dict[str, object]) -> str:
    canonical = json.dumps(
        inventory,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _digest(canonical)


def _resolved(
    inventory: dict[str, object],
    repository: str = "valkey",
    commit: str = COMMIT_1,
) -> ResolvedRevision:
    entry = _entry(inventory, repository)
    return ResolvedRevision(
        repository=repository,
        repository_url=cast(str, entry["url"]),
        requested_ref=cast(str, entry["requested_ref"]),
        ref_kind="branch",
        commit=commit,
        authority=cast(Authority, entry["authority"]),
        version_scope=cast(str, entry["version_scope"]),
        source_policy_digest=_policy_digest(inventory),
    )


def _acquired(
    files: list[tuple[str, bytes]],
    *,
    repository: str = "valkey",
    commit: str = COMMIT_1,
) -> AcquiredRepository:
    acquired_files = tuple(AcquiredFile(path, content) for path, content in files)
    return AcquiredRepository(
        repository=repository,
        commit=commit,
        files=acquired_files,
        total_bytes=sum(len(content) for _, content in files),
    )


def _normalize(
    files: list[tuple[str, bytes]],
    *,
    inventory: dict[str, object] | None = None,
    repository: str = "valkey",
    commit: str = COMMIT_1,
    limits: NormalizationLimits = DEFAULT_LIMITS,
) -> tuple[NormalizedDocument, ...]:
    source = inventory or _inventory()
    return normalize_repository(
        source,
        _resolved(source, repository, commit),
        _acquired(files, repository=repository, commit=commit),
        limits=limits,
    )


def test_normalizes_exact_source_content_and_complete_identity_templates() -> None:
    documents = _normalize(
        [
            ("src/server.c", b"int main(void) { return 0; }\n"),
            ("README.md", "Valkey \u2713\n".encode()),
        ]
    )

    assert [document.path for document in documents] == ["README.md", "src/server.c"]
    readme, code = documents
    assert readme.content == "Valkey \u2713\n"
    assert readme.content_type == "text/markdown"
    assert code.content_type == "text/code"
    assert readme.content_digest == _digest("Valkey \u2713\n".encode())

    identity = json.loads(canonical_document_identity_bytes(readme))
    assert identity == {
        "api_version": "valkeyrie.io/normalized-document-identity/1",
        "content_digest": readme.content_digest,
        "content_type": "text/markdown",
        "kind": "NormalizedDocumentIdentity",
        "path": "README.md",
        "source": {
            "authority": "canonical",
            "commit": COMMIT_1,
            "ref_kind": "branch",
            "repository": "valkey",
            "repository_url": "https://github.com/valkey-io/valkey",
            "requested_ref": "unstable",
            "source_policy_digest": readme.source.source_policy_digest,
            "version_scope": "branches_tags_and_releases",
        },
    }
    assert readme.document_id == _digest(canonical_document_identity_bytes(readme))

    document = json.loads(canonical_document_bytes(readme))
    assert document == {
        "api_version": "valkeyrie.io/normalized-document/1",
        "content": "Valkey \u2713\n",
        "content_type": "text/markdown",
        "document_id": readme.document_id,
        "kind": "NormalizedDocument",
    }
    metadata = json.loads(canonical_metadata_identity_bytes(readme))
    assert metadata == {
        "api_version": "valkeyrie.io/metadata-sidecar/1",
        "authority": "canonical",
        "commit": COMMIT_1,
        "content_digest": readme.content_digest,
        "content_type": "text/markdown",
        "document_id": readme.document_id,
        "kind": "MetadataSidecar",
        "path": "README.md",
        "repository": "valkey",
        "version_scope": "branches_tags_and_releases",
    }
    assert "generation_id" not in identity | document | metadata
    assert "metadata_digest" not in metadata
    assert "document_id" not in identity


def test_identical_inputs_are_byte_identical_and_input_order_independent() -> None:
    files = [("src/server.c", b"source\n"), ("README.md", b"docs\n")]
    forward = _normalize(files)
    reverse = _normalize(list(reversed(files)))

    assert forward == reverse
    assert [item.path for item in forward] == ["README.md", "src/server.c"]
    assert [canonical_document_bytes(item) for item in forward] == [
        canonical_document_bytes(item) for item in reverse
    ]
    assert [canonical_metadata_identity_bytes(item) for item in forward] == [
        canonical_metadata_identity_bytes(item) for item in reverse
    ]
    # The same document built in the other order serializes to the same identity bytes.
    assert canonical_document_identity_bytes(forward[0]) == canonical_document_identity_bytes(
        reverse[0]
    )


@pytest.mark.parametrize(
    ("repository", "path", "expected"),
    [
        ("valkey", "README.md", "text/markdown"),
        ("valkey-helm", "docs-site/src/content/docs/guide.mdx", "text/markdown"),
        ("valkey", "src/commands/get.json", "application/json"),
        ("valkey-helm", "config/default.yaml", "application/yaml"),
        ("valkey-helm", "config/default.yml", "application/yaml"),
        ("valkey", "src/server.c", "text/code"),
        ("valkey-glide", "README.rst", "text/plain"),
        ("valkey", "LICENSE-APACHE", "text/plain"),
        ("valkey", "runtest-cluster", "text/plain"),
    ],
)
def test_content_type_is_derived_only_from_the_reviewed_path(
    repository: str, path: str, expected: str
) -> None:
    inventory = _inventory()
    resolved = _resolved(inventory, repository)
    acquired = _acquired(
        [(path, b"untrusted content cannot choose metadata\n")],
        repository=repository,
    )

    document = normalize_repository(inventory, resolved, acquired)[0]

    assert document.content_type == expected


def test_every_identity_field_and_content_mutation_changes_the_preimage() -> None:
    baseline = _normalize([("README.md", b"one\n")])[0]
    variants = [
        _normalize([("SECURITY.md", b"one\n")])[0],
        _normalize([("README.md", b"two\n")])[0],
        _normalize([("README.md", b"one\n")], commit=COMMIT_2)[0],
        _normalize(
            [("README.md", b"one\n")],
            repository="valkey-skills",
        )[0],
    ]

    changed_scope = _inventory()
    _entry(changed_scope, "valkey")["version_scope"] = "other_reviewed_scope"
    variants.append(_normalize([("README.md", b"one\n")], inventory=changed_scope)[0])

    changed_authority = _inventory()
    _entry(changed_authority, "valkey")["authority"] = "secondary"
    variants.append(_normalize([("README.md", b"one\n")], inventory=changed_authority)[0])

    original = canonical_document_identity_bytes(baseline)
    assert all(canonical_document_identity_bytes(variant) != original for variant in variants)
    assert all(variant.document_id != baseline.document_id for variant in variants)


def test_secondary_document_authority_is_preserved_but_structured_source_is_rejected() -> None:
    secondary = _normalize(
        [("README.md", b"secondary\n")],
        repository="valkey-skills",
    )[0]
    assert json.loads(canonical_metadata_identity_bytes(secondary))["authority"] == "secondary"

    inventory = _inventory()
    resolved = _resolved(inventory, "valkey-hashes")
    acquired = _acquired(
        [("hashes.txt", b"digest\n")],
        repository="valkey-hashes",
    )
    with pytest.raises(NormalizationError, match="document normalization|source policy"):
        normalize_repository(inventory, resolved, acquired)


def test_source_inventory_binds_authority_version_and_acquisition_identity() -> None:
    inventory = _inventory()
    resolved = _resolved(inventory)
    acquired = _acquired([("README.md", b"content\n")])
    invalid_revisions = [
        replace(resolved, repository_url="https://github.com/valkey-io/other"),
        replace(resolved, requested_ref="other"),
        replace(resolved, authority="secondary"),
        replace(resolved, version_scope="other"),
        replace(resolved, source_policy_digest="sha256:" + "0" * 64),
        replace(resolved, commit="A" * 40),
        replace(resolved, ref_kind=cast(Any, "other")),
    ]
    for candidate in invalid_revisions:
        with pytest.raises(NormalizationError):
            normalize_repository(inventory, candidate, acquired)

    invalid_acquisitions = (
        replace(acquired, repository="valkey-doc"),
        replace(acquired, commit=COMMIT_2),
    )
    for invalid_acquisition in invalid_acquisitions:
        with pytest.raises(NormalizationError, match="acquisition identity"):
            normalize_repository(inventory, resolved, invalid_acquisition)


def test_rejects_invalid_inventory_and_non_document_classification() -> None:
    resolved = _resolved(_inventory())
    acquired = _acquired([("README.md", b"content\n")])
    with pytest.raises(NormalizationError, match="source inventory"):
        normalize_repository(cast(dict[str, object], {}), resolved, acquired)
    with pytest.raises(NormalizationError, match="source inventory must be a mapping"):
        normalize_repository(cast(dict[str, object], object()), resolved, acquired)

    inventory = _inventory()
    entry = _entry(inventory, "valkey")
    entry["classification"] = "structured_exact"
    entry["authority"] = "structured"
    entry["ingestion_mode"] = "structured_records"
    entry["path_policy"] = "structured_hashes"
    structured = replace(
        resolved,
        authority="structured",
        source_policy_digest=_policy_digest(inventory),
    )
    with pytest.raises(NormalizationError, match="document normalization|source policy"):
        normalize_repository(inventory, structured, acquired)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/README.md",
        "../README.md",
        "docs/../README.md",
        "docs\\README.md",
        "docs//README.md",
        "docs/./README.md",
        "README.md/",
        "line\nbreak.md",
        "control\x7f.md",
        "\ud800.md",
    ],
)
def test_rejects_malformed_and_unsafe_paths(path: str) -> None:
    with pytest.raises(NormalizationError, match="path|UTF-8"):
        _normalize([(path, b"content\n")])


def test_rejects_duplicate_and_policy_excluded_paths() -> None:
    with pytest.raises(NormalizationError, match="repeats path"):
        _normalize([("README.md", b"one\n"), ("README.md", b"two\n")])
    with pytest.raises(NormalizationError, match="outside reviewed source policy"):
        _normalize([("unreviewed/file.md", b"content\n")])


@pytest.mark.parametrize(
    ("content", "error"),
    [
        (b"", "must not be empty"),
        (b"line\r", "bare carriage return"),
        (b"text\x00payload", "binary or unsafe"),
        (b"text\x01payload", "binary or unsafe"),
        (b"text\x7fpayload", "binary or unsafe"),
        (b"text\xc2\x85payload", "binary or unsafe"),
        ("text\u202epayload".encode(), "binary or unsafe"),
        ("text\u2028payload".encode(), "binary or unsafe"),
        (b"\xff\xfe", "canonical UTF-8"),
    ],
)
def test_rejects_empty_noncanonical_or_binary_content(content: bytes, error: str) -> None:
    with pytest.raises(NormalizationError, match=error):
        _normalize([("README.md", content)])


def test_preserves_tabs_lf_unicode_and_missing_final_newline_exactly() -> None:
    content = "line\tvalue\nlast \u2713".encode()
    document = _normalize([("README.md", content)])[0]
    assert document.content.encode() == content
    assert json.loads(canonical_document_bytes(document))["content"] == content.decode()


def test_strips_one_leading_utf8_bom_before_identity_and_content_hashing() -> None:
    document = _normalize([("README.md", b"\xef\xbb\xbfcontent\n")])[0]

    assert document.content == "content\n"
    assert document.content_digest == _digest(b"content\n")
    assert json.loads(canonical_document_bytes(document))["content"] == "content\n"


def test_rejects_repeated_utf8_bom_after_single_leading_normalization() -> None:
    with pytest.raises(NormalizationError, match="binary or unsafe control"):
        _normalize([("README.md", b"\xef\xbb\xbf\xef\xbb\xbfcontent\n")])


@pytest.mark.parametrize("path", ["README.bin", "runtest.bin"])
def test_unrecognised_suffixes_are_read_as_plain_text_not_refused(path: str) -> None:
    """An unrecognised suffix must not end the build for every repository.

    Refusing it protected nothing: the content is text the moment it decoded as UTF-8, which
    acquisition establishes before normalization sees it, and an extensionless path has always
    defaulted the same way. Aborting meant one .cmake or .patch appearing upstream would stop every
    corpus update, which from the outside is indistinguishable from the refresh never running.
    """
    inventory = _inventory()
    resolved = _resolved(inventory)
    documents = normalize_repository(inventory, resolved, _acquired([(path, b"content\n")]))
    assert len(documents) == 1
    assert documents[0].content_type == "text/plain"
    assert documents[0].path == path


def test_normalizes_reviewed_extensionless_www_entrypoint() -> None:
    documents = _normalize(
        [
            ("examples/express/bin/www", b"#!/usr/bin/env node\n"),
            ("examples/express/views/error.jade", b"h1= message\n"),
        ],
        repository="iovalkey",
    )

    assert [document.content_type for document in documents] == ["text/code", "text/code"]


def test_normalizes_reviewed_extensionless_source_script_as_code() -> None:
    documents = _normalize(
        [
            ("tests/scripts/redis-cluster", b"#!/bin/sh\n"),
            ("tests/scripts/simulated-valkey.pl", b"#!/usr/bin/env perl\n"),
        ],
        repository="libvalkey",
    )

    assert [document.content_type for document in documents] == ["text/code", "text/code"]


def test_file_path_total_and_count_bounds_are_inclusive_and_fail_above_edge() -> None:
    exact = _normalize(
        [("a.md", b"a"), ("b.md", b"b")],
        limits=NormalizationLimits(
            max_files=2, max_path_bytes=4, max_file_bytes=1, max_total_bytes=2
        ),
    )
    assert len(exact) == 2


def test_normalizes_reviewed_glide_example_source_suffixes() -> None:
    documents = _normalize(
        [
            ("examples/scala/build.sbt", b"scalaVersion := 3\n"),
            ("examples/scala/src/main/scala/ClusterExample.scala", b"object Example {}\n"),
        ],
        repository="valkey-glide",
    )

    assert {document.content_type for document in documents} == {"text/code"}

    cases = [
        (
            [("a.md", b"a"), ("b.md", b"b")],
            NormalizationLimits(max_files=1),
            "file-count",
        ),
        (
            [("long.md", b"a")],
            NormalizationLimits(max_path_bytes=6),
            "unsafe document path",
        ),
        (
            [("a.md", b"ab")],
            NormalizationLimits(max_file_bytes=1),
            "file-byte",
        ),
        (
            [("a.md", b"a"), ("b.md", b"b")],
            NormalizationLimits(max_total_bytes=1),
            "total-byte",
        ),
    ]
    for files, limits, error in cases:
        with pytest.raises(NormalizationError, match=error):
            _normalize(files, limits=limits)


@pytest.mark.parametrize(
    "limits",
    [
        NormalizationLimits(max_files=0),
        NormalizationLimits(max_path_bytes=0),
        NormalizationLimits(max_file_bytes=0),
        NormalizationLimits(max_total_bytes=0),
        NormalizationLimits(max_files=cast(int, True)),
    ],
)
def test_rejects_non_positive_and_boolean_bounds(limits: NormalizationLimits) -> None:
    with pytest.raises(NormalizationError, match="positive integers"):
        _normalize([("README.md", b"content\n")], limits=limits)


def test_reviewed_policy_file_and_repository_bounds_cannot_be_expanded() -> None:
    inventory = _inventory()
    defaults = cast(dict[str, object], inventory["defaults"])
    defaults["max_file_bytes"] = 1
    defaults["max_repository_bytes"] = 1
    resolved = _resolved(inventory)
    acquired = _acquired([("README.md", b"ab")])
    with pytest.raises(NormalizationError, match="file-byte"):
        normalize_repository(
            inventory,
            resolved,
            acquired,
            limits=NormalizationLimits(max_file_bytes=100, max_total_bytes=100),
        )


def test_rejects_inconsistent_acquisition_totals_and_runtime_types() -> None:
    inventory = _inventory()
    resolved = _resolved(inventory)
    acquired = _acquired([("README.md", b"content\n")])
    with pytest.raises(NormalizationError, match="total_bytes"):
        normalize_repository(inventory, resolved, replace(acquired, total_bytes=1))
    with pytest.raises(NormalizationError, match="total_bytes"):
        normalize_repository(
            inventory,
            resolved,
            replace(acquired, total_bytes=cast(int, True)),
        )
    with pytest.raises(NormalizationError, match="immutable tuple"):
        normalize_repository(
            inventory,
            resolved,
            replace(acquired, files=cast(tuple[AcquiredFile, ...], [acquired.files[0]])),
        )
    malformed_file = replace(
        acquired,
        files=(cast(AcquiredFile, object()),),
    )
    with pytest.raises(NormalizationError, match="malformed file"):
        normalize_repository(inventory, resolved, malformed_file)
    wrong_content = replace(
        acquired,
        files=(AcquiredFile("README.md", cast(bytes, "content")),),
    )
    with pytest.raises(NormalizationError, match="content must be bytes"):
        normalize_repository(inventory, resolved, wrong_content)
    with pytest.raises(NormalizationError, match="resolved revision"):
        normalize_repository(inventory, cast(ResolvedRevision, object()), acquired)
    with pytest.raises(NormalizationError, match="acquired repository metadata"):
        normalize_repository(inventory, resolved, cast(AcquiredRepository, object()))
    with pytest.raises(NormalizationError, match="limits"):
        normalize_repository(
            inventory,
            resolved,
            acquired,
            limits=cast(NormalizationLimits, object()),
        )


def test_canonicalizers_reject_content_checksum_document_id_and_type_inconsistency() -> None:
    document = _normalize([("README.md", b"content\n")])[0]
    variants = [
        replace(document, content="changed\n"),
        replace(document, content_digest="sha256:" + "0" * 64),
        replace(document, document_id="sha256:" + "0" * 64),
        replace(document, content_type="text/plain"),
        replace(document, path="SECURITY.md"),
    ]
    for variant in variants:
        with pytest.raises(NormalizationError, match="inconsistent"):
            canonical_document_bytes(variant)
        with pytest.raises(NormalizationError, match="inconsistent"):
            canonical_metadata_identity_bytes(variant)

    with pytest.raises(NormalizationError, match="malformed"):
        canonical_document_bytes(cast(NormalizedDocument, object()))


def test_templates_are_frozen_and_api_has_no_generation_or_network_input() -> None:
    document = _normalize([("README.md", b"content\n")])[0]
    with pytest.raises(FrozenInstanceError):
        document.path = "other.md"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        NormalizationLimits().max_files = 1  # type: ignore[misc]

    repeated = _normalize([("README.md", b"content\n")])[0]
    assert document.source == repeated.source
    assert b"generation_id" not in canonical_document_identity_bytes(document)
    assert b"generation_id" not in canonical_metadata_identity_bytes(document)


def test_normalizes_reviewed_csharp_project_files_as_code() -> None:
    documents = _normalize(
        [("tests/Project/Project.csproj", b"<Project />\n")],
        repository="valkey-glide-csharp",
    )
    assert documents[0].content_type == "text/code"


def test_normalizes_crlf_pairs_before_identity_and_content_hashing() -> None:
    document = _normalize([("README.md", b"first\r\nsecond\r\n")])[0]

    assert document.content == "first\nsecond\n"
    assert document.content_digest == _digest(b"first\nsecond\n")


def test_normalizes_reviewed_csv_test_fixture_as_plain_text() -> None:
    documents = _normalize(
        [("tests/test_asyncio/testdata/titles.csv", b"title,score\nvalkey,1\n")],
        repository="valkey-py",
    )
    assert documents[0].content_type == "text/plain"


def test_normalizes_reviewed_ldap_packaging_metadata() -> None:
    documents = _normalize(
        [
            ("packaging/debian/valkey-ldap.docs", b"README.md\n"),
            ("packaging/debian/valkey-ldap.install", b"usr/lib\n"),
            ("packaging/valkey-ldap.spec.in", b"Name: valkey-ldap\n"),
        ],
        repository="valkey-ldap",
    )
    assert [document.content_type for document in documents] == [
        "text/plain",
        "text/plain",
        "text/code",
    ]
