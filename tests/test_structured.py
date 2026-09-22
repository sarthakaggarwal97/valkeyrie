from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import cast

import pytest

from valkeyrie.acquisition import AcquiredFile, AcquiredRepository
from valkeyrie.revisions import Authority, RefKind, ResolvedRevision
from valkeyrie.structured import (
    MAX_COMMAND_BYTES,
    MAX_GITHUB_OBJECT_ID,
    MAX_PATH_BYTES,
    MAX_SYMBOL_BYTES,
    AmbiguousIdentifierError,
    CommandIdentifier,
    CommandRecord,
    CommitIdentifier,
    CommitRecord,
    DuplicateIdentifierError,
    ExactIdentifier,
    ExactLookup,
    GitHubObjectIdentifier,
    GitHubObjectRecord,
    GitHubObjectType,
    MissingIdentifierError,
    PathIdentifier,
    PathRecord,
    ReleaseArtifactDigestRecord,
    ReleaseArtifactIdentifier,
    RepositoryIdentifier,
    RepositoryRecord,
    StructuredParsingLimits,
    StructuredRecord,
    StructuredRecordError,
    SymbolIdentifier,
    SymbolRecord,
    build_release_artifact_digest_records,
    canonical_record_bytes,
    canonical_records_bytes,
    record_checksum,
)

COMMIT = "1" * 40
OTHER_COMMIT = "2" * 40
POLICY_DIGEST = f"sha256:{'3' * 64}"
ARTIFACT_DIGEST = f"sha256:{'4' * 64}"
OTHER_ARTIFACT_DIGEST = f"sha256:{'5' * 64}"


def _source(
    repository: str = "valkey",
    *,
    commit: str = COMMIT,
    authority: Authority = "canonical",
    version_scope: str = "branches_tags_and_releases",
) -> ResolvedRevision:
    return ResolvedRevision(
        repository=repository,
        repository_url=f"https://github.com/valkey-io/{repository}",
        requested_ref="main",
        ref_kind="branch",
        commit=commit,
        authority=authority,
        version_scope=version_scope,
        source_policy_digest=POLICY_DIGEST,
    )


def _records() -> tuple[StructuredRecord, ...]:
    source = _source()
    hashes = _source("valkey-hashes", authority="structured", version_scope="release_artifacts")
    return (
        RepositoryRecord(source, RepositoryIdentifier("valkey")),
        CommitRecord(source, CommitIdentifier(COMMIT)),
        PathRecord(source, PathIdentifier("valkey", "src/server.c")),
        SymbolRecord(source, SymbolIdentifier("valkey", "src/server.c", "processCommand")),
        GitHubObjectRecord(
            source,
            GitHubObjectIdentifier("valkey", "pull_request", 4073),
        ),
        CommandRecord(source, CommandIdentifier("CLIENT CACHING")),
        ReleaseArtifactDigestRecord(
            hashes,
            ReleaseArtifactIdentifier("9.0.0", "valkey-9.0.0.tar.gz"),
            ARTIFACT_DIGEST,
        ),
    )


def test_canonical_templates_preserve_complete_source_and_typed_values() -> None:
    records = _records()
    templates = [json.loads(canonical_record_bytes(record)) for record in records]

    assert [template["record_type"] for template in templates] == [
        "repository",
        "commit",
        "path",
        "symbol",
        "github_object",
        "command",
        "release_artifact_digest",
    ]
    for template in templates:
        assert template["api_version"] == "valkeyrie.io/structured-record/1"
        assert template["kind"] == "StructuredRecord"
        assert template["source"] == {
            "authority": template["source"]["authority"],
            "commit": COMMIT,
            "ref_kind": "branch",
            "repository": template["source"]["repository"],
            "repository_url": (f"https://github.com/valkey-io/{template['source']['repository']}"),
            "requested_ref": "main",
            "source_policy_digest": POLICY_DIGEST,
            "version_scope": template["source"]["version_scope"],
        }
    assert templates[4]["identifier"] == {
        "object_id": 4073,
        "object_type": "pull_request",
        "repository": "valkey",
    }
    assert templates[5]["record_id"] == "command:client%20caching"
    assert templates[6]["value"] == {"digest": ARTIFACT_DIGEST}
    assert templates[2]["provenance"] == {
        "commit": COMMIT,
        "path": "src/server.c",
        "repository": "valkey",
    }
    assert templates[3]["provenance"] == templates[2]["provenance"]
    assert templates[6]["provenance"] == {
        "commit": COMMIT,
        "path": "README",
        "repository": "valkey-hashes",
    }


def test_identity_preimages_omit_generation_and_self_checksum_fields() -> None:
    preimage = canonical_records_bytes(_records())

    assert b"generation_id" not in preimage
    assert b"record_digest" not in preimage
    assert b"record_checksum" not in preimage
    assert b'"checksum"' not in preimage
    assert record_checksum(_records()[0]) == (
        f"sha256:{hashlib.sha256(canonical_record_bytes(_records()[0])).hexdigest()}"
    )


def test_record_and_collection_serialization_is_deterministic() -> None:
    records = _records()

    assert canonical_records_bytes(records) == canonical_records_bytes(tuple(reversed(records)))
    # A record rebuilt from scratch serializes and checksums identically: the serialization
    # depends on the record's content, never on object identity or construction order.
    rebuilt = _records()[3]
    assert rebuilt is not records[3]
    assert canonical_record_bytes(records[3]) == canonical_record_bytes(rebuilt)
    assert record_checksum(records[3]) == record_checksum(rebuilt)
    # And a different record does not collide.
    assert record_checksum(records[3]) != record_checksum(records[2])

    index = ExactLookup(tuple(reversed(records)))
    ordered = [
        json.loads(canonical_record_bytes(record))["record_type"] for record in index.records
    ]
    assert ordered == sorted(ordered)


def test_every_covered_identifier_resolves_by_type_and_exact_value() -> None:
    records = _records()
    lookup = ExactLookup(records)

    for record in records:
        assert lookup.lookup(record.identifier) is record

    with pytest.raises(MissingIdentifierError, match="missing exact identifier"):
        lookup.lookup(CommandIdentifier("GET"))
    with pytest.raises(StructuredRecordError, match="canonical uppercase"):
        lookup.lookup(CommandIdentifier("client caching"))
    with pytest.raises(MissingIdentifierError):
        lookup.lookup(PathIdentifier("valkey", "SRC/server.c"))


def test_github_lookup_requires_exact_repository_type_and_numeric_id() -> None:
    record = cast(GitHubObjectRecord, _records()[4])
    lookup = ExactLookup((record,))

    assert lookup.lookup(GitHubObjectIdentifier("valkey", "pull_request", 4073)) is record
    for identifier in (
        GitHubObjectIdentifier("valkey-doc", "pull_request", 4073),
        GitHubObjectIdentifier("valkey", "issue", 4073),
        GitHubObjectIdentifier("valkey", "pull_request", 4074),
    ):
        with pytest.raises(MissingIdentifierError):
            lookup.lookup(identifier)


def test_release_artifact_lookup_returns_only_explicit_valkey_hashes_record() -> None:
    record = cast(ReleaseArtifactDigestRecord, _records()[-1])
    lookup = ExactLookup((record,))

    found = lookup.lookup(ReleaseArtifactIdentifier("9.0.0", "valkey-9.0.0.tar.gz"))
    assert found is record
    assert found.digest == ARTIFACT_DIGEST
    with pytest.raises(MissingIdentifierError):
        lookup.lookup(ReleaseArtifactIdentifier("9.0.1", "valkey-9.0.0.tar.gz"))


def test_duplicate_and_ambiguous_identifier_records_fail_index_creation() -> None:
    record = cast(CommandRecord, _records()[5])
    with pytest.raises(DuplicateIdentifierError, match="duplicate exact identifier"):
        ExactLookup((record, record))

    conflicting = replace(record, source=replace(record.source, requested_ref="unstable"))
    with pytest.raises(AmbiguousIdentifierError, match="ambiguous exact identifier"):
        ExactLookup((record, conflicting))

    digest = cast(ReleaseArtifactDigestRecord, _records()[-1])
    changed_digest = replace(digest, digest=f"sha256:{'5' * 64}")
    with pytest.raises(AmbiguousIdentifierError):
        canonical_records_bytes((digest, changed_digest))


def test_source_revision_authority_version_and_policy_are_identity_inputs() -> None:
    record = cast(PathRecord, _records()[2])
    variants = (
        replace(record, source=replace(record.source, requested_ref="unstable")),
        replace(record, source=replace(record.source, ref_kind="tag")),
        replace(record, source=replace(record.source, authority="secondary")),
        replace(record, source=replace(record.source, version_scope="another_scope")),
        replace(
            record,
            source=replace(record.source, source_policy_digest=f"sha256:{'9' * 64}"),
        ),
    )

    original = canonical_record_bytes(record)
    assert all(canonical_record_bytes(variant) != original for variant in variants)
    assert all(record_checksum(variant) != record_checksum(record) for variant in variants)


@pytest.mark.parametrize(
    ("source", "error"),
    [
        (replace(_source(), repository_url="https://github.com/other/valkey"), "URL"),
        (replace(_source(), commit="A" * 40), "40-hex"),
        (replace(_source(), requested_ref=" "), "requested ref"),
        (replace(_source(), ref_kind=cast(RefKind, "tree")), "ref kind"),
        (replace(_source(), authority=cast(Authority, "none")), "authority"),
        (replace(_source(), version_scope=""), "version scope"),
        (replace(_source(), source_policy_digest="sha256:short"), "policy digest"),
    ],
)
def test_rejects_malformed_source_metadata(source: ResolvedRevision, error: str) -> None:
    record = CommandRecord(source, CommandIdentifier("GET"))
    with pytest.raises(StructuredRecordError, match=error):
        canonical_record_bytes(record)


@pytest.mark.parametrize(
    "identifier",
    [
        RepositoryIdentifier("Valkey"),
        CommitIdentifier("1" * 39),
        CommitIdentifier("A" * 40),
        PathIdentifier("valkey", "/src/server.c"),
        PathIdentifier("valkey", "src/../server.c"),
        PathIdentifier("valkey", "src\\server.c"),
        PathIdentifier("valkey", f"src/{'x' * MAX_PATH_BYTES}"),
        SymbolIdentifier("valkey", "src/server.c", " "),
        SymbolIdentifier("valkey", "src/server.c", "x\nvalue"),
        SymbolIdentifier("valkey", "src/server.c", "x" * (MAX_SYMBOL_BYTES + 1)),
        GitHubObjectIdentifier("valkey", cast(GitHubObjectType, "commit"), 1),
        GitHubObjectIdentifier("valkey", "issue", 0),
        GitHubObjectIdentifier("valkey", "issue", cast(int, True)),
        GitHubObjectIdentifier("valkey", "issue", MAX_GITHUB_OBJECT_ID + 1),
        CommandIdentifier("get"),
        CommandIdentifier("GET\nSET"),
        CommandIdentifier("X" * (MAX_COMMAND_BYTES + 1)),
        ReleaseArtifactIdentifier("../9.0", "valkey.tar.gz"),
        ReleaseArtifactIdentifier("9.0", "../valkey.tar.gz"),
    ],
)
def test_rejects_malformed_or_out_of_bounds_exact_identifiers(
    identifier: ExactIdentifier,
) -> None:
    with pytest.raises(StructuredRecordError):
        ExactLookup(()).lookup(identifier)


def test_accepts_exact_path_symbol_command_and_object_bounds() -> None:
    source = _source()
    path = "p" * MAX_PATH_BYTES
    symbol = "s" * MAX_SYMBOL_BYTES
    command = "X" * MAX_COMMAND_BYTES
    records: tuple[StructuredRecord, ...] = (
        PathRecord(source, PathIdentifier("valkey", path)),
        SymbolRecord(source, SymbolIdentifier("valkey", "src/server.c", symbol)),
        CommandRecord(source, CommandIdentifier(command)),
        GitHubObjectRecord(
            source,
            GitHubObjectIdentifier("valkey", "workflow_run", MAX_GITHUB_OBJECT_ID),
        ),
    )

    lookup = ExactLookup(records)
    assert all(lookup.lookup(record.identifier) is record for record in records)


@pytest.mark.parametrize(
    "record",
    [
        RepositoryRecord(_source(), RepositoryIdentifier("valkey-doc")),
        CommitRecord(_source(), CommitIdentifier(OTHER_COMMIT)),
        PathRecord(_source(), PathIdentifier("valkey-doc", "README.md")),
        SymbolRecord(_source(), SymbolIdentifier("valkey-doc", "README.md", "heading")),
        GitHubObjectRecord(_source(), GitHubObjectIdentifier("valkey-doc", "issue", 1)),
        CommandRecord(_source("valkey-doc"), CommandIdentifier("GET")),
        ReleaseArtifactDigestRecord(
            _source(),
            ReleaseArtifactIdentifier("9.0", "valkey.tar.gz"),
            ARTIFACT_DIGEST,
        ),
    ],
)
def test_record_identifier_must_match_its_authoritative_source(
    record: StructuredRecord,
) -> None:
    with pytest.raises(StructuredRecordError):
        canonical_record_bytes(record)


def test_every_record_wrapper_rejects_a_mismatched_identifier_type() -> None:
    source = _source()
    hashes = _source("valkey-hashes", authority="structured", version_scope="release_artifacts")
    records: tuple[StructuredRecord, ...] = (
        RepositoryRecord(source, cast(RepositoryIdentifier, CommitIdentifier(COMMIT))),
        CommitRecord(
            source,
            cast(CommitIdentifier, PathIdentifier("valkey", "src/server.c")),
        ),
        PathRecord(
            source,
            cast(PathIdentifier, SymbolIdentifier("valkey", "src/server.c", "main")),
        ),
        SymbolRecord(
            source,
            cast(SymbolIdentifier, GitHubObjectIdentifier("valkey", "issue", 1)),
        ),
        GitHubObjectRecord(
            source,
            cast(GitHubObjectIdentifier, CommandIdentifier("GET")),
        ),
        CommandRecord(
            source,
            cast(CommandIdentifier, RepositoryIdentifier("valkey")),
        ),
        ReleaseArtifactDigestRecord(
            hashes,
            cast(ReleaseArtifactIdentifier, RepositoryIdentifier("valkey-hashes")),
            ARTIFACT_DIGEST,
        ),
    )

    for record in records:
        with pytest.raises(StructuredRecordError, match="types do not match"):
            canonical_record_bytes(record)


def test_wrong_typed_and_non_utf8_runtime_values_fail_with_typed_error() -> None:
    malformed_identifiers = (
        GitHubObjectIdentifier(
            "valkey",
            cast(GitHubObjectType, []),
            1,
        ),
        PathIdentifier("valkey", "bad\ud800path"),
    )
    for identifier in malformed_identifiers:
        with pytest.raises(StructuredRecordError):
            ExactLookup(()).lookup(identifier)

    malformed_sources = (
        replace(_source(), ref_kind=cast(RefKind, [])),
        replace(_source(), authority=cast(Authority, [])),
        replace(_source(), version_scope="bad\ud800scope"),
    )
    for source in malformed_sources:
        with pytest.raises(StructuredRecordError):
            canonical_record_bytes(CommandRecord(source, CommandIdentifier("GET")))


@pytest.mark.parametrize(
    "digest",
    ["4" * 64, "sha256:short", f"sha256:{'A' * 64}"],
)
def test_release_artifact_digest_must_be_canonical_sha256(digest: str) -> None:
    hashes = _source("valkey-hashes", authority="structured", version_scope="release_artifacts")
    record = ReleaseArtifactDigestRecord(
        hashes,
        ReleaseArtifactIdentifier("9.0", "valkey.tar.gz"),
        digest,
    )
    with pytest.raises(StructuredRecordError, match="sha256"):
        canonical_record_bytes(record)


def test_unsupported_runtime_record_and_identifier_types_fail_closed() -> None:
    with pytest.raises(StructuredRecordError, match="record type"):
        canonical_record_bytes(cast(StructuredRecord, object()))
    with pytest.raises(StructuredRecordError, match="identifier type"):
        ExactLookup(()).lookup(cast(ExactIdentifier, object()))


def _hash_line(
    release: str = "9.0.0",
    digest: str = "4" * 64,
) -> bytes:
    artifact = f"valkey-{release}.tar.gz"
    url = (
        "https://github.com/valkey-io/valkey/archive/unstable.tar.gz"
        if release == "unstable"
        else f"https://github.com/valkey-io/valkey/archive/refs/tags/{release}.tar.gz"
    )
    return f"hash {artifact} sha256 {digest} {url}".encode()


def _acquired_hashes(
    files: tuple[AcquiredFile, ...],
    *,
    repository: str = "valkey-hashes",
    commit: str = COMMIT,
    total_bytes: int | None = None,
) -> AcquiredRepository:
    return AcquiredRepository(
        repository,
        commit,
        files,
        sum(len(item.content) for item in files) if total_bytes is None else total_bytes,
    )


def _hash_source(*, commit: str = COMMIT) -> ResolvedRevision:
    return _source(
        "valkey-hashes",
        commit=commit,
        authority="structured",
        version_scope="release_artifacts",
    )


def test_builds_release_records_from_exact_acquired_canonical_formats() -> None:
    readme = b"# reviewed Valkey release hashes\n" + _hash_line() + b"\n"
    sums = f"{'5' * 64}  valkey-9.1.0.tar.gz\n".encode()
    acquired = _acquired_hashes(
        (
            AcquiredFile("README", readme),
            AcquiredFile("README.md", b"Human documentation only.\n"),
            AcquiredFile("releases/9.1.sha256", sums),
        )
    )

    records = build_release_artifact_digest_records(acquired, _hash_source())

    assert [(record.identifier.release, record.identifier.artifact) for record in records] == [
        ("9.0.0", "valkey-9.0.0.tar.gz"),
        ("9.1.0", "valkey-9.1.0.tar.gz"),
    ]
    assert [record.digest for record in records] == [ARTIFACT_DIGEST, OTHER_ARTIFACT_DIGEST]
    assert [record.provenance_path for record in records] == [
        "README",
        "releases/9.1.sha256",
    ]
    assert all(record.provenance_path != record.identifier.artifact for record in records)
    assert b"generation_id" not in canonical_records_bytes(records)
    assert b"record_checksum" not in canonical_records_bytes(records)


def test_builder_emits_only_digests_present_in_acquired_content() -> None:
    acquired = _acquired_hashes((AcquiredFile("README", _hash_line() + b"\n"),))

    records = build_release_artifact_digest_records(acquired, _hash_source())

    assert len(records) == 1
    assert records[0].digest == ARTIFACT_DIGEST
    assert records[0].digest != OTHER_ARTIFACT_DIGEST


@pytest.mark.parametrize(
    ("line", "error"),
    [
        (b"not a digest", "malformed"),
        (_hash_line().replace(b" sha256 ", b" sha1 "), "malformed"),
        (_hash_line().replace(b"4" * 64, b"A" * 64), "malformed"),
        (_hash_line().replace(b"https://github.com", b"https://example.com"), "URL"),
        (b"4" * 64 + b" valkey-9.0.0.tar.gz", "malformed"),
        (b"4" * 64 + b"  ../valkey-9.0.0.tar.gz", "malformed"),
        (b"4" * 64 + b"  valkey-9.0.0.zip", "malformed"),
    ],
)
def test_rejects_malformed_digest_lines(line: bytes, error: str) -> None:
    path = (
        "README" if line.startswith(b"hash ") or line == b"not a digest" else "releases/9.0.sha256"
    )
    acquired = _acquired_hashes((AcquiredFile(path, line + b"\n"),))

    with pytest.raises(StructuredRecordError, match=error):
        build_release_artifact_digest_records(acquired, _hash_source())


@pytest.mark.parametrize(
    "content",
    [
        _hash_line() + b"\r\n",
        b"\xef\xbb\xbf" + _hash_line() + b"\n",
        b"hash \xff\n",
        _hash_line() + b"\x00\n",
    ],
)
def test_rejects_noncanonical_or_non_utf8_structured_content(content: bytes) -> None:
    acquired = _acquired_hashes((AcquiredFile("README", content),))

    with pytest.raises(StructuredRecordError):
        build_release_artifact_digest_records(acquired, _hash_source())


def test_rejects_duplicate_and_omits_conflicting_release_artifact_identifiers() -> None:
    duplicate = _acquired_hashes(
        (
            AcquiredFile("README", _hash_line() + b"\n"),
            AcquiredFile(
                "releases/9.0.sha256",
                b"4" * 64 + b"  valkey-9.0.0.tar.gz\n",
            ),
        )
    )
    with pytest.raises(DuplicateIdentifierError):
        build_release_artifact_digest_records(duplicate, _hash_source())

    safe_line = b"6" * 64 + b"  valkey-9.1.0.tar.gz\n"
    conflicting = replace(
        duplicate,
        files=(
            duplicate.files[0],
            AcquiredFile(
                "releases/9.0.sha256",
                b"5" * 64 + b"  valkey-9.0.0.tar.gz\n" + safe_line,
            ),
        ),
    )
    conflicting = replace(
        conflicting,
        total_bytes=sum(len(item.content) for item in conflicting.files),
    )

    records = build_release_artifact_digest_records(conflicting, _hash_source())

    assert len(records) == 1
    assert records[0].identifier == ReleaseArtifactIdentifier("9.1.0", "valkey-9.1.0.tar.gz")
    assert records[0].digest == f"sha256:{'6' * 64}"


@pytest.mark.parametrize(
    "acquired",
    [
        _acquired_hashes((AcquiredFile("docs/guide.md", b"text\n"),)),
        _acquired_hashes((AcquiredFile("../README", _hash_line()),)),
        _acquired_hashes(
            (AcquiredFile("README", _hash_line()), AcquiredFile("README", _hash_line()))
        ),
        _acquired_hashes(
            (
                AcquiredFile("releases/9.0.sha256", b"4" * 64 + b"  valkey-9.0.0.tar.gz"),
                AcquiredFile("README", _hash_line()),
            )
        ),
        _acquired_hashes((AcquiredFile("README", _hash_line()),), total_bytes=1),
    ],
)
def test_rejects_unreviewed_unsafe_duplicate_unordered_or_inconsistent_acquisition(
    acquired: AcquiredRepository,
) -> None:
    with pytest.raises(StructuredRecordError):
        build_release_artifact_digest_records(acquired, _hash_source())


def test_rejects_non_valkey_hashes_source_and_revision_mismatch() -> None:
    acquired = _acquired_hashes((AcquiredFile("README", _hash_line()),))
    invalid_sources = (
        _source(),
        replace(_hash_source(), authority="canonical"),
        replace(_hash_source(), version_scope="other"),
    )
    for source in invalid_sources:
        with pytest.raises(StructuredRecordError, match="valkey-hashes"):
            build_release_artifact_digest_records(acquired, source)

    with pytest.raises(StructuredRecordError, match="exact revision"):
        build_release_artifact_digest_records(acquired, _hash_source(commit=OTHER_COMMIT))


def test_release_record_requires_safe_reviewed_distinct_provenance_path() -> None:
    hashes = _hash_source()
    identifier = ReleaseArtifactIdentifier("9.0.0", "valkey-9.0.0.tar.gz")
    for provenance_path in ("docs/guide.md", "../README", "valkey-9.0.0.tar.gz"):
        with pytest.raises(StructuredRecordError, match="provenance"):
            canonical_record_bytes(
                ReleaseArtifactDigestRecord(
                    hashes,
                    identifier,
                    ARTIFACT_DIGEST,
                    provenance_path,
                )
            )


@pytest.mark.parametrize(
    "limits",
    [
        StructuredParsingLimits(max_files=0),
        StructuredParsingLimits(max_file_bytes=1),
        StructuredParsingLimits(max_total_bytes=1),
        StructuredParsingLimits(max_lines=1),
        StructuredParsingLimits(max_line_bytes=1),
        StructuredParsingLimits(max_records=1),
    ],
)
def test_structured_parser_enforces_every_bound(limits: StructuredParsingLimits) -> None:
    content = _hash_line() + b"\n" + _hash_line("9.1.0", "5" * 64) + b"\n"
    acquired = _acquired_hashes((AcquiredFile("README", content),))

    with pytest.raises(StructuredRecordError, match="bound"):
        build_release_artifact_digest_records(acquired, _hash_source(), limits=limits)


def test_structured_parser_rejects_wrong_typed_limits_and_acquisition_fields() -> None:
    acquired = _acquired_hashes((AcquiredFile("README", _hash_line()),))
    with pytest.raises(StructuredRecordError, match="limits"):
        build_release_artifact_digest_records(
            acquired,
            _hash_source(),
            limits=cast(StructuredParsingLimits, object()),
        )
    malformed = replace(acquired, files=cast(tuple[AcquiredFile, ...], []))
    with pytest.raises(StructuredRecordError, match="immutable tuple"):
        build_release_artifact_digest_records(malformed, _hash_source())
