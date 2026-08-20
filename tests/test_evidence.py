from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from tests.helpers import load_yaml
from valkeyrie.acquisition import AcquiredFile, AcquiredRepository
from valkeyrie.evidence import (
    ClaimSupport,
    DocumentExcerpt,
    EvidenceError,
    EvidenceLimits,
    EvidencePackage,
    StructuredRecordEvidence,
    canonical_evidence_record_bytes,
    create_evidence_package,
    evidence_record_value,
    render_citations,
    validate_claim_support,
    verify_evidence_package,
)
from valkeyrie.generation import GenerationBundle, create_generation_bundle
from valkeyrie.normalization import NormalizedDocument, normalize_repository
from valkeyrie.retrieval_config import FrozenRetrievalConfiguration, load_retrieval_config
from valkeyrie.revisions import Authority, ResolvedRevision
from valkeyrie.sources import load_source_inventory
from valkeyrie.structured import (
    CommandIdentifier,
    CommandRecord,
    CommitIdentifier,
    CommitRecord,
    GitHubObjectIdentifier,
    GitHubObjectRecord,
    PathIdentifier,
    PathRecord,
    ReleaseArtifactDigestRecord,
    RepositoryIdentifier,
    RepositoryRecord,
    StructuredRecord,
    SymbolIdentifier,
    SymbolRecord,
    build_release_artifact_digest_records,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "sources.yaml"
CONFIG = ROOT / "retrieval-config.yaml"
SCHEMA = cast(dict[str, Any], load_yaml(ROOT / "src/valkeyrie/schemas/contracts.schema.json"))
COMMIT = "1" * 40
OTHER_COMMIT = "2" * 40
CREATED_AT = "2026-08-19T00:04:20Z"
ARTIFACT_DIGEST = f"sha256:{'a' * 64}"


@pytest.fixture(scope="module")
def retrieval_config() -> FrozenRetrievalConfiguration:
    return load_retrieval_config(CONFIG)


def _digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _inventory() -> dict[str, object]:
    return deepcopy(load_source_inventory(SOURCES))


def _policy_digest(inventory: dict[str, object]) -> str:
    return _digest(
        json.dumps(
            inventory,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _source(
    inventory: dict[str, object],
    repository: str = "valkey",
    *,
    commit: str = COMMIT,
) -> ResolvedRevision:
    repositories = cast(list[dict[str, object]], inventory["repositories"])
    entry = next(item for item in repositories if item["name"] == repository)
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


def _documents(
    inventory: dict[str, object],
    *,
    commit: str = COMMIT,
    readme: bytes = b"Valkey documentation\n",
) -> tuple[NormalizedDocument, ...]:
    source = _source(inventory, commit=commit)
    files = (
        AcquiredFile("README.md", readme),
        AcquiredFile("src/server.c", b"int processCommand(void) { return 1; }\n"),
    )
    return normalize_repository(
        inventory,
        source,
        AcquiredRepository(
            repository="valkey",
            commit=commit,
            files=files,
            total_bytes=sum(len(item.content) for item in files),
        ),
    )


def _hash_acquisition(
    *,
    commit: str = COMMIT,
    digest: str = ARTIFACT_DIGEST,
) -> AcquiredRepository:
    line = (
        "hash valkey-9.0.0.tar.gz sha256 "
        f"{digest.removeprefix('sha256:')} "
        "https://github.com/valkey-io/valkey/archive/refs/tags/9.0.0.tar.gz\n"
    ).encode()
    return AcquiredRepository(
        "valkey-hashes",
        commit,
        (AcquiredFile("README", line),),
        len(line),
    )


def _default_records(
    inventory: dict[str, object],
    *,
    commit: str = COMMIT,
    acquisition: AcquiredRepository | None = None,
) -> tuple[StructuredRecord, ...]:
    source = _source(inventory, commit=commit)
    hashes = _source(inventory, "valkey-hashes", commit=commit)
    acquired = acquisition or _hash_acquisition(commit=commit)
    return (
        PathRecord(source, PathIdentifier("valkey", "src/server.c")),
        CommandRecord(source, CommandIdentifier("GET")),
        *build_release_artifact_digest_records(acquired, hashes),
    )


def _bundle(
    retrieval_config: FrozenRetrievalConfiguration,
    *,
    records: tuple[StructuredRecord, ...] | None = None,
    commit: str = COMMIT,
    readme: bytes = b"Valkey documentation\n",
    created_at: str = CREATED_AT,
    acquisition: AcquiredRepository | None = None,
) -> GenerationBundle:
    inventory = _inventory()
    acquired = acquisition or _hash_acquisition(commit=commit)
    selected_records = (
        _default_records(inventory, commit=commit, acquisition=acquired)
        if records is None
        else records
    )
    acquisitions = (
        (acquired,)
        if any(isinstance(record, ReleaseArtifactDigestRecord) for record in selected_records)
        else ()
    )
    return create_generation_bundle(
        SOURCES.read_bytes(),
        _documents(inventory, commit=commit, readme=readme),
        selected_records,
        retrieval_config,
        created_at=created_at,
        structured_acquisitions=acquisitions,
    )


def _readme(bundle: GenerationBundle) -> NormalizedDocument:
    return next(document for document in bundle.document_templates if document.path == "README.md")


def _release_record_id(bundle: GenerationBundle) -> str:
    return next(
        item.object_id
        for item in bundle.structured_records
        if item.object_id.startswith("release_artifact_digest:")
    )


def _package(bundle: GenerationBundle) -> EvidencePackage:
    readme = _readme(bundle)
    start = readme.content.index("documentation")
    return create_evidence_package(
        bundle,
        (
            StructuredRecordEvidence(_release_record_id(bundle)),
            DocumentExcerpt(readme.document_id, start, start + len("documentation")),
        ),
    )


def _validator(contract: str) -> Draft202012Validator:
    return Draft202012Validator(
        {
            "$schema": SCHEMA["$schema"],
            "$defs": SCHEMA["$defs"],
            "$ref": f"#/$defs/{contract}",
        },
        format_checker=FormatChecker(),
    )


def test_document_and_release_record_evidence_is_exact_schema_compatible_and_immutable(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    bundle = _bundle(retrieval_config)
    package = _package(bundle)
    verified = verify_evidence_package(bundle, package)

    assert verified == package
    assert verified is not package
    assert package.generation_id == bundle.generation_id
    assert [record.evidence_id for record in package.records] == sorted(
        record.evidence_id for record in package.records
    )
    document = next(record for record in package.records if record.object_kind == "document")
    structured = next(
        record for record in package.records if record.object_kind == "structured_record"
    )
    expected_line = (
        "hash valkey-9.0.0.tar.gz sha256 "
        f"{'a' * 64} "
        "https://github.com/valkey-io/valkey/archive/refs/tags/9.0.0.tar.gz"
    )
    assert document.excerpt == "documentation"
    assert document.object_digest == next(
        item.digest for item in bundle.documents if item.object_id == document.object_id
    )
    assert structured.excerpt == expected_line
    assert structured.excerpt.encode() in bundle.structured_acquisitions[0].files[0].content
    assert structured.path == "README"
    assert document.immutable_url == (
        f"https://github.com/valkey-io/valkey/blob/{COMMIT}/README.md"
    )
    assert structured.immutable_url == (
        f"https://github.com/valkey-io/valkey-hashes/blob/{COMMIT}/README"
    )

    for record in package.records:
        value = evidence_record_value(record)
        encoded = canonical_evidence_record_bytes(record)
        assert json.loads(encoded) == value
        assert encoded == json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _validator("evidence_record").validate(value)

    with pytest.raises(FrozenInstanceError):
        package.digest = f"sha256:{'0' * 64}"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        package.records[0].excerpt = "changed"  # type: ignore[misc]


def test_evidence_identity_and_package_digest_are_deterministic_and_order_independent(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    bundle = _bundle(retrieval_config)
    readme = _readme(bundle)
    start = readme.content.index("documentation")
    selections = (
        DocumentExcerpt(readme.document_id, start, start + len("documentation")),
        StructuredRecordEvidence(_release_record_id(bundle)),
    )

    forward = create_evidence_package(bundle, selections)
    reverse = create_evidence_package(bundle, tuple(reversed(selections)))
    later_observation = _bundle(retrieval_config, created_at="2026-08-20T00:00:00Z")
    different_excerpt = create_evidence_package(
        bundle,
        (DocumentExcerpt(readme.document_id, 0, len("Valkey")),),
    )

    assert forward == reverse
    assert create_evidence_package(later_observation, selections) == forward
    assert different_excerpt.records[0].evidence_id not in {
        record.evidence_id for record in forward.records
    }
    assert different_excerpt.digest != forward.digest


def test_unicode_character_offsets_and_independent_byte_and_character_bounds(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    bundle = _bundle(retrieval_config, readme="Aé🙂Z\n".encode())
    document = _readme(bundle)
    exact = EvidenceLimits(max_excerpt_characters=2, max_excerpt_bytes=6)
    package = create_evidence_package(
        bundle,
        (DocumentExcerpt(document.document_id, 1, 3),),
        limits=exact,
    )

    assert package.records[0].excerpt == "é🙂"
    assert package.records[0].excerpt_digest == _digest("é🙂".encode())
    verify_evidence_package(bundle, package, limits=exact)
    with pytest.raises(EvidenceError, match="character/byte bound"):
        create_evidence_package(
            bundle,
            (DocumentExcerpt(document.document_id, 1, 3),),
            limits=replace(exact, max_excerpt_bytes=5),
        )
    with pytest.raises(EvidenceError, match="character/byte bound"):
        create_evidence_package(
            bundle,
            (DocumentExcerpt(document.document_id, 1, 3),),
            limits=replace(exact, max_excerpt_characters=1),
        )


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 1), (0, 0), (1, 999), (True, 2), (0, False)],
)
def test_excerpt_boundaries_fail_closed(
    retrieval_config: FrozenRetrievalConfiguration,
    start: int,
    end: int,
) -> None:
    bundle = _bundle(retrieval_config)
    document = _readme(bundle)
    with pytest.raises(EvidenceError, match="boundar"):
        create_evidence_package(
            bundle,
            (DocumentExcerpt(document.document_id, start, end),),
        )


def test_only_release_digest_records_support_direct_structured_evidence(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    inventory = _inventory()
    source = _source(inventory)
    bundle = _bundle(retrieval_config)
    release_id = _release_record_id(bundle)
    package = create_evidence_package(
        bundle,
        (StructuredRecordEvidence(release_id),),
    )
    evidence = package.records[0]
    assert evidence.path == "README"
    assert evidence.excerpt.startswith("hash valkey-9.0.0.tar.gz sha256 ")
    assert evidence.immutable_url.endswith(f"/{COMMIT}/README")

    direct_document_records: tuple[StructuredRecord, ...] = (
        PathRecord(source, PathIdentifier("valkey", "src/server.c")),
        SymbolRecord(
            source,
            SymbolIdentifier("valkey", "src/server.c", "processCommand"),
        ),
    )
    for record in direct_document_records:
        record_bundle = _bundle(retrieval_config, records=(record,))
        with pytest.raises(EvidenceError, match="use DocumentExcerpt"):
            create_evidence_package(
                record_bundle,
                (StructuredRecordEvidence(record_bundle.structured_records[0].object_id),),
            )

    pathless: tuple[StructuredRecord, ...] = (
        RepositoryRecord(source, RepositoryIdentifier("valkey")),
        CommitRecord(source, CommitIdentifier(COMMIT)),
        GitHubObjectRecord(
            source,
            GitHubObjectIdentifier("valkey", "pull_request", 4073),
        ),
        CommandRecord(source, CommandIdentifier("GET")),
    )
    for record in pathless:
        record_bundle = _bundle(retrieval_config, records=(record,))
        with pytest.raises(EvidenceError, match="no reviewed repository path"):
            create_evidence_package(
                record_bundle,
                (StructuredRecordEvidence(record_bundle.structured_records[0].object_id),),
            )


def test_arbitrary_digest_nonexistent_provenance_and_tampered_acquisition_are_rejected(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    bundle = _bundle(retrieval_config)
    release_index = next(
        index
        for index, record in enumerate(bundle.structured_record_templates)
        if isinstance(record, ReleaseArtifactDigestRecord)
    )
    release = cast(ReleaseArtifactDigestRecord, bundle.structured_record_templates[release_index])
    selection = (StructuredRecordEvidence(_release_record_id(bundle)),)

    for forged in (
        replace(release, digest=f"sha256:{'b' * 64}"),
        replace(release, provenance_path="releases/nonexistent.sha256"),
    ):
        templates = list(bundle.structured_record_templates)
        templates[release_index] = forged
        with pytest.raises(EvidenceError, match="not verified"):
            create_evidence_package(
                replace(bundle, structured_record_templates=tuple(templates)),
                selection,
            )

    acquisition = bundle.structured_acquisitions[0]
    tampered_file = replace(
        acquisition.files[0],
        content=acquisition.files[0].content.replace(b"a" * 64, b"b" * 64),
    )
    tampered_acquisition = replace(acquisition, files=(tampered_file,))
    with pytest.raises(EvidenceError, match="not verified"):
        create_evidence_package(
            replace(bundle, structured_acquisitions=(tampered_acquisition,)),
            selection,
        )


def test_unknown_duplicate_and_malformed_selections_are_rejected(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    bundle = _bundle(retrieval_config)
    readme = _readme(bundle)
    selection = DocumentExcerpt(readme.document_id, 0, 1)

    with pytest.raises(EvidenceError, match="unknown document ID"):
        create_evidence_package(bundle, (DocumentExcerpt("missing", 0, 1),))
    with pytest.raises(EvidenceError, match="unknown structured record ID"):
        create_evidence_package(bundle, (StructuredRecordEvidence("missing"),))
    with pytest.raises(EvidenceError, match="duplicate evidence"):
        create_evidence_package(bundle, (selection, selection))
    with pytest.raises(EvidenceError, match="unsupported evidence selection"):
        create_evidence_package(bundle, cast(tuple[DocumentExcerpt, ...], (object(),)))
    with pytest.raises(EvidenceError, match="immutable tuple"):
        create_evidence_package(bundle, cast(tuple[DocumentExcerpt, ...], [selection]))
    with pytest.raises(EvidenceError, match="outside its bound"):
        create_evidence_package(bundle, ())


def test_only_a_fully_verified_generation_bundle_is_accepted(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    bundle = _bundle(retrieval_config)
    readme = _readme(bundle)
    selection = (DocumentExcerpt(readme.document_id, 0, 1),)
    tampered_object = replace(bundle.documents[0], digest=f"sha256:{'0' * 64}")

    with pytest.raises(EvidenceError, match="not verified"):
        create_evidence_package(
            replace(bundle, documents=(tampered_object, *bundle.documents[1:])),
            selection,
        )
    with pytest.raises(EvidenceError, match="not verified"):
        create_evidence_package(cast(GenerationBundle, object()), selection)


def test_tampered_evidence_package_fields_and_mixed_generations_are_rejected(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    bundle = _bundle(retrieval_config)
    package = _package(bundle)
    record = package.records[0]
    mutations = (
        replace(record, object_id="unknown"),
        replace(record, object_digest=f"sha256:{'0' * 64}"),
        replace(record, path="src/other.c"),
        replace(record, commit=OTHER_COMMIT),
        replace(record, authority="secondary"),
        replace(record, version_scope="other"),
        replace(record, immutable_url="https://example.invalid/model-link"),
        replace(record, excerpt=record.excerpt + "x", excerpt_end=record.excerpt_end + 1),
        replace(record, excerpt_digest=f"sha256:{'0' * 64}"),
        replace(record, evidence_id="ev_forged"),
    )
    for mutation in mutations:
        malformed = replace(package, records=(mutation, *package.records[1:]))
        with pytest.raises(EvidenceError):
            verify_evidence_package(bundle, malformed)

    with pytest.raises(EvidenceError, match="digest"):
        verify_evidence_package(bundle, replace(package, digest=f"sha256:{'0' * 64}"))
    with pytest.raises(EvidenceError, match="generation"):
        verify_evidence_package(
            _bundle(retrieval_config, commit=OTHER_COMMIT),
            package,
        )


def test_claim_support_requires_known_unique_evidence_for_every_bounded_claim(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    bundle = _bundle(retrieval_config)
    package = _package(bundle)
    evidence_ids = tuple(record.evidence_id for record in package.records)
    supports = (
        ClaimSupport("claim.z", tuple(reversed(evidence_ids))),
        ClaimSupport("claim.a", (evidence_ids[0],)),
    )

    canonical = validate_claim_support(bundle, package, supports)
    assert [support.claim_id for support in canonical] == ["claim.a", "claim.z"]
    assert canonical[1].evidence_ids == tuple(sorted(evidence_ids))
    # Coverage is intentionally structural: this API does not judge entailment.
    assert validate_claim_support(
        bundle,
        package,
        (ClaimSupport("claim.not-semantically-checked", (evidence_ids[0],)),),
    )

    invalid = (
        (),
        (ClaimSupport("claim", ()),),
        (ClaimSupport("claim", ("ev_unknown",)),),
        (ClaimSupport("claim", (evidence_ids[0], evidence_ids[0])),),
        (ClaimSupport("claim", (evidence_ids[0],)), ClaimSupport("claim", (evidence_ids[1],))),
        (ClaimSupport("bad claim", (evidence_ids[0],)),),
    )
    for value in invalid:
        with pytest.raises(EvidenceError):
            validate_claim_support(bundle, package, value)
    with pytest.raises(EvidenceError, match="immutable tuple"):
        validate_claim_support(
            bundle,
            package,
            cast(tuple[ClaimSupport, ...], [ClaimSupport("claim", (evidence_ids[0],))]),
        )
    with pytest.raises(EvidenceError, match="IDs only"):
        validate_claim_support(
            bundle,
            package,
            (ClaimSupport("claim", cast(tuple[str, ...], ({"url": "model"},))),),
        )


def test_claim_and_evidence_limits_are_inclusive_and_malformed_limits_fail(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    bundle = _bundle(retrieval_config)
    document = _readme(bundle)
    selection = DocumentExcerpt(document.document_id, 0, 1)
    limits = EvidenceLimits(
        max_evidence_records=1,
        max_excerpt_characters=1,
        max_excerpt_bytes=1,
        max_claims=1,
        max_claim_id_bytes=1,
        max_evidence_ids_per_claim=1,
    )
    package = create_evidence_package(bundle, (selection,), limits=limits)
    evidence_id = package.records[0].evidence_id
    assert validate_claim_support(
        bundle,
        package,
        (ClaimSupport("a", (evidence_id,)),),
        limits=limits,
    ) == (ClaimSupport("a", (evidence_id,)),)

    with pytest.raises(EvidenceError, match="byte bound"):
        validate_claim_support(
            bundle,
            package,
            (ClaimSupport("ab", (evidence_id,)),),
            limits=limits,
        )
    with pytest.raises(EvidenceError, match="outside its bound"):
        create_evidence_package(bundle, (selection, selection), limits=limits)
    for malformed in (
        replace(limits, max_claims=0),
        replace(limits, max_claims=cast(int, True)),
        cast(EvidenceLimits, object()),
    ):
        with pytest.raises(EvidenceError, match="bound|runtime type"):
            create_evidence_package(bundle, (selection,), limits=malformed)


def test_citation_rendering_is_id_only_canonical_and_application_owned(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    bundle = _bundle(retrieval_config)
    package = _package(bundle)
    evidence_ids = tuple(record.evidence_id for record in reversed(package.records))
    expected = tuple(
        f"[{record.repository}/{record.path}@{record.commit[:12]}]({record.immutable_url})"
        for record in package.records
    )

    assert render_citations(bundle, package, evidence_ids) == expected
    with pytest.raises(EvidenceError, match="unknown"):
        render_citations(bundle, package, ("ev_unknown",))
    with pytest.raises(EvidenceError, match="duplicated"):
        render_citations(bundle, package, (evidence_ids[0], evidence_ids[0]))
    with pytest.raises(EvidenceError, match="IDs only"):
        render_citations(
            bundle,
            package,
            cast(tuple[str, ...], ({"evidence_id": evidence_ids[0], "url": "model"},)),
        )
    with pytest.raises(EvidenceError, match="non-empty immutable tuple"):
        render_citations(bundle, package, cast(tuple[str, ...], list(evidence_ids)))
    with pytest.raises(EvidenceError, match="digest"):
        render_citations(
            bundle,
            replace(package, digest=f"sha256:{'0' * 64}"),
            (evidence_ids[0],),
        )


def test_evidence_record_serialization_rejects_internal_tampering(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    record = _package(_bundle(retrieval_config)).records[0]
    malformed = replace(record, immutable_url="https://example.invalid")
    with pytest.raises(EvidenceError, match="immutable URL"):
        evidence_record_value(malformed)
    with pytest.raises(EvidenceError, match="immutable URL"):
        canonical_evidence_record_bytes(malformed)


def test_evidence_module_is_pure_local_and_has_no_network_or_filesystem_writes() -> None:
    source = (ROOT / "src/valkeyrie/evidence.py").read_text(encoding="utf-8")
    for prohibited in (
        "boto3",
        "botocore",
        "requests",
        "httpx",
        "urllib.request",
        "open(",
        "write_text",
        "write_bytes",
    ):
        assert prohibited not in source
