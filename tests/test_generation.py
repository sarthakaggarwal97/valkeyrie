from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from tests.helpers import load_yaml
from valkeyrie.acquisition import AcquiredFile, AcquiredRepository
from valkeyrie.generation import (
    CanonicalObject,
    GenerationBundle,
    GenerationError,
    GenerationLimits,
    canonical_generation_preimage,
    create_generation_bundle,
    verify_generation_bundle,
)
from valkeyrie.normalization import NormalizedDocument, normalize_repository
from valkeyrie.retrieval_config import FrozenRetrievalConfiguration, load_retrieval_config
from valkeyrie.revisions import Authority, ResolvedRevision
from valkeyrie.sources import load_source_inventory
from valkeyrie.structured import (
    CommandIdentifier,
    CommandRecord,
    PathIdentifier,
    PathRecord,
    StructuredRecord,
    SymbolIdentifier,
    SymbolRecord,
    build_release_artifact_digest_records,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "sources.yaml"
CONFIG = ROOT / "retrieval-config.yaml"
SCHEMA = cast(dict[str, Any], load_yaml(ROOT / "src/valkeyrie/schemas/contracts.schema.json"))
CREATED_AT = "2026-08-18T23:53:20.319Z"
COMMIT = "1" * 40
OTHER_COMMIT = "2" * 40
ARTIFACT_DIGEST = f"sha256:{'a' * 64}"
DEFAULT_LIMITS = GenerationLimits()


@pytest.fixture(scope="module")
def retrieval_config() -> FrozenRetrievalConfiguration:
    return load_retrieval_config(CONFIG)


def _digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _inventory() -> dict[str, object]:
    return deepcopy(load_source_inventory(SOURCES))


def _policy_digest(inventory: dict[str, object]) -> str:
    encoded = json.dumps(
        inventory,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _digest(encoded)


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
    server: bytes = b"int processCommand(void) { return 1; }\n",
) -> tuple[NormalizedDocument, ...]:
    source = _source(inventory, commit=commit)
    files = (
        AcquiredFile("README.md", readme),
        AcquiredFile("src/server.c", server),
    )
    acquired = AcquiredRepository(
        repository="valkey",
        commit=commit,
        files=files,
        total_bytes=sum(len(item.content) for item in files),
    )
    return normalize_repository(inventory, source, acquired)


def _hash_acquisition(
    *,
    commit: str = COMMIT,
    digest: str = ARTIFACT_DIGEST,
    auxiliary: bytes = b"Human release documentation.\n",
) -> AcquiredRepository:
    line = (
        "hash valkey-9.0.0.tar.gz sha256 "
        f"{digest.removeprefix('sha256:')} "
        "https://github.com/valkey-io/valkey/archive/refs/tags/9.0.0.tar.gz\n"
    ).encode()
    files = (
        AcquiredFile("README", line),
        AcquiredFile("README.md", auxiliary),
    )
    return AcquiredRepository(
        "valkey-hashes",
        commit,
        files,
        sum(len(item.content) for item in files),
    )


def _records(
    inventory: dict[str, object],
    *,
    commit: str = COMMIT,
    command: str = "GET",
    acquisition: AcquiredRepository | None = None,
) -> tuple[StructuredRecord, ...]:
    source = _source(inventory, commit=commit)
    hashes = _source(inventory, "valkey-hashes", commit=commit)
    acquired = acquisition or _hash_acquisition(commit=commit)
    return (
        PathRecord(source, PathIdentifier("valkey", "src/server.c")),
        CommandRecord(source, CommandIdentifier(command)),
        *build_release_artifact_digest_records(acquired, hashes),
    )


def _bundle(
    retrieval_config: FrozenRetrievalConfiguration,
    *,
    sources_yaml: bytes | None = None,
    inventory: dict[str, object] | None = None,
    commit: str = COMMIT,
    readme: bytes = b"Valkey documentation\n",
    command: str = "GET",
    created_at: str = CREATED_AT,
    limits: GenerationLimits = DEFAULT_LIMITS,
    acquisition: AcquiredRepository | None = None,
) -> GenerationBundle:
    source_inventory = inventory or _inventory()
    source_bytes = SOURCES.read_bytes() if sources_yaml is None else sources_yaml
    acquired = acquisition or _hash_acquisition(commit=commit)
    return create_generation_bundle(
        source_bytes,
        _documents(source_inventory, commit=commit, readme=readme),
        _records(
            source_inventory,
            commit=commit,
            command=command,
            acquisition=acquired,
        ),
        retrieval_config,
        created_at=created_at,
        structured_acquisitions=(acquired,),
        limits=limits,
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


def test_generation_preimage_is_complete_deterministic_and_generation_free(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    inventory = _inventory()
    documents = _documents(inventory)
    records = _records(inventory)
    acquisition = _hash_acquisition()
    sources = SOURCES.read_bytes()

    forward = canonical_generation_preimage(
        sources,
        documents,
        records,
        retrieval_config,
        structured_acquisitions=(acquisition,),
    )
    reverse = canonical_generation_preimage(
        sources,
        tuple(reversed(documents)),
        tuple(reversed(records)),
        retrieval_config,
        structured_acquisitions=(acquisition,),
    )
    value = json.loads(forward)

    assert forward == reverse
    assert value["api_version"] == "valkeyrie.io/generation-preimage/1"
    assert value["kind"] == "GenerationPreimage"
    assert base64.b64decode(value["sources_yaml_base64"]) == sources
    assert [item["document"]["document_id"] for item in value["documents"]] == sorted(
        document.document_id for document in documents
    )
    assert [item["record_id"] for item in value["structured_records"]] == sorted(
        [
            "command:get",
            "path:valkey:src%2Fserver.c",
            "release_artifact_digest:9.0.0:valkey-9.0.0.tar.gz",
        ]
    )
    assert value["retrieval_configuration"] == {
        "config_revision": retrieval_config.config_revision,
        "selected_candidate": "titan-v2-1024-fixed-300-20",
        "embedding": {
            "model_id": "amazon.titan-embed-text-v2:0",
            "dimensions": 1024,
            "output_normalization": "model_defined",
        },
        "chunking": {
            "strategy": "FIXED_SIZE",
            "max_tokens": 300,
            "overlap_percentage": 20,
        },
        "index": {
            "dimensions": 1024,
            "engine": "faiss",
            "algorithm": "hnsw",
            "distance_metric": "l2",
            "vector_field": "bedrock-knowledge-base-default-vector",
            "text_field": "AMAZON_BEDROCK_TEXT_CHUNK",
            "metadata_field": "AMAZON_BEDROCK_METADATA",
        },
        "retrieval": {
            "search_type": "HYBRID",
            "number_of_results": 10,
            "generation_filter_field": "generation_id",
            "exact_identifier_route": "deterministic_exact_lookup",
            "unavailable_generation_behavior": "fail_closed",
            "reranking": False,
        },
    }
    assert b'"generation_id":' not in forward
    assert b'"metadata_digest":' not in forward
    assert b'"record_digest":' not in forward
    assert b'"created_at":' not in forward
    assert b"GenerationManifest" not in forward


def test_exact_sources_bytes_and_every_canonical_input_change_generation_identity(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    baseline = _bundle(retrieval_config)
    sources_with_comment = SOURCES.read_bytes() + b"\n# exact-byte identity mutation\n"
    changed_sources = _bundle(retrieval_config, sources_yaml=sources_with_comment)
    changed_content = _bundle(retrieval_config, readme=b"Different documentation\n")
    changed_commit = _bundle(retrieval_config, commit=OTHER_COMMIT)
    changed_record = _bundle(retrieval_config, command="SET")

    variants = (changed_sources, changed_content, changed_commit, changed_record)
    assert all(item.generation_id != baseline.generation_id for item in variants)
    assert changed_sources.document_templates == baseline.document_templates
    assert changed_sources.structured_record_templates == baseline.structured_record_templates


def test_forged_runtime_configuration_with_genuine_revision_is_rejected_on_create_and_verify(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    forged_selected = replace(
        retrieval_config.selected,
        embedding=replace(
            retrieval_config.selected.embedding,
            model_id="forged.model:0",
        ),
    )
    forged = replace(retrieval_config, selected=forged_selected)
    with pytest.raises(GenerationError, match="not authoritative"):
        _bundle(forged)

    bundle = _bundle(retrieval_config)
    with pytest.raises(GenerationError, match="not authoritative"):
        verify_generation_bundle(replace(bundle, retrieval_config=forged))


def test_created_at_is_observation_metadata_not_generation_identity(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    first = _bundle(retrieval_config, created_at="2026-08-18T23:53:20Z")
    second = _bundle(retrieval_config, created_at="2026-08-19T00:00:00Z")

    assert first.generation_id == second.generation_id
    assert first.documents == second.documents
    assert first.metadata_sidecars == second.metadata_sidecars
    assert first.structured_records == second.structured_records
    assert first.manifest.content != second.manifest.content
    assert first.manifest.digest != second.manifest.digest
    assert json.loads(first.manifest.content)["created_at"] == "2026-08-18T23:53:20Z"


def test_final_objects_manifest_and_bedrock_sidecars_are_exact_and_keyed(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    bundle = _bundle(retrieval_config)
    manifest = json.loads(bundle.manifest.content)

    assert [item.object_id for item in bundle.documents] == sorted(
        item.object_id for item in bundle.documents
    )
    assert [item.object_id for item in bundle.structured_records] == sorted(
        item.object_id for item in bundle.structured_records
    )
    assert bundle.manifest.object_id == bundle.generation_id
    assert bundle.manifest.object_key == (
        f"control/generations/{bundle.generation_id.removeprefix('sha256:')}/manifest.json"
    )
    assert bundle.manifest.digest == _digest(bundle.manifest.content)
    assert manifest == {
        "api_version": "valkeyrie.io/generation-manifest/1",
        "kind": "GenerationManifest",
        "generation_id": bundle.generation_id,
        "sources_revision": _digest(SOURCES.read_bytes()),
        "retrieval_config_revision": retrieval_config.config_revision,
        "created_at": CREATED_AT,
        "documents": [
            {
                "document_id": document.object_id,
                "content_digest": document.digest,
                "metadata_digest": sidecar.digest,
            }
            for document, sidecar in zip(bundle.documents, bundle.metadata_sidecars, strict=True)
        ],
        "structured_records": [
            {"record_id": record.object_id, "record_digest": record.digest}
            for record in bundle.structured_records
        ],
    }
    _validator("generation_manifest").validate(manifest)

    for document, sidecar in zip(bundle.documents, bundle.metadata_sidecars, strict=True):
        sidecar_value = json.loads(sidecar.content)
        attributes = sidecar_value["metadataAttributes"]
        assert set(sidecar_value) == {"metadataAttributes"}
        assert sidecar.object_key == f"{document.object_key}.metadata.json"
        # The published body is the document text itself. It used to be the canonical JSON
        # envelope, which meant Bedrock embedded `{"api_version":...,"content":"` on every
        # document and saw the body JSON-escaped. Provenance belongs in the sidecar, which
        # Bedrock reads as metadata and never embeds.
        body = document.content.decode("utf-8")
        assert "valkeyrie.io/normalized-document" not in body
        assert not body.startswith("{")
        suffix = "md" if attributes["content_type"] == "text/markdown" else "txt"
        assert document.object_key == (
            f"documents/{document.object_id.removeprefix('sha256:')}.{suffix}"
        )
        assert document.digest == _digest(document.content)
        assert sidecar.digest == _digest(sidecar.content)
        assert attributes["generation_id"] == bundle.generation_id
        assert attributes["document_id"] == document.object_id
        assert attributes["repository"] == "valkey"
        assert attributes["commit"] == COMMIT
        assert (
            sidecar.content
            == json.dumps(
                {"metadataAttributes": attributes},
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )

    for record in bundle.structured_records:
        value = json.loads(record.content)
        assert record.object_key == f"structured/{quote(record.object_id, safe='')}.json"
        assert {
            "api_version",
            "kind",
            "record_id",
            "record_type",
            "source",
            "identifier",
            "generation_id",
        } <= set(value)
        assert set(value) <= {
            "api_version",
            "kind",
            "record_id",
            "record_type",
            "source",
            "identifier",
            "provenance",
            "value",
            "generation_id",
        }
        assert value["generation_id"] == bundle.generation_id
        assert value["record_id"] == record.object_id
        assert record.digest == _digest(record.content)

    keys = [
        *(item.object_key for item in bundle.documents),
        *(item.object_key for item in bundle.metadata_sidecars),
        *(item.object_key for item in bundle.structured_records),
        bundle.manifest.object_key,
    ]
    assert len(keys) == len(set(keys))


def test_verification_reconstructs_and_returns_an_equal_immutable_bundle(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    bundle = _bundle(retrieval_config)
    verified = verify_generation_bundle(bundle)

    assert verified == bundle
    assert verified is not bundle
    with pytest.raises(FrozenInstanceError):
        verified.generation_id = "sha256:" + "0" * 64  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        verified.manifest.content = b"changed"  # type: ignore[misc]


def test_verification_rejects_missing_extra_duplicate_and_wrong_ordered_objects(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    bundle = _bundle(retrieval_config)
    mutations = (
        replace(bundle, document_templates=tuple(reversed(bundle.document_templates))),
        replace(
            bundle,
            structured_record_templates=tuple(reversed(bundle.structured_record_templates)),
        ),
        replace(bundle, documents=bundle.documents[:-1]),
        replace(bundle, documents=bundle.documents + (bundle.documents[0],)),
        replace(bundle, documents=tuple(reversed(bundle.documents))),
        replace(bundle, metadata_sidecars=bundle.metadata_sidecars[:-1]),
        replace(bundle, metadata_sidecars=tuple(reversed(bundle.metadata_sidecars))),
        replace(bundle, structured_records=bundle.structured_records[:-1]),
        replace(
            bundle,
            structured_records=bundle.structured_records + (bundle.structured_records[0],),
        ),
        replace(bundle, structured_records=tuple(reversed(bundle.structured_records))),
    )
    for mutation in mutations:
        with pytest.raises(GenerationError):
            verify_generation_bundle(mutation)


def test_verification_rejects_every_output_and_identity_tamper(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    bundle = _bundle(retrieval_config)
    changed_document = replace(
        bundle.documents[0],
        content=bundle.documents[0].content + b" ",
    )
    changed_sidecar = replace(
        bundle.metadata_sidecars[0],
        digest=f"sha256:{'0' * 64}",
    )
    changed_record = replace(
        bundle.structured_records[0],
        content=bundle.structured_records[0].content + b" ",
    )
    changed_manifest_bytes = bundle.manifest.content + b" "
    mutations = (
        replace(bundle, generation_id=f"sha256:{'0' * 64}"),
        replace(bundle, sources_yaml=bundle.sources_yaml + b"\n# tamper\n"),
        replace(
            bundle,
            structured_acquisitions=(
                replace(
                    bundle.structured_acquisitions[0],
                    total_bytes=bundle.structured_acquisitions[0].total_bytes + 1,
                ),
            ),
        ),
        replace(
            bundle,
            documents=(
                replace(bundle.documents[0], object_key="documents/forged.json"),
                *bundle.documents[1:],
            ),
        ),
        replace(bundle, documents=(changed_document, *bundle.documents[1:])),
        replace(
            bundle,
            metadata_sidecars=(changed_sidecar, *bundle.metadata_sidecars[1:]),
        ),
        replace(
            bundle,
            structured_records=(changed_record, *bundle.structured_records[1:]),
        ),
        replace(
            bundle,
            manifest=replace(bundle.manifest, content=changed_manifest_bytes),
        ),
        replace(
            bundle,
            manifest=CanonicalObject(
                bundle.generation_id,
                bundle.manifest.object_key,
                changed_manifest_bytes,
                _digest(changed_manifest_bytes),
            ),
        ),
        replace(
            bundle,
            retrieval_config=replace(
                retrieval_config,
                config_revision=f"sha256:{'8' * 64}",
            ),
        ),
    )
    for mutation in mutations:
        with pytest.raises(GenerationError):
            verify_generation_bundle(mutation)


def test_rejects_duplicate_ids_inconsistent_documents_records_and_sources(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    inventory = _inventory()
    documents = _documents(inventory)
    records = _records(inventory)
    sources = SOURCES.read_bytes()

    with pytest.raises(GenerationError, match="duplicate document ID"):
        create_generation_bundle(
            sources,
            (documents[0], documents[0]),
            records,
            retrieval_config,
            created_at=CREATED_AT,
        )
    with pytest.raises(GenerationError, match="duplicate exact identifier"):
        create_generation_bundle(
            sources,
            documents,
            (records[0], records[0]),
            retrieval_config,
            created_at=CREATED_AT,
        )

    inconsistent_document = replace(documents[0], content="tampered\n")
    with pytest.raises(GenerationError, match="invalid normalized document"):
        create_generation_bundle(
            sources,
            (inconsistent_document,),
            records,
            retrieval_config,
            created_at=CREATED_AT,
        )

    other_source_record = CommandRecord(
        _source(inventory, commit=OTHER_COMMIT),
        CommandIdentifier("SET"),
    )
    with pytest.raises(GenerationError, match="inconsistent revisions"):
        create_generation_bundle(
            sources,
            documents,
            (other_source_record,),
            retrieval_config,
            created_at=CREATED_AT,
        )

    mismatched_wrapper = CommandRecord(
        _source(inventory),
        cast(CommandIdentifier, PathIdentifier("valkey", "README.md")),
    )
    with pytest.raises(GenerationError, match="record and identifier types"):
        create_generation_bundle(
            sources,
            documents,
            (mismatched_wrapper,),
            retrieval_config,
            created_at=CREATED_AT,
        )


def test_structured_acquisitions_reconstruct_exact_records_and_provenance(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    inventory = _inventory()
    documents = _documents(inventory)
    acquisition = _hash_acquisition()
    hashes = _source(inventory, "valkey-hashes")
    release_records = build_release_artifact_digest_records(acquisition, hashes)

    bundle = create_generation_bundle(
        SOURCES.read_bytes(),
        documents,
        release_records,
        retrieval_config,
        created_at=CREATED_AT,
        structured_acquisitions=(acquisition,),
    )
    assert bundle.structured_acquisitions == (acquisition,)

    with pytest.raises(GenerationError, match="missing their exact structured acquisition"):
        create_generation_bundle(
            SOURCES.read_bytes(),
            documents,
            release_records,
            retrieval_config,
            created_at=CREATED_AT,
        )
    forged_digest = replace(release_records[0], digest=f"sha256:{'b' * 64}")
    with pytest.raises(GenerationError, match="do not exactly match"):
        create_generation_bundle(
            SOURCES.read_bytes(),
            documents,
            (forged_digest,),
            retrieval_config,
            created_at=CREATED_AT,
            structured_acquisitions=(acquisition,),
        )
    tampered_file = replace(
        acquisition.files[0],
        content=acquisition.files[0].content.replace(b"a" * 64, b"b" * 64),
    )
    tampered = replace(
        acquisition,
        files=(tampered_file, *acquisition.files[1:]),
        total_bytes=acquisition.total_bytes,
    )
    with pytest.raises(GenerationError, match="do not exactly match"):
        create_generation_bundle(
            SOURCES.read_bytes(),
            documents,
            release_records,
            retrieval_config,
            created_at=CREATED_AT,
            structured_acquisitions=(tampered,),
        )
    with pytest.raises(GenerationError, match="repeat an exact repository revision"):
        create_generation_bundle(
            SOURCES.read_bytes(),
            documents,
            release_records,
            retrieval_config,
            created_at=CREATED_AT,
            structured_acquisitions=(acquisition, acquisition),
        )


def test_path_and_symbol_records_require_exact_normalized_document_provenance(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    inventory = _inventory()
    source = _source(inventory)
    documents = _documents(inventory)
    symbol = SymbolRecord(
        source,
        SymbolIdentifier("valkey", "src/server.c", "processCommand"),
    )
    assert create_generation_bundle(
        SOURCES.read_bytes(),
        documents,
        (symbol,),
        retrieval_config,
        created_at=CREATED_AT,
    )

    missing_path = PathRecord(source, PathIdentifier("valkey", "src/missing.c"))
    with pytest.raises(GenerationError, match="exactly one normalized document"):
        create_generation_bundle(
            SOURCES.read_bytes(),
            documents,
            (missing_path,),
            retrieval_config,
            created_at=CREATED_AT,
        )
    missing_symbol = replace(
        symbol,
        identifier=SymbolIdentifier("valkey", "src/server.c", "missingSymbol"),
    )
    with pytest.raises(GenerationError, match="does not occur"):
        create_generation_bundle(
            SOURCES.read_bytes(),
            documents,
            (missing_symbol,),
            retrieval_config,
            created_at=CREATED_AT,
        )


def test_unparsed_auxiliary_structured_bytes_do_not_change_generation_identity(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    first = _bundle(retrieval_config, acquisition=_hash_acquisition(auxiliary=b"first\n"))
    second = _bundle(retrieval_config, acquisition=_hash_acquisition(auxiliary=b"second\n"))

    assert first.structured_acquisitions != second.structured_acquisitions
    assert first.generation_id == second.generation_id
    assert first.documents == second.documents
    assert first.metadata_sidecars == second.metadata_sidecars
    assert first.structured_records == second.structured_records


def test_rejects_source_semantic_and_retrieval_configuration_mismatch(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    inventory = _inventory()
    documents = _documents(inventory)
    records = _records(inventory)

    with pytest.raises(GenerationError, match="sources.yaml semantics"):
        create_generation_bundle(
            SOURCES.read_bytes().replace(
                b"active_nonfork_repositories: 46",
                b"active_nonfork_repositories: 45",
            ),
            documents,
            records,
            retrieval_config,
            created_at=CREATED_AT,
        )

    stale_source = replace(documents[0].source, version_scope="other_scope")
    with pytest.raises(GenerationError, match="invalid normalized document|conflicts"):
        create_generation_bundle(
            SOURCES.read_bytes(),
            (replace(documents[0], source=stale_source),),
            records,
            retrieval_config,
            created_at=CREATED_AT,
        )

    other_candidate = next(
        candidate
        for candidate in retrieval_config.candidates
        if candidate.candidate_id != retrieval_config.selected_candidate
    )
    wrong_selected = replace(
        retrieval_config,
        selected=other_candidate.configuration,
    )
    with pytest.raises(GenerationError, match="not authoritative"):
        create_generation_bundle(
            SOURCES.read_bytes(),
            documents,
            records,
            wrong_selected,
            created_at=CREATED_AT,
        )


def test_rejects_wrong_runtime_types_malformed_timestamps_and_digests(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    inventory = _inventory()
    documents = _documents(inventory)
    records = _records(inventory)
    sources = SOURCES.read_bytes()

    bad_calls = (
        (cast(bytes, "not-bytes"), documents, records, retrieval_config),
        (sources, cast(tuple[NormalizedDocument, ...], list(documents)), records, retrieval_config),
        (sources, documents, cast(tuple[StructuredRecord, ...], list(records)), retrieval_config),
        (
            sources,
            documents,
            records,
            cast(FrozenRetrievalConfiguration, object()),
        ),
    )
    for source_value, document_value, record_value, config_value in bad_calls:
        with pytest.raises(GenerationError):
            create_generation_bundle(
                source_value,
                document_value,
                record_value,
                config_value,
                created_at=CREATED_AT,
            )

    for timestamp in (
        cast(str, object()),
        "2026-08-18 23:53:20Z",
        "2026-08-18T23:53:20+00:00",
        "2026-02-30T00:00:00Z",
    ):
        with pytest.raises(GenerationError, match="created_at"):
            create_generation_bundle(
                sources,
                documents,
                records,
                retrieval_config,
                created_at=timestamp,
            )

    malformed_configs = (
        replace(retrieval_config, config_revision="sha256:short"),
        replace(retrieval_config, candidates=cast(Any, (object(),))),
        replace(
            retrieval_config,
            selected=replace(
                retrieval_config.selected,
                embedding=replace(
                    retrieval_config.selected.embedding,
                    dimensions=cast(int, True),
                ),
            ),
        ),
    )
    for malformed_config in malformed_configs:
        with pytest.raises(GenerationError, match="not authoritative"):
            create_generation_bundle(
                sources,
                documents,
                records,
                malformed_config,
                created_at=CREATED_AT,
            )

    with pytest.raises(GenerationError, match="wrong runtime type"):
        verify_generation_bundle(cast(GenerationBundle, object()))


def test_generation_bounds_are_inclusive_and_fail_above_each_edge(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    bundle = _bundle(retrieval_config)
    preimage = canonical_generation_preimage(
        bundle.sources_yaml,
        bundle.document_templates,
        bundle.structured_record_templates,
        retrieval_config,
        structured_acquisitions=bundle.structured_acquisitions,
    )
    exact = GenerationLimits(
        max_sources_bytes=len(bundle.sources_yaml),
        max_documents=len(bundle.document_templates),
        max_structured_records=len(bundle.structured_record_templates),
        max_structured_acquisitions=len(bundle.structured_acquisitions),
        max_object_bytes=max(
            len(item.content)
            for item in (
                *bundle.documents,
                *bundle.metadata_sidecars,
                *bundle.structured_records,
                bundle.manifest,
            )
        ),
        max_preimage_bytes=len(preimage),
        max_manifest_bytes=len(bundle.manifest.content),
    )
    assert _bundle(retrieval_config, limits=exact).generation_id == bundle.generation_id

    failing = (
        replace(exact, max_sources_bytes=len(bundle.sources_yaml) - 1),
        replace(exact, max_documents=len(bundle.document_templates) - 1),
        replace(
            exact,
            max_structured_records=len(bundle.structured_record_templates) - 1,
        ),
        replace(
            exact,
            max_structured_acquisitions=len(bundle.structured_acquisitions) - 1,
        ),
        replace(exact, max_object_bytes=1),
        replace(exact, max_preimage_bytes=len(preimage) - 1),
        replace(exact, max_manifest_bytes=len(bundle.manifest.content) - 1),
    )
    for limits in failing:
        with pytest.raises(GenerationError, match="bound"):
            _bundle(retrieval_config, limits=limits)

    for limits in (
        replace(exact, max_documents=0),
        replace(exact, max_documents=cast(int, True)),
        cast(GenerationLimits, object()),
    ):
        with pytest.raises(GenerationError, match="bound|runtime type"):
            _bundle(retrieval_config, limits=limits)


def test_generation_module_is_pure_local_and_has_no_network_or_filesystem_writes() -> None:
    source = (ROOT / "src/valkeyrie/generation.py").read_text(encoding="utf-8")
    for prohibited in (
        "boto3",
        "botocore",
        "requests",
        "urllib.request",
        "open(",
        "write_text",
        "write_bytes",
    ):
        assert prohibited not in source
