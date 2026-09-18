"""Canonical generation identity, immutable objects, manifests, and verification."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Final, cast
from urllib.parse import quote

import yaml

from valkeyrie.acquisition import AcquiredRepository
from valkeyrie.normalization import (
    NormalizationError,
    NormalizedDocument,
    canonical_document_bytes,
    canonical_document_identity_bytes,
    canonical_metadata_identity_bytes,
)
from valkeyrie.retrieval_config import (
    ChunkingConfiguration,
    FrozenRetrievalConfiguration,
    RetrievalConfigError,
    validate_frozen_retrieval_configuration,
)
from valkeyrie.revisions import ResolvedRevision
from valkeyrie.sources import SourceInventoryError, _load_yaml_mapping, validate_source_inventory
from valkeyrie.structured import (
    ExactLookup,
    PathRecord,
    ReleaseArtifactDigestRecord,
    StructuredRecord,
    StructuredRecordError,
    SymbolRecord,
    build_release_artifact_digest_records,
    canonical_record_bytes,
)


class GenerationError(ValueError):
    """Generation inputs or emitted objects are invalid or inconsistent."""


@dataclass(frozen=True)
class GenerationLimits:
    """Hard local bounds for one canonical generation bundle."""

    max_sources_bytes: int = 1024 * 1024
    max_documents: int = 100_000
    max_structured_records: int = 1_000_000
    max_structured_acquisitions: int = 64
    max_object_bytes: int = 2 * 1024 * 1024
    max_preimage_bytes: int = 1024 * 1024 * 1024
    max_manifest_bytes: int = 256 * 1024 * 1024


@dataclass(frozen=True)
class CanonicalObject:
    """One immutable canonical object, relative key, and exact byte checksum."""

    object_id: str
    object_key: str
    content: bytes
    digest: str


@dataclass(frozen=True)
class GenerationBundle:
    """All identity inputs and immutable outputs for one generation."""

    generation_id: str
    created_at: str
    sources_yaml: bytes
    retrieval_config: FrozenRetrievalConfiguration
    document_templates: tuple[NormalizedDocument, ...]
    structured_record_templates: tuple[StructuredRecord, ...]
    structured_acquisitions: tuple[AcquiredRepository, ...]
    documents: tuple[CanonicalObject, ...]
    metadata_sidecars: tuple[CanonicalObject, ...]
    structured_records: tuple[CanonicalObject, ...]
    manifest: CanonicalObject


@dataclass(frozen=True)
class _CanonicalInputs:
    preimage: bytes
    documents: tuple[NormalizedDocument, ...]
    records: tuple[StructuredRecord, ...]
    acquisitions: tuple[AcquiredRepository, ...]


_DEFAULT_LIMITS = GenerationLimits()
_PREIMAGE_API_VERSION: Final = "valkeyrie.io/generation-preimage/1"
_MANIFEST_API_VERSION: Final = "valkeyrie.io/generation-manifest/1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_TIMESTAMP = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$")


def canonical_generation_preimage(
    sources_yaml: bytes,
    documents: tuple[NormalizedDocument, ...],
    structured_records: tuple[StructuredRecord, ...],
    retrieval_config: FrozenRetrievalConfiguration,
    *,
    structured_acquisitions: tuple[AcquiredRepository, ...] = (),
    limits: GenerationLimits = _DEFAULT_LIMITS,
) -> bytes:
    """Return the canonical generation preimage.

    Exact reviewed source bytes and every generation-free input are present.
    ``generation_id``, final object checksums, the manifest, and ``created_at``
    are deliberately absent, so no derived value hashes itself.
    """
    return _canonical_inputs(
        sources_yaml,
        documents,
        structured_records,
        retrieval_config,
        structured_acquisitions,
        limits,
    ).preimage


def create_generation_bundle(
    sources_yaml: bytes,
    documents: tuple[NormalizedDocument, ...],
    structured_records: tuple[StructuredRecord, ...],
    retrieval_config: FrozenRetrievalConfiguration,
    *,
    created_at: str,
    structured_acquisitions: tuple[AcquiredRepository, ...] = (),
    limits: GenerationLimits = _DEFAULT_LIMITS,
) -> GenerationBundle:
    """Create deterministic immutable objects and one canonical manifest."""
    _validate_timestamp(created_at)
    inputs = _canonical_inputs(
        sources_yaml,
        documents,
        structured_records,
        retrieval_config,
        structured_acquisitions,
        limits,
    )
    generation_id = _sha256(inputs.preimage)

    final_documents: list[CanonicalObject] = []
    sidecars: list[CanonicalObject] = []
    for document in inputs.documents:
        # The published body is the document text itself, not the canonical envelope. The
        # envelope is still derived below for identity and contract validation, so the
        # document ID is unchanged; only what Bedrock embeds differs.
        document_bytes = document.content.encode("utf-8")
        metadata = _decode_object(canonical_metadata_identity_bytes(document), "metadata template")
        metadata["generation_id"] = generation_id
        metadata_bytes = _canonical_json({"metadataAttributes": metadata})
        document_key = _document_object_key(document.document_id, document.content_type)
        final_documents.append(_object(document.document_id, document_key, document_bytes, limits))
        sidecars.append(
            _object(
                document.document_id,
                f"{document_key}.metadata.json",
                metadata_bytes,
                limits,
            )
        )

    final_records: list[CanonicalObject] = []
    for record in inputs.records:
        template = _decode_object(canonical_record_bytes(record), "structured record template")
        record_id = cast(str, template["record_id"])
        template["generation_id"] = generation_id
        final_records.append(
            _object(
                record_id,
                _structured_object_key(record_id),
                _canonical_json(template),
                limits,
            )
        )

    manifest_value: dict[str, object] = {
        "api_version": _MANIFEST_API_VERSION,
        "kind": "GenerationManifest",
        "generation_id": generation_id,
        "sources_revision": _sha256(sources_yaml),
        "retrieval_config_revision": retrieval_config.config_revision,
        "created_at": created_at,
        "documents": [
            {
                "document_id": document.object_id,
                "content_digest": document.digest,
                "metadata_digest": sidecar.digest,
            }
            for document, sidecar in zip(final_documents, sidecars, strict=True)
        ],
        "structured_records": [
            {"record_id": record.object_id, "record_digest": record.digest}
            for record in final_records
        ],
    }
    manifest_bytes = _canonical_json(manifest_value)
    if len(manifest_bytes) > limits.max_manifest_bytes:
        raise GenerationError("generation manifest exceeded its byte bound")
    manifest = CanonicalObject(
        generation_id,
        _control_object_key(generation_id),
        manifest_bytes,
        _sha256(manifest_bytes),
    )
    _validate_object(manifest, limits.max_manifest_bytes, "manifest")
    return GenerationBundle(
        generation_id=generation_id,
        created_at=created_at,
        sources_yaml=sources_yaml,
        retrieval_config=retrieval_config,
        document_templates=inputs.documents,
        structured_record_templates=inputs.records,
        structured_acquisitions=inputs.acquisitions,
        documents=tuple(final_documents),
        metadata_sidecars=tuple(sidecars),
        structured_records=tuple(final_records),
        manifest=manifest,
    )


def verify_generation_bundle(
    bundle: GenerationBundle,
    *,
    limits: GenerationLimits = _DEFAULT_LIMITS,
) -> GenerationBundle:
    """Reconstruct every identity input and return a verified immutable bundle.

    Observation time is validated and reconstructed in the manifest, but is not
    an identity input. Any missing, extra, duplicate, reordered, malformed, or
    mismatched object is rejected.
    """
    _validate_limits(limits)
    if not isinstance(bundle, GenerationBundle):
        raise GenerationError("generation bundle has the wrong runtime type")
    _validate_bundle_shape(bundle, limits)
    expected = create_generation_bundle(
        bundle.sources_yaml,
        bundle.document_templates,
        bundle.structured_record_templates,
        bundle.retrieval_config,
        created_at=bundle.created_at,
        structured_acquisitions=bundle.structured_acquisitions,
        limits=limits,
    )
    if bundle != expected:
        raise GenerationError("generation bundle does not match its reconstructed identity inputs")
    return expected


def is_ingestible_document(document: NormalizedDocument) -> bool:
    """Report whether the knowledge base can parse this document.

    Bedrock classifies content beginning with a shebang as an executable script and refuses it
    whatever the object key suffix says. Its own words for each of the 54 such documents in the
    published corpus were "Ignored 1 files as their file format was not supported", and every one
    of them is a shell or Python script whose first two bytes are these.

    A document the knowledge base refuses cannot be retrieved or cited, so publishing it buys
    nothing and costs the entire release: ingestion treats any failed document as fatal and marks
    the candidate non-retryable, which held the corpus frozen from 2026-09-11. Excluding them here,
    before the preimage is computed, keeps the generation identity, the manifest, and the published
    objects describing the same set of documents.
    """
    first_line = document.content.partition("\n")[0]
    if not first_line.startswith("#!"):
        return True
    # An interpreter path is what makes it a shebang. Requiring one was measured against the 54
    # documents Bedrock actually refused: matching "#!" alone also excluded seven that index
    # correctly, because "#!" opens a Rust inner attribute (#![allow(...)]) and a Valkey function
    # library header (#!js api_version=1.0), neither of which is an executable.
    return not first_line[2:].lstrip().startswith("/")


def _canonical_inputs(
    sources_yaml: bytes,
    documents: tuple[NormalizedDocument, ...],
    records: tuple[StructuredRecord, ...],
    retrieval_config: FrozenRetrievalConfiguration,
    structured_acquisitions: tuple[AcquiredRepository, ...],
    limits: GenerationLimits,
) -> _CanonicalInputs:
    _validate_limits(limits)
    inventory = _source_inventory(sources_yaml, limits)
    if not isinstance(documents, tuple):
        raise GenerationError("normalized documents must be an immutable tuple")
    if not isinstance(records, tuple):
        raise GenerationError("structured records must be an immutable tuple")
    # Applied before the bound and the preimage, so the count that is checked and the identity that
    # is derived both describe the documents actually published.
    documents = tuple(document for document in documents if is_ingestible_document(document))
    if not 1 <= len(documents) <= limits.max_documents:
        raise GenerationError("generation document count is outside its bound")
    if len(records) > limits.max_structured_records:
        raise GenerationError("generation structured-record count exceeded its bound")
    retrieval_value = _retrieval_template(retrieval_config)

    policy_digest = _sha256(_canonical_json(inventory))
    sources: dict[str, ResolvedRevision] = {}
    document_values: list[tuple[str, dict[str, object]]] = []
    documents_by_id: dict[str, NormalizedDocument] = {}
    for document in documents:
        try:
            identity = _decode_object(
                canonical_document_identity_bytes(document), "document identity"
            )
            final_document = _decode_object(canonical_document_bytes(document), "document")
            metadata = _decode_object(
                canonical_metadata_identity_bytes(document), "metadata template"
            )
        except NormalizationError as error:
            raise GenerationError(f"invalid normalized document: {error}") from error
        if document.document_id in documents_by_id:
            raise GenerationError(f"duplicate document ID: {document.document_id}")
        documents_by_id[document.document_id] = document
        _bind_source(inventory, document.source, policy_digest, sources, documents=True)
        document_values.append(
            (
                document.document_id,
                {"identity": identity, "document": final_document, "metadata": metadata},
            )
        )
    document_values.sort(key=lambda item: item[0])
    ordered_documents = tuple(documents_by_id[document_id] for document_id, _ in document_values)

    try:
        index = ExactLookup(records)
    except StructuredRecordError as error:
        raise GenerationError(f"invalid structured records: {error}") from error
    record_values: list[tuple[str, dict[str, object], StructuredRecord]] = []
    record_ids: set[str] = set()
    for record in index.records:
        try:
            template = _decode_object(canonical_record_bytes(record), "structured record template")
        except StructuredRecordError as error:
            raise GenerationError(f"invalid structured record: {error}") from error
        record_id = template.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise GenerationError("structured record has a malformed record ID")
        if record_id in record_ids:
            raise GenerationError(f"duplicate structured record ID: {record_id}")
        record_ids.add(record_id)
        _bind_source(inventory, record.source, policy_digest, sources, documents=False)
        record_values.append((record_id, template, record))
    record_values.sort(key=lambda item: item[0])
    ordered_records = tuple(item[2] for item in record_values)
    ordered_acquisitions = _validate_structured_provenance(
        inventory,
        ordered_documents,
        ordered_records,
        structured_acquisitions,
        policy_digest,
        sources,
        limits,
    )

    preimage = _canonical_json(
        {
            "api_version": _PREIMAGE_API_VERSION,
            "kind": "GenerationPreimage",
            "sources_yaml_base64": base64.b64encode(sources_yaml).decode("ascii"),
            "documents": [item[1] for item in document_values],
            "structured_records": [item[1] for item in record_values],
            "retrieval_configuration": retrieval_value,
        }
    )
    if len(preimage) > limits.max_preimage_bytes:
        raise GenerationError("generation preimage exceeded its byte bound")
    return _CanonicalInputs(preimage, ordered_documents, ordered_records, ordered_acquisitions)


def _source_inventory(sources_yaml: object, limits: GenerationLimits) -> dict[str, object]:
    if not isinstance(sources_yaml, bytes):
        raise GenerationError("sources.yaml content must be exact bytes")
    if not 1 <= len(sources_yaml) <= limits.max_sources_bytes:
        raise GenerationError("sources.yaml content is outside its byte bound")
    try:
        text = sources_yaml.decode("utf-8")
        inventory = _load_yaml_mapping(text, "sources.yaml bytes", reject_merge_keys=True)
        validate_source_inventory(inventory)
    except (UnicodeError, yaml.YAMLError, ValueError, TypeError, SourceInventoryError) as error:
        raise GenerationError(f"sources.yaml semantics are invalid: {error}") from error
    return inventory


def _bind_source(
    inventory: Mapping[str, object],
    source: ResolvedRevision,
    policy_digest: str,
    seen: dict[str, ResolvedRevision],
    *,
    documents: bool,
) -> None:
    if not isinstance(source, ResolvedRevision):
        raise GenerationError("source revision has the wrong runtime type")
    repositories = cast(list[dict[str, object]], inventory["repositories"])
    entry = next((item for item in repositories if item["name"] == source.repository), None)
    if entry is None:
        raise GenerationError(f"source {source.repository!r} is absent from sources.yaml")
    if documents and (
        entry["classification"] != "curated" or entry["ingestion_mode"] != "documents"
    ):
        raise GenerationError(f"source {source.repository!r} is not a reviewed document source")
    if not documents and entry["classification"] not in {"curated", "structured_exact"}:
        raise GenerationError(f"source {source.repository!r} is not a reviewed static source")
    expected = {
        "repository_url": entry["url"],
        "requested_ref": entry["requested_ref"],
        "authority": entry["authority"],
        "version_scope": entry["version_scope"],
        "source_policy_digest": policy_digest,
    }
    if any(getattr(source, field) != value for field, value in expected.items()):
        raise GenerationError(
            f"source {source.repository!r} revision metadata conflicts with sources.yaml"
        )
    previous = seen.get(source.repository)
    if previous is not None and previous != source:
        raise GenerationError(f"source {source.repository!r} has inconsistent revisions")
    seen[source.repository] = source


def _validate_structured_provenance(
    inventory: Mapping[str, object],
    documents: tuple[NormalizedDocument, ...],
    records: tuple[StructuredRecord, ...],
    acquisitions: object,
    policy_digest: str,
    seen_sources: dict[str, ResolvedRevision],
    limits: GenerationLimits,
) -> tuple[AcquiredRepository, ...]:
    if not isinstance(acquisitions, tuple):
        raise GenerationError("structured acquisitions must be an immutable tuple")
    if len(acquisitions) > limits.max_structured_acquisitions:
        raise GenerationError("structured acquisition count exceeded its bound")

    release_records: dict[tuple[str, str], list[ReleaseArtifactDigestRecord]] = {}
    release_sources: dict[tuple[str, str], ResolvedRevision] = {}
    for record in records:
        if isinstance(record, PathRecord | SymbolRecord):
            matches = [
                document
                for document in documents
                if document.source.repository == record.source.repository
                and document.source.commit == record.source.commit
                and document.path == record.provenance_path
            ]
            if len(matches) != 1:
                raise GenerationError(
                    "path-bearing structured record must match exactly one normalized document"
                )
            if (
                isinstance(record, SymbolRecord)
                and record.identifier.symbol not in matches[0].content
            ):
                raise GenerationError(
                    "structured symbol does not occur in its normalized provenance document"
                )
        elif isinstance(record, ReleaseArtifactDigestRecord):
            key = (record.source.repository, record.source.commit)
            release_records.setdefault(key, []).append(record)
            previous = release_sources.setdefault(key, record.source)
            if previous != record.source:
                raise GenerationError("structured records have inconsistent exact source metadata")

    indexed: dict[tuple[str, str], AcquiredRepository] = {}
    for acquisition in acquisitions:
        if not isinstance(acquisition, AcquiredRepository):
            raise GenerationError("structured acquisition has the wrong runtime type")
        key = (acquisition.repository, acquisition.commit)
        if key in indexed:
            raise GenerationError("structured acquisitions repeat an exact repository revision")
        source = release_sources.get(key)
        if source is None:
            raise GenerationError("structured acquisition has no matching release digest records")
        _bind_source(inventory, source, policy_digest, seen_sources, documents=False)
        try:
            reconstructed = build_release_artifact_digest_records(acquisition, source)
        except StructuredRecordError as error:
            raise GenerationError(f"invalid structured acquisition: {error}") from error
        if reconstructed != tuple(release_records[key]):
            raise GenerationError(
                "release digest records do not exactly match their structured acquisition"
            )
        indexed[key] = acquisition

    if set(indexed) != set(release_records):
        raise GenerationError(
            "release digest records are missing their exact structured acquisition"
        )
    return tuple(indexed[key] for key in sorted(indexed))


def _chunking_template(chunking: ChunkingConfiguration) -> dict[str, object]:
    if chunking.strategy == "HIERARCHICAL":
        return {
            "strategy": chunking.strategy,
            "max_tokens": chunking.max_tokens,
            "parent_max_tokens": chunking.parent_max_tokens,
            "overlap_tokens": chunking.overlap_tokens,
        }
    return {
        "strategy": chunking.strategy,
        "max_tokens": chunking.max_tokens,
        "overlap_percentage": chunking.overlap_percentage,
    }


def _retrieval_template(config: object) -> dict[str, object]:
    try:
        validated = validate_frozen_retrieval_configuration(
            cast(FrozenRetrievalConfiguration, config)
        )
        selected = validated.selected
        value: dict[str, object] = {
            "config_revision": validated.config_revision,
            "selected_candidate": validated.selected_candidate,
            "embedding": {
                "model_id": selected.embedding.model_id,
                "dimensions": selected.embedding.dimensions,
                "output_normalization": selected.embedding.output_normalization,
            },
            # Every field the strategy uses is in the preimage, so two hierarchical
            # configurations that differ only in parent size or overlap yield different
            # generation identities. Serializing the fixed-size fields alone would have hashed
            # every hierarchical variant identically.
            "chunking": _chunking_template(selected.chunking),
            "index": {
                "dimensions": selected.index.dimensions,
                "engine": selected.index.engine,
                "algorithm": selected.index.algorithm,
                "distance_metric": selected.index.distance_metric,
                "vector_field": selected.index.vector_field,
                "text_field": selected.index.text_field,
                "metadata_field": selected.index.metadata_field,
            },
            "retrieval": {
                "search_type": selected.retrieval.search_type,
                "number_of_results": selected.retrieval.number_of_results,
                "generation_filter_field": selected.retrieval.generation_filter_field,
                "exact_identifier_route": selected.retrieval.exact_identifier_route,
                "unavailable_generation_behavior": (
                    selected.retrieval.unavailable_generation_behavior
                ),
                "reranking": selected.retrieval.reranking,
            },
        }
        _canonical_json(value)
    except RetrievalConfigError as error:
        raise GenerationError(f"retrieval configuration is not authoritative: {error}") from error
    return value


def _validate_bundle_shape(bundle: GenerationBundle, limits: GenerationLimits) -> None:
    _validate_timestamp(bundle.created_at)
    if not isinstance(bundle.generation_id, str) or _DIGEST.fullmatch(bundle.generation_id) is None:
        raise GenerationError("generation ID is malformed")
    collections = (
        (bundle.documents, "documents"),
        (bundle.metadata_sidecars, "metadata sidecars"),
        (bundle.structured_records, "structured records"),
    )
    object_keys: list[str] = []
    for values, name in collections:
        if not isinstance(values, tuple):
            raise GenerationError(f"generation {name} must be an immutable tuple")
        identifiers: list[str] = []
        for value in values:
            _validate_object(value, limits.max_object_bytes, name)
            identifiers.append(value.object_id)
            object_keys.append(value.object_key)
        if identifiers != sorted(identifiers):
            raise GenerationError(f"generation {name} are not in lexical order")
        if len(identifiers) != len(set(identifiers)):
            raise GenerationError(f"generation {name} contain duplicate IDs")
    if [item.object_id for item in bundle.documents] != [
        item.object_id for item in bundle.metadata_sidecars
    ]:
        raise GenerationError("document and metadata-sidecar identities do not match")
    _validate_object(bundle.manifest, limits.max_manifest_bytes, "manifest")
    object_keys.append(bundle.manifest.object_key)
    if len(object_keys) != len(set(object_keys)):
        raise GenerationError("generation objects contain duplicate object keys")
    if bundle.manifest.object_id != bundle.generation_id:
        raise GenerationError("manifest object identity does not match generation ID")


def _validate_object(value: object, maximum_bytes: int, name: str) -> None:
    if not isinstance(value, CanonicalObject):
        raise GenerationError(f"generation {name} contain an object with the wrong runtime type")
    if not isinstance(value.object_id, str) or not value.object_id:
        raise GenerationError(f"generation {name} contain a malformed object ID")
    _validate_object_key(value.object_key, name)
    if not isinstance(value.content, bytes) or not 1 <= len(value.content) <= maximum_bytes:
        raise GenerationError(f"generation {name} contain content outside its byte bound")
    if not isinstance(value.digest, str) or _DIGEST.fullmatch(value.digest) is None:
        raise GenerationError(f"generation {name} contain a malformed object checksum")
    if value.digest != _sha256(value.content):
        raise GenerationError(f"generation {name} contain a mismatched object checksum")


def _object(
    object_id: str,
    object_key: str,
    content: bytes,
    limits: GenerationLimits,
) -> CanonicalObject:
    value = CanonicalObject(object_id, object_key, content, _sha256(content))
    _validate_object(value, limits.max_object_bytes, "outputs")
    return value


def _validate_limits(limits: object) -> None:
    if not isinstance(limits, GenerationLimits):
        raise GenerationError("generation limits have the wrong runtime type")
    values = (
        limits.max_sources_bytes,
        limits.max_documents,
        limits.max_structured_records,
        limits.max_structured_acquisitions,
        limits.max_object_bytes,
        limits.max_preimage_bytes,
        limits.max_manifest_bytes,
    )
    if any(type(value) is not int or value < 1 for value in values):
        raise GenerationError("generation bounds must be positive integers")


def _is_calendar_timestamp(value: str) -> bool:
    """Reject impossible dates and times the shape regex admits.

    The regex pins digit layout only, so 2026-99-99T99:99:99Z matches it. Parsing is what
    establishes the value names a real instant.
    """
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def _validate_timestamp(value: object) -> None:
    if (
        not isinstance(value, str)
        or _TIMESTAMP.fullmatch(value) is None
        or not _is_calendar_timestamp(value)
    ):
        raise GenerationError("created_at must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise GenerationError("created_at must be a valid UTC timestamp") from error
    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise GenerationError("created_at must be a UTC timestamp")


# Bedrock parses a knowledge-base object by extension and embeds whatever text it finds.
# Publishing the document body as .md or .txt means the embedding covers the prose. It used
# to be a .json envelope, so every document's first chunk began with the identical
# `{"api_version":...,"content":"` prefix and the body arrived JSON-escaped, which flattened
# similarity scores and destroyed markdown structure. Provenance lives in the sidecar, which
# Bedrock reads as metadata and never embeds.
_DOCUMENT_SUFFIXES: Final[Mapping[str, str]] = {"text/markdown": "md"}
_DEFAULT_DOCUMENT_SUFFIX: Final = "txt"


def _document_object_key(document_id: str, content_type: str) -> str:
    if _DIGEST.fullmatch(document_id) is None:
        raise GenerationError("document ID is malformed for object-key derivation")
    suffix = _DOCUMENT_SUFFIXES.get(content_type, _DEFAULT_DOCUMENT_SUFFIX)
    return f"documents/{document_id.removeprefix('sha256:')}.{suffix}"


def _structured_object_key(record_id: str) -> str:
    if not isinstance(record_id, str) or not record_id:
        raise GenerationError("record ID is malformed for object-key derivation")
    return f"structured/{quote(record_id, safe='')}.json"


def _control_object_key(generation_id: str) -> str:
    if _DIGEST.fullmatch(generation_id) is None:
        raise GenerationError("generation ID is malformed for object-key derivation")
    return f"control/generations/{generation_id.removeprefix('sha256:')}/manifest.json"


def _validate_object_key(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise GenerationError(f"generation {name} contain a malformed object key")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise GenerationError(f"generation {name} contain a non-UTF-8 object key") from error
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != value
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise GenerationError(f"generation {name} contain an unsafe object key")


def _decode_object(value: bytes, name: str) -> dict[str, object]:
    try:
        decoded = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GenerationError(f"canonical {name} is not JSON") from error
    if not isinstance(decoded, dict):  # pragma: no cover - producers own it
        raise GenerationError(f"canonical {name} must be an object")
    return cast(dict[str, object], decoded)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise GenerationError("generation value is not canonically encodable") from error


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"
