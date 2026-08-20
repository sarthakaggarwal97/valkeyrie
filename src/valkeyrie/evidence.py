"""Deterministic evidence packaging and application-owned citation rendering."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final, Literal, TypeAlias, cast
from urllib.parse import quote

from valkeyrie.generation import (
    CanonicalObject,
    GenerationBundle,
    GenerationError,
    verify_generation_bundle,
)
from valkeyrie.revisions import ResolvedRevision
from valkeyrie.sources import SourceInventoryError, _load_yaml_mapping, classify_path
from valkeyrie.structured import (
    PathRecord,
    ReleaseArtifactDigestRecord,
    StructuredRecord,
    StructuredRecordError,
    SymbolRecord,
    build_release_artifact_digest_records,
    canonical_record_bytes,
)


class EvidenceError(ValueError):
    """Evidence provenance, bounds, coverage, or citation input is invalid."""


@dataclass(frozen=True)
class EvidenceLimits:
    """Hard local bounds for one evidence package."""

    max_evidence_records: int = 50
    max_excerpt_characters: int = 4_000
    max_excerpt_bytes: int = 16_000
    max_claims: int = 100
    max_claim_id_bytes: int = 128
    max_evidence_ids_per_claim: int = 20


@dataclass(frozen=True)
class DocumentExcerpt:
    """A character-safe half-open excerpt span in one normalized document."""

    document_id: str
    start: int
    end: int


@dataclass(frozen=True)
class StructuredRecordEvidence:
    """One upstream-provable structured record, selected by record ID."""

    record_id: str


EvidenceSelection: TypeAlias = DocumentExcerpt | StructuredRecordEvidence


@dataclass(frozen=True)
class ClaimSupport:
    """Evidence IDs attached to one claim.

    This mapping proves provenance coverage only. It does not determine whether
    the cited excerpt semantically entails the claim.
    """

    claim_id: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceRecord:
    """Frozen schema record plus the provenance needed for exact reconstruction."""

    evidence_id: str
    generation_id: str
    object_kind: Literal["document", "structured_record"]
    object_id: str
    object_digest: str
    excerpt_start: int
    excerpt_end: int
    repository: str
    path: str
    commit: str
    authority: str
    version_scope: str
    immutable_url: str
    excerpt: str
    excerpt_digest: str


@dataclass(frozen=True)
class EvidencePackage:
    """A frozen, lexically ordered set of validated evidence records."""

    generation_id: str
    records: tuple[EvidenceRecord, ...]
    digest: str


_DEFAULT_LIMITS = EvidenceLimits()
_EVIDENCE_API_VERSION: Final = "valkeyrie.io/evidence-record/1"
_PACKAGE_API_VERSION: Final = "valkeyrie.io/evidence-package/1"
_EVIDENCE_ID_PREIMAGE_VERSION: Final = "valkeyrie.io/evidence-identity/1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_EVIDENCE_ID = re.compile(r"^ev_[a-z0-9-]+$")
_CLAIM_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_REPOSITORY = re.compile(r"^[a-z0-9.][a-z0-9._-]*$")


def create_evidence_package(
    bundle: GenerationBundle,
    selections: tuple[EvidenceSelection, ...],
    *,
    limits: EvidenceLimits = _DEFAULT_LIMITS,
) -> EvidencePackage:
    """Create evidence only from a completely reconstructed generation bundle."""
    verified = _verified_bundle(bundle)
    _validate_limits(limits)
    if not isinstance(selections, tuple):
        raise EvidenceError("evidence selections must be an immutable tuple")
    if not 1 <= len(selections) <= limits.max_evidence_records:
        raise EvidenceError("evidence selection count is outside its bound")

    inventory = _source_inventory(verified)
    records = tuple(
        sorted(
            (_build_record(verified, inventory, selection, limits) for selection in selections),
            key=lambda record: record.evidence_id,
        )
    )
    evidence_ids = [record.evidence_id for record in records]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise EvidenceError("duplicate evidence selection")
    digest = _package_digest(verified.generation_id, records)
    return EvidencePackage(verified.generation_id, records, digest)


def verify_evidence_package(
    bundle: GenerationBundle,
    package: EvidencePackage,
    *,
    limits: EvidenceLimits = _DEFAULT_LIMITS,
) -> EvidencePackage:
    """Reconstruct and return one valid evidence package or fail closed."""
    verified = _verified_bundle(bundle)
    _validate_limits(limits)
    _validate_package_shape(package, limits)
    if package.generation_id != verified.generation_id:
        raise EvidenceError("evidence package and generation IDs do not match")

    selections: list[EvidenceSelection] = []
    for record in package.records:
        if record.object_kind == "document":
            selections.append(
                DocumentExcerpt(record.object_id, record.excerpt_start, record.excerpt_end)
            )
        elif record.object_kind == "structured_record":
            if record.excerpt_start != 0:
                raise EvidenceError("structured-record evidence must contain the exact full record")
            selections.append(StructuredRecordEvidence(record.object_id))
        else:  # pragma: no cover - shape validation owns this branch
            raise EvidenceError("unsupported evidence object kind")

    expected = create_evidence_package(
        verified,
        tuple(selections),
        limits=limits,
    )
    if package != expected:
        raise EvidenceError("evidence package does not match its generation objects")
    return expected


def evidence_record_value(record: EvidenceRecord) -> dict[str, object]:
    """Return a fresh value matching the ``EvidenceRecord`` JSON schema."""
    _validate_record_shape(record)
    return {
        "api_version": _EVIDENCE_API_VERSION,
        "kind": "EvidenceRecord",
        "evidence_id": record.evidence_id,
        "generation_id": record.generation_id,
        "source": {
            "repository": record.repository,
            "path": record.path,
            "commit": record.commit,
            "authority": record.authority,
            "version_scope": record.version_scope,
            "immutable_url": record.immutable_url,
        },
        "excerpt": record.excerpt,
        "excerpt_digest": record.excerpt_digest,
    }


def canonical_evidence_record_bytes(record: EvidenceRecord) -> bytes:
    """Serialize one internally consistent evidence record as strict canonical JSON."""
    return _canonical_json(evidence_record_value(record))


def validate_claim_support(
    bundle: GenerationBundle,
    package: EvidencePackage,
    claim_support: tuple[ClaimSupport, ...],
    *,
    limits: EvidenceLimits = _DEFAULT_LIMITS,
) -> tuple[ClaimSupport, ...]:
    """Validate and canonicalize claim-to-evidence provenance coverage.

    Every supplied bounded claim must reference known evidence. This validates
    provenance coverage only; it does not establish semantic entailment.
    """
    verified = verify_evidence_package(bundle, package, limits=limits)
    if not isinstance(claim_support, tuple):
        raise EvidenceError("claim support must be an immutable tuple")
    if not 1 <= len(claim_support) <= limits.max_claims:
        raise EvidenceError("claim count is outside its bound")
    return _canonical_claim_support(
        claim_support,
        frozenset(record.evidence_id for record in verified.records),
        limits,
    )


def render_citations(
    bundle: GenerationBundle,
    package: EvidencePackage,
    evidence_ids: tuple[str, ...],
    *,
    limits: EvidenceLimits = _DEFAULT_LIMITS,
) -> tuple[str, ...]:
    """Render canonical Markdown citations from evidence IDs only.

    Labels and immutable links are application-owned. The API has no URL input;
    malformed objects that attempt to carry model-supplied links are rejected as
    invalid evidence IDs.
    """
    verified = verify_evidence_package(bundle, package, limits=limits)
    if not isinstance(evidence_ids, tuple) or not evidence_ids:
        raise EvidenceError("citation evidence IDs must be a non-empty immutable tuple")
    if any(not isinstance(evidence_id, str) for evidence_id in evidence_ids):
        raise EvidenceError("citation input may contain evidence IDs only")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise EvidenceError("citation evidence IDs must not be duplicated")
    by_id = {record.evidence_id: record for record in verified.records}
    unknown = sorted(set(evidence_ids) - set(by_id))
    if unknown:
        raise EvidenceError(f"unknown citation evidence ID: {unknown[0]}")
    return tuple(_render_citation(by_id[evidence_id]) for evidence_id in sorted(evidence_ids))


def _build_record(
    bundle: GenerationBundle,
    inventory: Mapping[str, object],
    selection: EvidenceSelection,
    limits: EvidenceLimits,
) -> EvidenceRecord:
    if isinstance(selection, DocumentExcerpt):
        document = next(
            (
                item
                for item in bundle.document_templates
                if item.document_id == selection.document_id
            ),
            None,
        )
        source_object = next(
            (item for item in bundle.documents if item.object_id == selection.document_id),
            None,
        )
        if document is None or source_object is None:
            raise EvidenceError(f"unknown document ID: {selection.document_id!r}")
        start, end = _validate_span(selection.start, selection.end, document.content, limits)
        excerpt = document.content[start:end]
        source = document.source
        path = document.path
        object_kind: Literal["document", "structured_record"] = "document"
    elif isinstance(selection, StructuredRecordEvidence):
        template, source_object = _structured_record(bundle, selection.record_id)
        source, path, excerpt = _structured_record_excerpt(bundle, template)
        start, end = _validate_span(0, len(excerpt), excerpt, limits)
        object_kind = "structured_record"
    else:
        raise EvidenceError("unsupported evidence selection type")

    _require_reviewed_path(inventory, source.repository, path)
    excerpt_digest = _sha256(excerpt.encode("utf-8"))
    immutable_url = _immutable_url(source, path)
    identity = {
        "api_version": _EVIDENCE_ID_PREIMAGE_VERSION,
        "kind": "EvidenceIdentity",
        "generation_id": bundle.generation_id,
        "object_kind": object_kind,
        "object_id": source_object.object_id,
        "object_digest": source_object.digest,
        "excerpt_start": start,
        "excerpt_end": end,
        "repository": source.repository,
        "path": path,
        "commit": source.commit,
        "authority": source.authority,
        "version_scope": source.version_scope,
        "excerpt_digest": excerpt_digest,
    }
    evidence_id = f"ev_{hashlib.sha256(_canonical_json(identity)).hexdigest()}"
    record = EvidenceRecord(
        evidence_id=evidence_id,
        generation_id=bundle.generation_id,
        object_kind=object_kind,
        object_id=source_object.object_id,
        object_digest=source_object.digest,
        excerpt_start=start,
        excerpt_end=end,
        repository=source.repository,
        path=path,
        commit=source.commit,
        authority=source.authority,
        version_scope=source.version_scope,
        immutable_url=immutable_url,
        excerpt=excerpt,
        excerpt_digest=excerpt_digest,
    )
    _validate_record_shape(record)
    return record


def _structured_record(
    bundle: GenerationBundle, record_id: object
) -> tuple[StructuredRecord, CanonicalObject]:
    if not isinstance(record_id, str) or not record_id:
        raise EvidenceError("structured record ID must be a non-empty string")
    indexed: dict[str, tuple[StructuredRecord, CanonicalObject]] = {}
    try:
        for template, source_object in zip(
            bundle.structured_record_templates,
            bundle.structured_records,
            strict=True,
        ):
            value = cast(dict[str, object], json.loads(canonical_record_bytes(template)))
            identifier = value.get("record_id")
            if not isinstance(identifier, str) or not identifier:
                raise EvidenceError("structured record has a malformed record ID")
            indexed[identifier] = (template, source_object)
    except (StructuredRecordError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"structured record is malformed: {error}") from error
    found = indexed.get(record_id)
    if found is None:
        raise EvidenceError(f"unknown structured record ID: {record_id!r}")
    if found[1].object_id != record_id:
        raise EvidenceError("structured record object identity is inconsistent")
    return found


def _structured_record_excerpt(
    bundle: GenerationBundle,
    record: StructuredRecord,
) -> tuple[ResolvedRevision, str, str]:
    if isinstance(record, PathRecord | SymbolRecord):
        raise EvidenceError(
            "PathRecord and SymbolRecord direct evidence is unsupported; use DocumentExcerpt"
        )
    if not isinstance(record, ReleaseArtifactDigestRecord):
        raise EvidenceError(
            "structured record has no reviewed repository path; use downstream document evidence"
        )

    acquisition = next(
        (
            item
            for item in bundle.structured_acquisitions
            if item.repository == record.source.repository and item.commit == record.source.commit
        ),
        None,
    )
    if acquisition is None:
        raise EvidenceError("release digest evidence is missing its exact structured acquisition")
    try:
        reconstructed = build_release_artifact_digest_records(acquisition, record.source)
    except StructuredRecordError as error:
        raise EvidenceError(f"release digest acquisition is invalid: {error}") from error
    if record not in reconstructed:
        raise EvidenceError("release digest is not present in its exact structured acquisition")

    if record.provenance_path == "README":
        release = record.identifier.release
        url = (
            "https://github.com/valkey-io/valkey/archive/unstable.tar.gz"
            if release == "unstable"
            else f"https://github.com/valkey-io/valkey/archive/refs/tags/{release}.tar.gz"
        )
        expected = (
            f"hash {record.identifier.artifact} sha256 "
            f"{record.digest.removeprefix('sha256:')} {url}"
        )
    else:
        expected = f"{record.digest.removeprefix('sha256:')}  {record.identifier.artifact}"
    source_file = next(
        (item for item in acquisition.files if item.path == record.provenance_path),
        None,
    )
    if source_file is None:
        raise EvidenceError("release digest provenance path is absent from its acquisition")
    expected_bytes = expected.encode("utf-8")
    if sum(line == expected_bytes for line in source_file.content.split(b"\n")) != 1:
        raise EvidenceError("release digest canonical source line is not uniquely present")
    return record.source, record.provenance_path, expected


def _canonical_claim_support(
    supports: tuple[ClaimSupport, ...],
    known_evidence_ids: frozenset[str],
    limits: EvidenceLimits,
) -> tuple[ClaimSupport, ...]:
    canonical: list[ClaimSupport] = []
    seen_claims: set[str] = set()
    for support in supports:
        if not isinstance(support, ClaimSupport):
            raise EvidenceError("claim support contains a malformed entry")
        _validate_claim_id(support.claim_id, limits.max_claim_id_bytes)
        if support.claim_id in seen_claims:
            raise EvidenceError(f"duplicate claim ID: {support.claim_id}")
        seen_claims.add(support.claim_id)
        if not isinstance(support.evidence_ids, tuple):
            raise EvidenceError("claim evidence IDs must be an immutable tuple")
        if not 1 <= len(support.evidence_ids) <= limits.max_evidence_ids_per_claim:
            raise EvidenceError("claim evidence-ID count is outside its bound")
        if any(not isinstance(item, str) for item in support.evidence_ids):
            raise EvidenceError("claim support may contain evidence IDs only")
        if len(support.evidence_ids) != len(set(support.evidence_ids)):
            raise EvidenceError(f"claim {support.claim_id!r} repeats an evidence ID")
        unknown = sorted(set(support.evidence_ids) - known_evidence_ids)
        if unknown:
            raise EvidenceError(
                f"claim {support.claim_id!r} references unknown evidence ID {unknown[0]!r}"
            )
        canonical.append(ClaimSupport(support.claim_id, tuple(sorted(support.evidence_ids))))
    return tuple(sorted(canonical, key=lambda support: support.claim_id))


def _validate_package_shape(package: object, limits: EvidenceLimits) -> None:
    if not isinstance(package, EvidencePackage):
        raise EvidenceError("evidence package has the wrong runtime type")
    if (
        not isinstance(package.generation_id, str)
        or _DIGEST.fullmatch(package.generation_id) is None
    ):
        raise EvidenceError("evidence package generation ID is malformed")
    if not isinstance(package.records, tuple):
        raise EvidenceError("evidence records must be an immutable tuple")
    if not 1 <= len(package.records) <= limits.max_evidence_records:
        raise EvidenceError("evidence record count is outside its bound")
    for record in package.records:
        _validate_record_shape(record)
        if record.generation_id != package.generation_id:
            raise EvidenceError("evidence record belongs to a different generation")
    record_ids = [record.evidence_id for record in package.records]
    if record_ids != sorted(record_ids) or len(record_ids) != len(set(record_ids)):
        raise EvidenceError("evidence records must have unique lexical IDs")
    expected_digest = _package_digest(package.generation_id, package.records)
    if package.digest != expected_digest:
        raise EvidenceError("evidence package digest is inconsistent")


def _validate_record_shape(record: object) -> None:
    if not isinstance(record, EvidenceRecord):
        raise EvidenceError("evidence package contains a malformed record")
    text_fields = (
        record.object_id,
        record.repository,
        record.path,
        record.version_scope,
        record.excerpt,
    )
    if any(not isinstance(value, str) or not value for value in text_fields):
        raise EvidenceError("evidence record contains empty or malformed text")
    if (
        not isinstance(record.evidence_id, str)
        or _EVIDENCE_ID.fullmatch(record.evidence_id) is None
    ):
        raise EvidenceError("evidence ID is malformed")
    for value, name in (
        (record.generation_id, "generation ID"),
        (record.object_digest, "object checksum"),
        (record.excerpt_digest, "excerpt checksum"),
    ):
        if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
            raise EvidenceError(f"evidence {name} is malformed")
    if record.object_kind not in {"document", "structured_record"}:
        raise EvidenceError("evidence object kind is malformed")
    if not isinstance(record.commit, str) or _SHA.fullmatch(record.commit) is None:
        raise EvidenceError("evidence commit must be a full lowercase 40-hex SHA")
    if record.authority not in {"canonical", "secondary", "structured"}:
        raise EvidenceError("evidence authority is malformed")
    _validate_repository_and_path(record.repository, record.path)
    if (
        type(record.excerpt_start) is not int
        or type(record.excerpt_end) is not int
        or record.excerpt_start < 0
        or record.excerpt_end <= record.excerpt_start
        or record.excerpt_end - record.excerpt_start != len(record.excerpt)
    ):
        raise EvidenceError("evidence excerpt span is malformed")
    try:
        excerpt_bytes = record.excerpt.encode("utf-8")
    except UnicodeEncodeError as error:
        raise EvidenceError("evidence excerpt is not valid UTF-8") from error
    if record.excerpt_digest != _sha256(excerpt_bytes):
        raise EvidenceError("evidence excerpt checksum is inconsistent")
    if record.immutable_url != _immutable_url_parts(record.repository, record.commit, record.path):
        raise EvidenceError("evidence immutable URL is inconsistent with source metadata")
    expected_id = _evidence_id_from_record(record)
    if record.evidence_id != expected_id:
        raise EvidenceError("evidence ID is inconsistent with provenance")


def _evidence_id_from_record(record: EvidenceRecord) -> str:
    identity = {
        "api_version": _EVIDENCE_ID_PREIMAGE_VERSION,
        "kind": "EvidenceIdentity",
        "generation_id": record.generation_id,
        "object_kind": record.object_kind,
        "object_id": record.object_id,
        "object_digest": record.object_digest,
        "excerpt_start": record.excerpt_start,
        "excerpt_end": record.excerpt_end,
        "repository": record.repository,
        "path": record.path,
        "commit": record.commit,
        "authority": record.authority,
        "version_scope": record.version_scope,
        "excerpt_digest": record.excerpt_digest,
    }
    return f"ev_{hashlib.sha256(_canonical_json(identity)).hexdigest()}"


def _package_digest(
    generation_id: str,
    records: tuple[EvidenceRecord, ...],
) -> str:
    value = {
        "api_version": _PACKAGE_API_VERSION,
        "kind": "EvidencePackage",
        "generation_id": generation_id,
        "records": [
            {
                "record": evidence_record_value(record),
                "object_kind": record.object_kind,
                "object_id": record.object_id,
                "object_digest": record.object_digest,
                "excerpt_start": record.excerpt_start,
                "excerpt_end": record.excerpt_end,
            }
            for record in records
        ],
    }
    return _sha256(_canonical_json(value))


def _verified_bundle(bundle: object) -> GenerationBundle:
    try:
        return verify_generation_bundle(cast(GenerationBundle, bundle))
    except (GenerationError, TypeError, ValueError) as error:
        raise EvidenceError(f"generation bundle is not verified: {error}") from error


def _source_inventory(bundle: GenerationBundle) -> dict[str, object]:
    try:
        return _load_yaml_mapping(
            bundle.sources_yaml.decode("utf-8"),
            "verified sources.yaml",
            reject_merge_keys=True,
        )
    except (UnicodeError, ValueError) as error:  # pragma: no cover - verified above
        raise EvidenceError(f"verified source inventory is malformed: {error}") from error


def _require_reviewed_path(inventory: Mapping[str, object], repository: str, path: str) -> None:
    try:
        classification = classify_path(inventory, repository, path)
    except SourceInventoryError as error:
        raise EvidenceError(f"evidence source path is invalid: {error}") from error
    if classification != "include":
        raise EvidenceError(
            f"evidence path {repository}/{path} is excluded from reviewed source material"
        )


def _validate_span(
    start: object,
    end: object,
    content: str,
    limits: EvidenceLimits,
) -> tuple[int, int]:
    if type(start) is not int or type(end) is not int:
        raise EvidenceError("excerpt boundaries must be integer character offsets")
    if start < 0 or end <= start or end > len(content):
        raise EvidenceError("excerpt boundaries are outside canonical content")
    excerpt = content[start:end]
    try:
        excerpt_bytes = excerpt.encode("utf-8")
    except UnicodeEncodeError as error:  # pragma: no cover - verified generation owns it
        raise EvidenceError("excerpt is not valid UTF-8") from error
    if (
        not excerpt
        or len(excerpt) > limits.max_excerpt_characters
        or len(excerpt_bytes) > limits.max_excerpt_bytes
    ):
        raise EvidenceError("excerpt is empty or exceeds its character/byte bound")
    return start, end


def _validate_claim_id(value: object, maximum_bytes: int) -> None:
    if not isinstance(value, str) or _CLAIM_ID.fullmatch(value) is None:
        raise EvidenceError("claim ID is malformed")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:  # pragma: no cover - pattern is ASCII
        raise EvidenceError("claim ID is not valid UTF-8") from error
    if len(encoded) > maximum_bytes:
        raise EvidenceError("claim ID exceeds its byte bound")


def _validate_limits(limits: object) -> None:
    if not isinstance(limits, EvidenceLimits):
        raise EvidenceError("evidence limits have the wrong runtime type")
    values = (
        limits.max_evidence_records,
        limits.max_excerpt_characters,
        limits.max_excerpt_bytes,
        limits.max_claims,
        limits.max_claim_id_bytes,
        limits.max_evidence_ids_per_claim,
    )
    if any(type(value) is not int or value < 1 for value in values):
        raise EvidenceError("evidence bounds must be positive integers")


def _immutable_url(source: ResolvedRevision, path: str) -> str:
    return _immutable_url_parts(source.repository, source.commit, path)


def _immutable_url_parts(repository: str, commit: str, path: str) -> str:
    _validate_repository_and_path(repository, path)
    if not isinstance(commit, str) or _SHA.fullmatch(commit) is None:
        raise EvidenceError("immutable URL requires a full lowercase 40-hex commit")
    return f"https://github.com/valkey-io/{repository}/blob/{commit}/{quote(path, safe='/')}"


def _validate_repository_and_path(repository: object, path: object) -> None:
    if not isinstance(repository, str) or _REPOSITORY.fullmatch(repository) is None:
        raise EvidenceError("evidence repository is malformed")
    if not isinstance(path, str) or not path:
        raise EvidenceError("evidence path must be non-empty")
    try:
        path.encode("utf-8")
    except UnicodeEncodeError as error:
        raise EvidenceError("evidence path is not valid UTF-8") from error
    candidate = PurePosixPath(path)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != path
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
    ):
        raise EvidenceError("evidence path is not a safe repository-relative path")


def _render_citation(record: EvidenceRecord) -> str:
    label = f"{record.repository}/{record.path}@{record.commit[:12]}"
    escaped_label = label.replace("\\", "\\\\").replace("]", "\\]")
    return f"[{escaped_label}]({record.immutable_url})"


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
        raise EvidenceError("evidence value is not canonically encodable") from error


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"
