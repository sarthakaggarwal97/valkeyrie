"""Complete in-memory orchestration for reviewed static corpus generations."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import yaml

from valkeyrie.acquisition import AcquiredFile, AcquiredRepository, acquire_source
from valkeyrie.generation import (
    GenerationBundle,
    create_generation_bundle,
    verify_generation_bundle,
)
from valkeyrie.normalization import NormalizedDocument, normalize_repository
from valkeyrie.retrieval_config import FrozenRetrievalConfiguration
from valkeyrie.revisions import ResolvedRevision, resolve_source_revision
from valkeyrie.sources import (
    SourceInventoryError,
    _load_yaml_mapping,
    classify_path,
    validate_source_inventory,
)
from valkeyrie.structured import (
    StructuredRecord,
    build_release_artifact_digest_records,
)


class CorpusBuildError(ValueError):
    """A complete reviewed corpus generation could not be assembled."""


class RevisionResolver(Protocol):
    def __call__(
        self, source_inventory: Mapping[str, object], repository_name: str, /
    ) -> ResolvedRevision: ...


class SourceAcquirer(Protocol):
    def __call__(
        self, source_inventory: Mapping[str, object], resolved: ResolvedRevision, /
    ) -> AcquiredRepository: ...


class DocumentNormalizer(Protocol):
    def __call__(
        self,
        source_inventory: Mapping[str, object],
        resolved: ResolvedRevision,
        acquired: AcquiredRepository,
        /,
    ) -> tuple[NormalizedDocument, ...]: ...


class StructuredRecordBuilder(Protocol):
    def __call__(
        self, acquired: AcquiredRepository, source: ResolvedRevision, /
    ) -> tuple[StructuredRecord, ...]: ...


class GenerationCreator(Protocol):
    def __call__(
        self,
        sources_yaml: bytes,
        documents: tuple[NormalizedDocument, ...],
        structured_records: tuple[StructuredRecord, ...],
        retrieval_config: FrozenRetrievalConfiguration,
        /,
        *,
        created_at: str,
        structured_acquisitions: tuple[AcquiredRepository, ...],
    ) -> GenerationBundle: ...


class GenerationVerifier(Protocol):
    def __call__(self, bundle: GenerationBundle) -> GenerationBundle: ...


@dataclass(frozen=True)
class CorpusFunctions:
    """Injectable pure pipeline functions used by :func:`build_corpus`."""

    resolve: RevisionResolver = resolve_source_revision
    acquire: SourceAcquirer = acquire_source
    normalize: DocumentNormalizer = normalize_repository
    build_records: StructuredRecordBuilder = build_release_artifact_digest_records
    create_generation: GenerationCreator = create_generation_bundle
    verify_generation: GenerationVerifier = verify_generation_bundle


_DEFAULT_FUNCTIONS = CorpusFunctions()


def build_corpus(
    sources_yaml: bytes,
    retrieval_config: FrozenRetrievalConfiguration,
    *,
    created_at: str,
    profile: str | None = None,
    functions: CorpusFunctions = _DEFAULT_FUNCTIONS,
) -> GenerationBundle:
    """Build and verify one complete reviewed generation entirely in memory.

    ``profile`` optionally scopes the build to one reviewed repository family.
    Paths are never supplied by the caller: each acquisition must be the exact,
    complete path set consumed by normalization or structured parsing.
    """
    inventory = _load_inventory(sources_yaml)
    selected = _select_sources(inventory, profile)
    if not isinstance(functions, CorpusFunctions):
        raise CorpusBuildError("corpus functions have the wrong runtime type")

    documents: list[NormalizedDocument] = []
    records: list[StructuredRecord] = []
    structured_acquisitions: list[AcquiredRepository] = []
    for entry in selected:
        repository = cast(str, entry["name"])
        try:
            resolved = functions.resolve(inventory, repository)
            if not isinstance(resolved, ResolvedRevision) or resolved.repository != repository:
                raise CorpusBuildError(
                    f"resolver omitted or substituted selected source {repository!r}"
                )
            acquired = functions.acquire(inventory, resolved)
            acquired_paths = _validate_acquisition(inventory, resolved, acquired)
            if entry["ingestion_mode"] == "documents":
                normalized = functions.normalize(inventory, resolved, acquired)
                _validate_documents(resolved, acquired_paths, normalized)
                documents.extend(normalized)
            else:
                built_records = functions.build_records(acquired, resolved)
                _validate_records(resolved, built_records)
                records.extend(built_records)
                structured_acquisitions.append(acquired)
            # After either mode: both acquire files, so both can leave content out.
            _report_skipped(acquired)
        except CorpusBuildError:
            raise
        except Exception as error:
            raise CorpusBuildError(
                f"cannot build selected source {repository!r}: {error}"
            ) from error

    try:
        bundle = functions.create_generation(
            sources_yaml,
            tuple(documents),
            tuple(records),
            retrieval_config,
            created_at=created_at,
            structured_acquisitions=tuple(structured_acquisitions),
        )
        if not isinstance(bundle, GenerationBundle):
            raise CorpusBuildError("generation creator returned the wrong runtime type")
        verified = functions.verify_generation(bundle)
    except CorpusBuildError:
        raise
    except Exception as error:
        raise CorpusBuildError(f"cannot create complete generation: {error}") from error
    if not isinstance(verified, GenerationBundle) or verified != bundle:
        raise CorpusBuildError("generation verifier did not return the complete created bundle")
    return verified


def _load_inventory(sources_yaml: object) -> dict[str, object]:
    if not isinstance(sources_yaml, bytes) or not sources_yaml:
        raise CorpusBuildError("sources.yaml content must be non-empty exact bytes")
    try:
        inventory = _load_yaml_mapping(
            sources_yaml.decode("utf-8"), "sources.yaml bytes", reject_merge_keys=True
        )
        validate_source_inventory(inventory)
    except (UnicodeError, yaml.YAMLError, ValueError, TypeError, SourceInventoryError) as error:
        raise CorpusBuildError(f"sources.yaml semantics are invalid: {error}") from error
    return inventory


def _select_sources(
    inventory: Mapping[str, object], profile: str | None
) -> tuple[dict[str, object], ...]:
    if profile is not None and (not isinstance(profile, str) or not profile.strip()):
        raise CorpusBuildError("corpus profile must be a non-blank source family")
    repositories = cast(list[dict[str, object]], inventory["repositories"])
    reviewed = [
        entry
        for entry in repositories
        if entry["classification"] in {"curated", "structured_exact"}
        and (profile is None or entry["family"] == profile)
    ]
    if not reviewed:
        scope = "all reviewed sources" if profile is None else f"profile {profile!r}"
        raise CorpusBuildError(f"corpus scope selects no reviewed sources: {scope}")
    return tuple(sorted(reviewed, key=lambda entry: cast(str, entry["name"])))


def _report_skipped(acquired: object) -> None:
    """Name every path an acquisition left out, on stderr.

    A file that policy included but that cannot become a document is skipped rather than ending the
    build. That is only safe if it is stated: silently missing content is indistinguishable from
    content that was never there, and this runs unattended once a week. stderr because the build
    writes its canonical report to stdout.
    """
    if not isinstance(acquired, AcquiredRepository) or not acquired.skipped:
        return
    for path, reason in acquired.skipped:
        print(
            f"skipped {acquired.repository}/{path}: {reason}",
            file=sys.stderr,
        )


def _validate_acquisition(
    inventory: Mapping[str, object],
    resolved: ResolvedRevision,
    acquired: object,
) -> tuple[str, ...]:
    if not isinstance(acquired, AcquiredRepository):
        raise CorpusBuildError("source acquirer returned the wrong runtime type")
    if acquired.repository != resolved.repository or acquired.commit != resolved.commit:
        raise CorpusBuildError(
            f"acquisition identity does not match selected source {resolved.repository!r}"
        )
    if not isinstance(acquired.files, tuple) or not acquired.files:
        raise CorpusBuildError(
            f"selected source {resolved.repository!r} reported no regular text paths"
        )

    paths: list[str] = []
    total_bytes = 0
    for acquired_file in acquired.files:
        if not isinstance(acquired_file, AcquiredFile):
            raise CorpusBuildError("source acquisition contains a malformed file")
        if classify_path(inventory, resolved.repository, acquired_file.path) != "include":
            raise CorpusBuildError(f"acquirer reported non-selected path {acquired_file.path!r}")
        if not isinstance(acquired_file.content, bytes):
            raise CorpusBuildError(
                f"acquirer reported non-byte content for path {acquired_file.path!r}"
            )
        paths.append(acquired_file.path)
        total_bytes += len(acquired_file.content)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise CorpusBuildError("acquirer path report must be unique and in lexical order")
    if type(acquired.total_bytes) is not int or acquired.total_bytes != total_bytes:
        raise CorpusBuildError("acquisition total_bytes does not match its complete path report")
    return tuple(paths)


def _validate_documents(
    resolved: ResolvedRevision,
    acquired_paths: tuple[str, ...],
    normalized: object,
) -> None:
    if not isinstance(normalized, tuple):
        raise CorpusBuildError("document normalizer must return an immutable tuple")
    if not all(isinstance(document, NormalizedDocument) for document in normalized):
        raise CorpusBuildError("document normalizer returned a malformed document")
    documents = cast(tuple[NormalizedDocument, ...], normalized)
    normalized_paths = tuple(document.path for document in documents)
    if normalized_paths != acquired_paths:
        raise CorpusBuildError(
            f"normalized paths do not exactly match the complete acquisition for "
            f"{resolved.repository!r}"
        )
    if any(document.source != resolved for document in documents):
        raise CorpusBuildError(
            f"normalized documents do not match selected source {resolved.repository!r}"
        )


def _validate_records(
    resolved: ResolvedRevision,
    records: object,
) -> None:
    if not isinstance(records, tuple) or not records:
        raise CorpusBuildError(
            f"structured source {resolved.repository!r} produced no exact records"
        )
    if any(getattr(record, "source", None) != resolved for record in records):
        raise CorpusBuildError(
            f"structured records do not match selected source {resolved.repository!r}"
        )
