from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from valkeyrie.acquisition import AcquiredFile, AcquiredRepository
from valkeyrie.corpus import (
    CorpusBuildError,
    CorpusFunctions,
    DocumentNormalizer,
    GenerationCreator,
    GenerationVerifier,
    build_corpus,
)
from valkeyrie.generation import (
    GenerationBundle,
    create_generation_bundle,
    verify_generation_bundle,
)
from valkeyrie.normalization import NormalizedDocument, normalize_repository
from valkeyrie.retrieval_config import FrozenRetrievalConfiguration, load_retrieval_config
from valkeyrie.revisions import Authority, ResolvedRevision
from valkeyrie.sources import load_source_inventory
from valkeyrie.structured import StructuredRecord, build_release_artifact_digest_records

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "sources.yaml"
CONFIG = ROOT / "retrieval-config.yaml"
CREATED_AT = "2026-08-19T23:58:28.439Z"
COMMIT = "1" * 40


@pytest.fixture(scope="module")
def source_inventory() -> dict[str, object]:
    return load_source_inventory(SOURCES)


@pytest.fixture(scope="module")
def retrieval_config() -> FrozenRetrievalConfiguration:
    return load_retrieval_config(CONFIG)


def _entries(inventory: Mapping[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], inventory["repositories"])


def _policy_digest(inventory: Mapping[str, object]) -> str:
    encoded = json.dumps(
        inventory,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _resolved(inventory: Mapping[str, object], repository: str) -> ResolvedRevision:
    entry = next(item for item in _entries(inventory) if item["name"] == repository)
    return ResolvedRevision(
        repository=repository,
        repository_url=cast(str, entry["url"]),
        requested_ref=cast(str, entry["requested_ref"]),
        ref_kind="branch",
        commit=COMMIT,
        authority=cast(Authority, entry["authority"]),
        version_scope=cast(str, entry["version_scope"]),
        source_policy_digest=_policy_digest(inventory),
    )


def _acquired(resolved: ResolvedRevision) -> AcquiredRepository:
    if resolved.repository == "valkey-hashes":
        content = (
            "hash valkey-9.0.0.tar.gz sha256 "
            f"{'a' * 64} "
            "https://github.com/valkey-io/valkey/archive/refs/tags/9.0.0.tar.gz\n"
        ).encode()
        path = "README"
    else:
        content = f"# {resolved.repository}\n".encode()
        path = "README.md"
    file = AcquiredFile(path, content)
    return AcquiredRepository(resolved.repository, resolved.commit, (file,), len(content))


def _functions(
    calls: list[tuple[str, str]],
    *,
    normalize: DocumentNormalizer = normalize_repository,
    acquire: Callable[[ResolvedRevision], AcquiredRepository] = _acquired,
    create: GenerationCreator = create_generation_bundle,
    verify: GenerationVerifier = verify_generation_bundle,
) -> CorpusFunctions:
    def resolve(inventory: Mapping[str, object], repository: str) -> ResolvedRevision:
        calls.append(("resolve", repository))
        return _resolved(inventory, repository)

    def acquire_source(
        inventory: Mapping[str, object], resolved: ResolvedRevision
    ) -> AcquiredRepository:
        del inventory
        calls.append(("acquire", resolved.repository))
        return acquire(resolved)

    def normalize_source(
        inventory: Mapping[str, object],
        resolved: ResolvedRevision,
        acquired: AcquiredRepository,
    ) -> tuple[NormalizedDocument, ...]:
        calls.append(("normalize", resolved.repository))
        return normalize(inventory, resolved, acquired)

    def build_records(
        acquired: AcquiredRepository, resolved: ResolvedRevision
    ) -> tuple[StructuredRecord, ...]:
        calls.append(("records", resolved.repository))
        return build_release_artifact_digest_records(acquired, resolved)

    def create_generation(
        sources_yaml: bytes,
        documents: tuple[NormalizedDocument, ...],
        records: tuple[StructuredRecord, ...],
        config: FrozenRetrievalConfiguration,
        *,
        created_at: str,
        structured_acquisitions: tuple[AcquiredRepository, ...],
    ) -> GenerationBundle:
        calls.append(("create", "generation"))
        return create(
            sources_yaml,
            documents,
            records,
            config,
            created_at=created_at,
            structured_acquisitions=structured_acquisitions,
        )

    def verify_generation(bundle: GenerationBundle) -> GenerationBundle:
        calls.append(("verify", "generation"))
        return verify(bundle)

    return CorpusFunctions(
        resolve=resolve,
        acquire=acquire_source,
        normalize=normalize_source,
        build_records=build_records,
        create_generation=create_generation,
        verify_generation=verify_generation,
    )


def test_builds_every_reviewed_source_and_release_digest_record(
    source_inventory: dict[str, object],
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    calls: list[tuple[str, str]] = []
    bundle = build_corpus(
        SOURCES.read_bytes(),
        retrieval_config,
        created_at=CREATED_AT,
        functions=_functions(calls),
    )
    selected = sorted(
        cast(str, entry["name"])
        for entry in _entries(source_inventory)
        if entry["classification"] in {"curated", "structured_exact"}
    )

    assert [name for action, name in calls if action == "resolve"] == selected
    assert [name for action, name in calls if action == "acquire"] == selected
    assert {document.source.repository for document in bundle.document_templates} == (
        set(selected) - {"valkey-hashes"}
    )
    assert bundle.structured_acquisitions[0].repository == "valkey-hashes"
    assert [record.object_id for record in bundle.structured_records] == [
        "release_artifact_digest:9.0.0:valkey-9.0.0.tar.gz"
    ]
    assert calls[-2:] == [("create", "generation"), ("verify", "generation")]


def test_rejects_omitted_or_substituted_selected_source(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    calls: list[tuple[str, str]] = []
    functions = _functions(calls)

    def substitute(inventory: Mapping[str, object], repository: str) -> ResolvedRevision:
        return _resolved(inventory, "community" if repository == ".github" else repository)

    with pytest.raises(CorpusBuildError, match="omitted or substituted"):
        build_corpus(
            SOURCES.read_bytes(),
            retrieval_config,
            created_at=CREATED_AT,
            functions=replace(functions, resolve=substitute),
        )
    assert not any(action in {"create", "verify"} for action, _ in calls)


def test_rejects_any_path_omitted_by_normalization(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    calls: list[tuple[str, str]] = []

    def omit_path(
        inventory: Mapping[str, object],
        resolved: ResolvedRevision,
        acquired: AcquiredRepository,
    ) -> tuple[NormalizedDocument, ...]:
        documents = normalize_repository(inventory, resolved, acquired)
        return () if resolved.repository == ".github" else documents

    with pytest.raises(CorpusBuildError, match="exactly match the complete acquisition"):
        build_corpus(
            SOURCES.read_bytes(),
            retrieval_config,
            created_at=CREATED_AT,
            profile="core_docs_policy",
            functions=_functions(calls, normalize=omit_path),
        )
    assert not any(action in {"create", "verify"} for action, _ in calls)


def test_failure_never_exposes_or_verifies_a_partial_bundle(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    calls: list[tuple[str, str]] = []

    def fail_acquisition(resolved: ResolvedRevision) -> AcquiredRepository:
        if resolved.repository == "community":
            raise RuntimeError("exact tree unavailable")
        return _acquired(resolved)

    with pytest.raises(CorpusBuildError, match="exact tree unavailable"):
        build_corpus(
            SOURCES.read_bytes(),
            retrieval_config,
            created_at=CREATED_AT,
            profile="core_docs_policy",
            functions=_functions(calls, acquire=fail_acquisition),
        )
    assert not any(action in {"create", "verify"} for action, _ in calls)


def test_creation_failure_is_atomic_and_does_not_run_verification(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    calls: list[tuple[str, str]] = []

    def fail_creation(*args: object, **kwargs: object) -> GenerationBundle:
        del args, kwargs
        raise RuntimeError("generation rejected")

    with pytest.raises(CorpusBuildError, match="generation rejected"):
        build_corpus(
            SOURCES.read_bytes(),
            retrieval_config,
            created_at=CREATED_AT,
            profile="clients",
            functions=_functions(calls, create=fail_creation),
        )
    assert calls.count(("create", "generation")) == 1
    assert ("verify", "generation") not in calls


def test_identical_complete_inputs_are_deterministic(
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    first_calls: list[tuple[str, str]] = []
    second_calls: list[tuple[str, str]] = []
    first = build_corpus(
        SOURCES.read_bytes(),
        retrieval_config,
        created_at=CREATED_AT,
        profile="clients",
        functions=_functions(first_calls),
    )
    second = build_corpus(
        SOURCES.read_bytes(),
        retrieval_config,
        created_at=CREATED_AT,
        profile="clients",
        functions=_functions(second_calls),
    )

    assert first == second
    assert first_calls == second_calls


def test_profile_scope_selects_exactly_one_reviewed_family(
    source_inventory: dict[str, object],
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    calls: list[tuple[str, str]] = []
    bundle = build_corpus(
        SOURCES.read_bytes(),
        retrieval_config,
        created_at=CREATED_AT,
        profile="clients",
        functions=_functions(calls),
    )
    expected = {
        cast(str, entry["name"])
        for entry in _entries(source_inventory)
        if entry["family"] == "clients" and entry["classification"] == "curated"
    }

    assert {document.source.repository for document in bundle.document_templates} == expected
    assert {name for action, name in calls if action == "resolve"} == expected
    assert bundle.structured_acquisitions == ()
    assert bundle.structured_records == ()

    with pytest.raises(CorpusBuildError, match="selects no reviewed sources"):
        build_corpus(
            SOURCES.read_bytes(),
            retrieval_config,
            created_at=CREATED_AT,
            profile="excluded",
            functions=_functions([]),
        )


def test_reports_every_skipped_path_on_stderr_and_nothing_on_stdout(
    capsys: pytest.CaptureFixture[str],
    retrieval_config: FrozenRetrievalConfiguration,
) -> None:
    """A file that policy included but that cannot become a document is skipped, not fatal. That
    is only safe if it is stated: this runs unattended once a week, and silently missing content
    is indistinguishable from content that was never there. stderr, because stdout carries the
    canonical report."""
    calls: list[tuple[str, str]] = []

    def acquire(resolved: ResolvedRevision) -> AcquiredRepository:
        acquired = _acquired(resolved)
        if resolved.repository == ".github":
            return AcquiredRepository(
                acquired.repository,
                acquired.commit,
                acquired.files,
                acquired.total_bytes,
                skipped=(("bad.txt", "not valid UTF-8"), ("empty.md", "empty")),
            )
        return acquired

    build_corpus(
        SOURCES.read_bytes(),
        retrieval_config,
        created_at=CREATED_AT,
        functions=_functions(calls, acquire=acquire),
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == [
        "skipped .github/bad.txt: not valid UTF-8",
        "skipped .github/empty.md: empty",
    ]
