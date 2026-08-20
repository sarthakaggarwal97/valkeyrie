"""Load and verify the bounded, frozen beta retrieval configuration."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml

from valkeyrie.evaluations import (
    EvaluationError,
    EvaluationSuite,
    RetrievalFixture,
    calculate_ndcg_at_10,
    calculate_recall_at_5,
    compute_retrieval_fixture_revision,
    load_evaluation_suite,
)
from valkeyrie.sources import _load_yaml_mapping


class RetrievalConfigError(ValueError):
    """The retrieval qualification artifact is malformed or no longer frozen."""


@dataclass(frozen=True)
class EmbeddingConfiguration:
    """Bedrock properties plus the pinned model-defined output invariant."""

    model_id: str
    dimensions: int
    output_normalization: str

    @property
    def normalize(self) -> bool:
        """Derive compatibility behavior; this is not a configurable AWS property."""
        return self.output_normalization == "model_defined"


@dataclass(frozen=True)
class ChunkingConfiguration:
    """Bedrock fixed-size chunking settings."""

    strategy: str
    max_tokens: int
    overlap_percentage: int


@dataclass(frozen=True)
class IndexConfiguration:
    """OpenSearch vector mapping settings."""

    dimensions: int
    engine: str
    algorithm: str
    distance_metric: str
    vector_field: str
    text_field: str
    metadata_field: str


@dataclass(frozen=True)
class RetrievalParameters:
    """Operational Bedrock Knowledge Base retrieval settings."""

    search_type: str
    number_of_results: int
    generation_filter_field: str
    exact_identifier_route: str
    unavailable_generation_behavior: str
    reranking: bool


@dataclass(frozen=True)
class CandidateConfiguration:
    """One complete configuration compared by the local qualification."""

    embedding: EmbeddingConfiguration
    chunking: ChunkingConfiguration
    index: IndexConfiguration
    retrieval: RetrievalParameters


@dataclass(frozen=True)
class CandidateScore:
    """Deterministic aggregates and gate result for one candidate."""

    candidate_id: str
    qualifies: bool
    overall_recall_at_5: float
    overall_ndcg_at_10: float
    p95_latency_seconds: float
    family_recall_at_5: Mapping[str, float]
    failed_gates: tuple[str, ...]


@dataclass(frozen=True)
class CandidateQualification:
    """One candidate's fixed settings and locally recorded score."""

    candidate_id: str
    configuration: CandidateConfiguration
    score: CandidateScore


@dataclass(frozen=True)
class FrozenRetrievalConfiguration:
    """The single selected configuration frozen through the public beta."""

    config_revision: str
    selected_candidate: str
    selected: CandidateConfiguration
    candidates: tuple[CandidateQualification, ...]


_MAX_ARTIFACT_BYTES = 512 * 1024
_MAX_CANDIDATES = 8
_MAX_LATENCY_SECONDS = 60.0
_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_FROZEN_BETA_REVISION = "sha256:b0db86c993c9147ff0078a78c55163b8a6ba462e400ce4fad9ab4b3263d48223"
_FROZEN_RUNTIME_CONFIGURATION_DIGEST = (
    "sha256:94059e09803aa505a6e77460fcee45cba9675436d0c2272afd6253497139ebf2"
)
_TOP_FIELDS = {
    "api_version",
    "kind",
    "status",
    "qualification",
    "selection",
    "freeze",
    "revision",
    "candidates",
}
_CONFIGURATION_FIELDS = {"embedding", "chunking", "index", "retrieval"}
_MEASUREMENT_FIELDS = {
    "fixture_id",
    "route_selected",
    "exact_identifier_found",
    "generation_filter_present",
    "cross_generation_leaks",
    "excluded_path_leaks",
    "prompt_or_evaluation_leaks",
    "malformed_or_unverifiable_metadata",
    "ranked_evidence_ids",
    "measured_latency_seconds",
    "failed_closed",
}
_SELECTION_QUALITY_ORDER = [
    "overall_recall_at_5_desc",
    "overall_ndcg_at_10_desc",
    "p95_latency_seconds_asc",
]
_FREEZE_MESSAGE = (
    "retrieval configuration is frozen through beta; change requires a separately "
    "approved full-index migration"
)


def load_retrieval_config(
    path: Path, *, project_root: Path | None = None
) -> FrozenRetrievalConfiguration:
    """Load the local artifact and enforce qualification, revision, and beta freeze."""
    document = _read_artifact(path)
    try:
        suite = load_evaluation_suite(project_root or path.parent)
    except EvaluationError as error:
        raise RetrievalConfigError(f"approved retrieval inputs are invalid: {error}") from error
    return validate_retrieval_config(document, suite)


def validate_retrieval_config(
    document: Mapping[str, object], suite: EvaluationSuite
) -> FrozenRetrievalConfiguration:
    """Validate one already-loaded artifact against the approved local suite."""
    _exact_fields(document, _TOP_FIELDS, "retrieval configuration")
    if (
        document.get("api_version"),
        document.get("kind"),
        document.get("status"),
    ) != (
        "valkeyrie.io/retrieval-config/1",
        "FrozenRetrievalConfiguration",
        "reviewed_frozen_beta",
    ):
        raise RetrievalConfigError("retrieval configuration has an incompatible identity")

    qualification = _mapping(document.get("qualification"), "qualification")
    _exact_fields(
        qualification,
        {
            "label",
            "claim_scope",
            "candidate_limit",
            "fixture_suite",
            "fixture_revision",
            "criteria",
        },
        "qualification",
    )
    expected_qualification = {
        "label": "bounded_local_synthetic_qualification",
        "claim_scope": "local_synthetic_measurements_not_deployed_production_claims",
        "fixture_suite": "evals/retrieval.yaml",
        "criteria": "evals/criteria/retrieval.yaml",
    }
    if any(qualification.get(field) != value for field, value in expected_qualification.items()):
        raise RetrievalConfigError(
            "qualification must remain bounded local synthetic evidence, not a production claim"
        )
    if qualification.get("fixture_revision") != compute_retrieval_fixture_revision(
        suite.retrieval_fixtures
    ):
        raise RetrievalConfigError(
            "qualification does not bind the exact reviewed synthetic fixture semantics"
        )
    candidate_limit = _integer(
        qualification.get("candidate_limit"), "qualification candidate_limit", 2, _MAX_CANDIDATES
    )

    selection = _mapping(document.get("selection"), "selection")
    _exact_fields(
        selection,
        {"eligibility", "quality_order", "tie_break", "selected_candidate"},
        "selection",
    )
    if (
        selection.get("eligibility") != "passes_all_approved_retrieval_gates"
        or _strings(selection.get("quality_order"), "selection quality_order")
        != _SELECTION_QUALITY_ORDER
        or selection.get("tie_break") != "candidate_id_lexical_asc"
    ):
        raise RetrievalConfigError(
            "selection must use the reviewed quality-first order and tie-break"
        )
    selected_candidate = _identifier(selection.get("selected_candidate"), "selected_candidate")

    freeze = _mapping(document.get("freeze"), "freeze")
    _exact_fields(
        freeze,
        {"lifecycle", "routine_change", "migration_requirement"},
        "freeze",
    )
    if freeze != {
        "lifecycle": "through_public_beta",
        "routine_change": "prohibited",
        "migration_requirement": "separately_approved_full_index_migration",
    }:
        raise RetrievalConfigError(_FREEZE_MESSAGE)

    revision = _mapping(document.get("revision"), "revision")
    _exact_fields(
        revision,
        {
            "algorithm",
            "derived_field",
            "config_revision_is_not_part_of_preimage",
            "config_revision",
        },
        "revision",
    )
    if (
        revision.get("algorithm") != "sha256_canonical_semantics"
        or revision.get("derived_field") != "config_revision"
        or type(revision.get("config_revision_is_not_part_of_preimage")) is not bool
        or revision.get("config_revision_is_not_part_of_preimage") is not True
    ):
        raise RetrievalConfigError("revision semantics must exclude only their self-reference")
    declared_revision = revision.get("config_revision")
    if not isinstance(declared_revision, str) or _DIGEST.fullmatch(declared_revision) is None:
        raise RetrievalConfigError("revision config_revision must be a sha256 digest")

    raw_candidates = _list(document.get("candidates"), "candidates")
    if len(raw_candidates) != candidate_limit:
        raise RetrievalConfigError("candidate_limit must equal the complete explicit candidate set")
    fixtures = {fixture.fixture_id: fixture for fixture in suite.retrieval_fixtures}
    candidates = tuple(
        sorted(
            (_candidate(raw, fixtures, suite) for raw in raw_candidates),
            key=lambda candidate: candidate.candidate_id,
        )
    )
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RetrievalConfigError("candidate IDs must be unique")
    if len({candidate.configuration for candidate in candidates}) != len(candidates):
        raise RetrievalConfigError("candidate configurations must be unique")

    qualifying = [candidate for candidate in candidates if candidate.score.qualifies]
    if len(qualifying) != 1:
        raise RetrievalConfigError(
            "reviewed artifact must contain exactly one qualifying candidate"
        )
    expected_selection = select_candidate([candidate.score for candidate in candidates])
    if selected_candidate != expected_selection:
        raise RetrievalConfigError(
            "selected_candidate does not match deterministic quality-first selection"
        )

    computed_revision = compute_config_revision(document)
    if declared_revision != computed_revision:
        raise RetrievalConfigError("declared config_revision does not match canonical semantics")
    if declared_revision != _FROZEN_BETA_REVISION:
        raise RetrievalConfigError(_FREEZE_MESSAGE)

    selected = next(
        candidate.configuration
        for candidate in candidates
        if candidate.candidate_id == selected_candidate
    )
    return validate_frozen_retrieval_configuration(
        FrozenRetrievalConfiguration(
            config_revision=declared_revision,
            selected_candidate=selected_candidate,
            selected=selected,
            candidates=candidates,
        )
    )


def validate_frozen_retrieval_configuration(
    config: FrozenRetrievalConfiguration,
) -> FrozenRetrievalConfiguration:
    """Reject any runtime object other than the exact reviewed frozen configuration."""
    if type(config) is not FrozenRetrievalConfiguration:
        raise RetrievalConfigError("runtime retrieval configuration is not the frozen type")
    if config.config_revision != _FROZEN_BETA_REVISION:
        raise RetrievalConfigError(_FREEZE_MESSAGE)
    if _runtime_configuration_digest(config) != _FROZEN_RUNTIME_CONFIGURATION_DIGEST:
        raise RetrievalConfigError(
            "runtime retrieval configuration differs from the exact reviewed frozen selection"
        )
    return config


def _runtime_configuration_digest(config: FrozenRetrievalConfiguration) -> str:
    def require_type(value: object, expected: type[object], name: str) -> None:
        if type(value) is not expected:
            raise RetrievalConfigError(f"runtime {name} has an unexpected type")

    def embedding(value: EmbeddingConfiguration) -> dict[str, object]:
        require_type(value, EmbeddingConfiguration, "embedding configuration")
        return {
            "model_id": value.model_id,
            "dimensions": value.dimensions,
            "output_normalization": value.output_normalization,
        }

    def chunking(value: ChunkingConfiguration) -> dict[str, object]:
        require_type(value, ChunkingConfiguration, "chunking configuration")
        return {
            "strategy": value.strategy,
            "max_tokens": value.max_tokens,
            "overlap_percentage": value.overlap_percentage,
        }

    def index(value: IndexConfiguration) -> dict[str, object]:
        require_type(value, IndexConfiguration, "index configuration")
        return {
            "dimensions": value.dimensions,
            "engine": value.engine,
            "algorithm": value.algorithm,
            "distance_metric": value.distance_metric,
            "vector_field": value.vector_field,
            "text_field": value.text_field,
            "metadata_field": value.metadata_field,
        }

    def retrieval(value: RetrievalParameters) -> dict[str, object]:
        require_type(value, RetrievalParameters, "retrieval parameters")
        return {
            "search_type": value.search_type,
            "number_of_results": value.number_of_results,
            "generation_filter_field": value.generation_filter_field,
            "exact_identifier_route": value.exact_identifier_route,
            "unavailable_generation_behavior": value.unavailable_generation_behavior,
            "reranking": value.reranking,
        }

    def configuration(value: CandidateConfiguration) -> dict[str, object]:
        require_type(value, CandidateConfiguration, "candidate configuration")
        return {
            "embedding": embedding(value.embedding),
            "chunking": chunking(value.chunking),
            "index": index(value.index),
            "retrieval": retrieval(value.retrieval),
        }

    def score(value: CandidateScore) -> dict[str, object]:
        require_type(value, CandidateScore, "candidate score")
        families = _mapping(value.family_recall_at_5, "runtime family recall")
        return {
            "candidate_id": value.candidate_id,
            "qualifies": value.qualifies,
            "overall_recall_at_5": value.overall_recall_at_5,
            "overall_ndcg_at_10": value.overall_ndcg_at_10,
            "p95_latency_seconds": value.p95_latency_seconds,
            "family_recall_at_5": dict(sorted(families.items())),
            "failed_gates": list(value.failed_gates),
        }

    require_type(config.candidates, tuple, "candidate collection")
    candidates = []
    for candidate in config.candidates:
        require_type(candidate, CandidateQualification, "candidate qualification")
        candidates.append(
            {
                "candidate_id": candidate.candidate_id,
                "configuration": configuration(candidate.configuration),
                "score": score(candidate.score),
            }
        )
    semantics = {
        "config_revision": config.config_revision,
        "selected_candidate": config.selected_candidate,
        "selected": configuration(config.selected),
        "candidates": candidates,
    }
    try:
        encoded = json.dumps(
            semantics,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise RetrievalConfigError("runtime retrieval configuration is not canonical") from error
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def select_candidate(scores: Sequence[CandidateScore]) -> str:
    """Select qualifying quality first, then break an exact tie by lexical ID."""
    if (
        not isinstance(scores, Sequence)
        or isinstance(scores, (str, bytes))
        or not 1 <= len(scores) <= _MAX_CANDIDATES
    ):
        raise RetrievalConfigError("candidate scores must be a bounded non-empty sequence")
    candidate_ids: list[str] = []
    qualifying: list[CandidateScore] = []
    for score in scores:
        if not isinstance(score, CandidateScore):
            raise RetrievalConfigError("candidate score is malformed")
        candidate_id = _identifier(score.candidate_id, "candidate score ID")
        qualifies = _boolean(score.qualifies, f"{candidate_id} qualifies")
        _number(score.overall_recall_at_5, f"{candidate_id} recall", 0, 1)
        _number(score.overall_ndcg_at_10, f"{candidate_id} ndcg", 0, 1)
        _number(
            score.p95_latency_seconds,
            f"{candidate_id} p95 latency",
            0,
            _MAX_LATENCY_SECONDS,
        )
        families = _mapping(score.family_recall_at_5, f"{candidate_id} family recall")
        if not families:
            raise RetrievalConfigError("candidate score family recall must not be empty")
        for family, value in families.items():
            _text(family, "candidate score family")
            _number(value, f"{candidate_id} {family} recall", 0, 1)
        if not isinstance(score.failed_gates, tuple) or any(
            not isinstance(gate, str) or not gate for gate in score.failed_gates
        ):
            raise RetrievalConfigError("candidate score failed_gates is malformed")
        if qualifies == bool(score.failed_gates):
            raise RetrievalConfigError("candidate score qualification conflicts with failed_gates")
        candidate_ids.append(candidate_id)
        if qualifies:
            qualifying.append(score)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RetrievalConfigError("candidate score IDs must be unique")
    if not qualifying:
        raise RetrievalConfigError("no candidate passes every approved retrieval gate")
    return min(
        qualifying,
        key=lambda score: (
            -score.overall_recall_at_5,
            -score.overall_ndcg_at_10,
            score.p95_latency_seconds,
            score.candidate_id,
        ),
    ).candidate_id


def compute_config_revision(document: Mapping[str, object]) -> str:
    """Hash canonical semantics, excluding only the derived revision value."""
    if not isinstance(document, Mapping):
        raise RetrievalConfigError("retrieval configuration must be a mapping")
    try:
        canonical = copy.deepcopy(dict(document))
        revision = _mapping(canonical.get("revision"), "revision")
        revision_copy = dict(revision)
        if "config_revision" not in revision_copy:
            raise RetrievalConfigError("revision is missing config_revision")
        revision_copy.pop("config_revision")
        canonical["revision"] = revision_copy
        raw_candidates = _list(canonical.get("candidates"), "candidates")
        candidate_copies: list[dict[str, object]] = []
        for raw_candidate in raw_candidates:
            candidate = dict(_mapping(raw_candidate, "candidate"))
            measurements = [
                dict(_mapping(value, "measurement"))
                for value in _list(candidate.get("measurements"), "measurements")
            ]
            candidate["measurements"] = sorted(
                measurements, key=lambda value: _text(value.get("fixture_id"), "fixture_id")
            )
            candidate_copies.append(candidate)
        canonical["candidates"] = sorted(
            candidate_copies, key=lambda value: _text(value.get("id"), "candidate id")
        )
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        if isinstance(error, RetrievalConfigError):
            raise
        raise RetrievalConfigError(
            "retrieval configuration is not canonically encodable"
        ) from error
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _read_artifact(path: Path) -> dict[str, object]:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise RetrievalConfigError("retrieval configuration must be a regular file")
        if metadata.st_size < 1 or metadata.st_size > _MAX_ARTIFACT_BYTES:
            raise RetrievalConfigError("retrieval configuration exceeds its local file bound")
        content = path.read_bytes()
    except RetrievalConfigError:
        raise
    except OSError as error:
        raise RetrievalConfigError(f"cannot read retrieval configuration: {error}") from error
    if len(content) < 1 or len(content) > _MAX_ARTIFACT_BYTES:
        raise RetrievalConfigError("retrieval configuration exceeds its local file bound")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RetrievalConfigError("retrieval configuration is not UTF-8") from error
    if "\r" in text:
        raise RetrievalConfigError("retrieval configuration must use LF newlines")
    try:
        return _load_yaml_mapping(text, str(path), reject_merge_keys=True)
    except (UnicodeError, yaml.YAMLError, ValueError) as error:
        raise RetrievalConfigError(f"cannot parse retrieval configuration: {error}") from error


def _candidate(
    raw: object,
    fixtures: Mapping[str, RetrievalFixture],
    suite: EvaluationSuite,
) -> CandidateQualification:
    value = _mapping(raw, "candidate")
    _exact_fields(value, {"id", "configuration", "measurements"}, "candidate")
    candidate_id = _identifier(value.get("id"), "candidate id")
    configuration = _configuration(value.get("configuration"), candidate_id)
    measurements = _measurements(value.get("measurements"), candidate_id, fixtures)
    score = _score(candidate_id, measurements, fixtures, suite)
    return CandidateQualification(candidate_id, configuration, score)


def _configuration(raw: object, candidate_id: str) -> CandidateConfiguration:
    value = _mapping(raw, f"{candidate_id} configuration")
    _exact_fields(value, _CONFIGURATION_FIELDS, f"{candidate_id} configuration")

    embedding = _mapping(value.get("embedding"), f"{candidate_id} embedding")
    _exact_fields(
        embedding,
        {"model_id", "dimensions", "output_normalization"},
        "embedding",
    )
    model_id = _text(embedding.get("model_id"), "embedding model_id")
    if model_id != "amazon.titan-embed-text-v2:0":
        raise RetrievalConfigError("embedding model_id is not the reviewed Bedrock model")
    dimensions = _integer(embedding.get("dimensions"), "embedding dimensions", 256, 1024)
    if dimensions not in {256, 512, 1024}:
        raise RetrievalConfigError("embedding dimensions are unsupported by the reviewed model")
    output_normalization = _text(
        embedding.get("output_normalization"), "embedding output_normalization"
    )
    if output_normalization != "model_defined":
        raise RetrievalConfigError(
            "embedding output normalization must remain the pinned model-defined invariant"
        )

    chunking = _mapping(value.get("chunking"), f"{candidate_id} chunking")
    _exact_fields(
        chunking,
        {"strategy", "max_tokens", "overlap_percentage"},
        "chunking",
    )
    if chunking.get("strategy") != "FIXED_SIZE":
        raise RetrievalConfigError("chunking strategy must be FIXED_SIZE")
    max_tokens = _integer(chunking.get("max_tokens"), "chunking max_tokens", 20, 8192)
    overlap = _integer(chunking.get("overlap_percentage"), "chunking overlap_percentage", 0, 99)

    index = _mapping(value.get("index"), f"{candidate_id} index")
    _exact_fields(
        index,
        {
            "dimensions",
            "engine",
            "algorithm",
            "distance_metric",
            "vector_field",
            "text_field",
            "metadata_field",
        },
        "index",
    )
    index_dimensions = _integer(index.get("dimensions"), "index dimensions", 256, 1024)
    fixed_index = {
        "engine": "faiss",
        "algorithm": "hnsw",
        "distance_metric": "l2",
        "vector_field": "bedrock-knowledge-base-default-vector",
        "text_field": "AMAZON_BEDROCK_TEXT_CHUNK",
        "metadata_field": "AMAZON_BEDROCK_METADATA",
    }
    if index_dimensions != dimensions or any(
        index.get(field) != expected for field, expected in fixed_index.items()
    ):
        raise RetrievalConfigError("index mapping is incompatible with the embedding configuration")

    retrieval = _mapping(value.get("retrieval"), f"{candidate_id} retrieval")
    _exact_fields(
        retrieval,
        {
            "search_type",
            "number_of_results",
            "generation_filter_field",
            "exact_identifier_route",
            "unavailable_generation_behavior",
            "reranking",
        },
        "retrieval",
    )
    reranking = _boolean(retrieval.get("reranking"), "retrieval reranking")
    fixed_retrieval = {
        "search_type": "HYBRID",
        "generation_filter_field": "generation_id",
        "exact_identifier_route": "deterministic_exact_lookup",
        "unavailable_generation_behavior": "fail_closed",
    }
    if reranking or any(
        retrieval.get(field) != expected for field, expected in fixed_retrieval.items()
    ):
        raise RetrievalConfigError(
            "operational retrieval settings differ from the reviewed boundary"
        )
    number_of_results = _integer(
        retrieval.get("number_of_results"), "retrieval number_of_results", 1, 100
    )

    return CandidateConfiguration(
        embedding=EmbeddingConfiguration(model_id, dimensions, output_normalization),
        chunking=ChunkingConfiguration("FIXED_SIZE", max_tokens, overlap),
        index=IndexConfiguration(
            index_dimensions,
            cast(str, index["engine"]),
            cast(str, index["algorithm"]),
            cast(str, index["distance_metric"]),
            cast(str, index["vector_field"]),
            cast(str, index["text_field"]),
            cast(str, index["metadata_field"]),
        ),
        retrieval=RetrievalParameters(
            "HYBRID",
            number_of_results,
            "generation_id",
            "deterministic_exact_lookup",
            "fail_closed",
            False,
        ),
    )


def _measurements(
    raw: object,
    candidate_id: str,
    fixtures: Mapping[str, RetrievalFixture],
) -> dict[str, Mapping[str, object]]:
    values = _list(raw, f"{candidate_id} measurements")
    parsed: dict[str, Mapping[str, object]] = {}
    for raw_measurement in values:
        measurement = _mapping(raw_measurement, f"{candidate_id} measurement")
        _exact_fields(measurement, _MEASUREMENT_FIELDS, f"{candidate_id} measurement")
        fixture_id = _identifier(measurement.get("fixture_id"), "measurement fixture_id")
        if fixture_id in parsed:
            raise RetrievalConfigError(f"{candidate_id} repeats fixture {fixture_id}")
        fixture = fixtures.get(fixture_id)
        if fixture is None:
            raise RetrievalConfigError(f"{candidate_id} has unknown fixture {fixture_id}")
        _validate_measurement(measurement, fixture)
        parsed[fixture_id] = measurement
    if set(parsed) != set(fixtures):
        raise RetrievalConfigError(
            f"{candidate_id} measurements must contain every approved fixture exactly once"
        )
    return parsed


def _validate_measurement(measurement: Mapping[str, object], fixture: RetrievalFixture) -> None:
    fixture_id = fixture.fixture_id
    for field in ("route_selected", "generation_filter_present"):
        _boolean(measurement.get(field), f"{fixture_id} {field}")
    for field in (
        "cross_generation_leaks",
        "excluded_path_leaks",
        "prompt_or_evaluation_leaks",
        "malformed_or_unverifiable_metadata",
    ):
        _integer(measurement.get(field), f"{fixture_id} {field}", 0, 1_000_000)
    ranked = _ranked_evidence_ids(
        measurement.get("ranked_evidence_ids"), f"{fixture_id} ranked evidence"
    )
    _number(
        measurement.get("measured_latency_seconds"),
        f"{fixture_id} measured_latency_seconds",
        0,
        _MAX_LATENCY_SECONDS,
    )

    exact = measurement.get("exact_identifier_found")
    if fixture.exact_identifier:
        _boolean(exact, f"{fixture_id} exact_identifier_found")
    elif exact is not None:
        raise RetrievalConfigError(f"{fixture_id} records an inapplicable exact lookup")

    failed_closed = measurement.get("failed_closed")
    if fixture.generation_available:
        if failed_closed is not None:
            raise RetrievalConfigError(f"{fixture_id} records inapplicable fail-closed behavior")
    else:
        if ranked:
            raise RetrievalConfigError(f"{fixture_id} ranks unavailable-generation evidence")
        _boolean(failed_closed, f"{fixture_id} failed_closed")


def _score(
    candidate_id: str,
    measurements: Mapping[str, Mapping[str, object]],
    fixtures: Mapping[str, RetrievalFixture],
    suite: EvaluationSuite,
) -> CandidateScore:
    available = [fixture for fixture in fixtures.values() if fixture.generation_available]
    exact = [fixture for fixture in fixtures.values() if fixture.exact_identifier]
    unavailable = [fixture for fixture in fixtures.values() if not fixture.generation_available]

    def ranked(fixture: RetrievalFixture) -> tuple[str, ...]:
        return cast(tuple[str, ...], measurements[fixture.fixture_id]["ranked_evidence_ids"])

    def recall(fixture: RetrievalFixture) -> float:
        return calculate_recall_at_5(fixture.expected_evidence, ranked(fixture))

    overall_recall = math.fsum(recall(fixture) for fixture in available) / len(available)
    overall_ndcg = math.fsum(
        calculate_ndcg_at_10(fixture.expected_evidence, ranked(fixture)) for fixture in available
    ) / len(available)
    latencies = sorted(
        float(cast(float, measurement["measured_latency_seconds"]))
        for measurement in measurements.values()
    )
    p95_latency = latencies[math.ceil(0.95 * len(latencies)) - 1]
    families = {
        family: math.fsum(recall(fixture) for fixture in available if fixture.family == family)
        / sum(fixture.family == family for fixture in available)
        for family in suite.required_families
    }

    threshold = suite.thresholds
    failed: list[str] = []
    if not all(cast(bool, item["route_selected"]) for item in measurements.values()):
        failed.append("route_selection_rate")
    if not all(
        cast(bool, measurements[item.fixture_id]["exact_identifier_found"]) for item in exact
    ):
        failed.append("exact_identifier_lookup_rate")
    if not all(cast(bool, item["generation_filter_present"]) for item in measurements.values()):
        failed.append("generation_filter_presence_rate")
    for field, gate in (
        ("cross_generation_leaks", "cross_generation_leaks"),
        ("excluded_path_leaks", "excluded_path_leaks"),
        ("prompt_or_evaluation_leaks", "prompt_or_evaluation_leaks"),
        ("malformed_or_unverifiable_metadata", "malformed_or_unverifiable_metadata"),
    ):
        if sum(cast(int, item[field]) for item in measurements.values()) != 0:
            failed.append(gate)
    if overall_recall < threshold["retrieval.recall.min"]:
        failed.append("canonical_expected_evidence_recall_at_5")
    if overall_ndcg < threshold["retrieval.ndcg.min"]:
        failed.append("normalized_discounted_cumulative_gain_at_10")
    failed.extend(
        f"family_recall_at_5.{family}"
        for family, actual in families.items()
        if actual < threshold["retrieval.family.min"]
    )
    if p95_latency > threshold["retrieval.latency.max"]:
        failed.append("fixture_p95_latency_seconds")
    if not all(cast(bool, measurements[item.fixture_id]["failed_closed"]) for item in unavailable):
        failed.append("missing_or_unavailable_generation")

    return CandidateScore(
        candidate_id=candidate_id,
        qualifies=not failed,
        overall_recall_at_5=overall_recall,
        overall_ndcg_at_10=overall_ndcg,
        p95_latency_seconds=p95_latency,
        family_recall_at_5=MappingProxyType(dict(sorted(families.items()))),
        failed_gates=tuple(sorted(failed)),
    )


def _ranked_evidence_ids(value: object, name: str) -> tuple[str, ...]:
    identifiers = tuple(_identifier(item, name) for item in _list(value, name))
    if len(identifiers) > 10 or len(identifiers) != len(set(identifiers)):
        raise RetrievalConfigError(f"{name} must contain at most ten unique evidence IDs")
    return identifiers


def _exact_fields(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise RetrievalConfigError(f"{name} has an unknown or missing field")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise RetrievalConfigError(f"{name} must be a mapping with string keys")
    return cast(Mapping[str, object], value)


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise RetrievalConfigError(f"{name} must be a list")
    return cast(list[object], value)


def _strings(value: object, name: str) -> list[str]:
    return [_text(item, name) for item in _list(value, name)]


def _identifier(value: object, name: str) -> str:
    result = _text(value, name)
    if _ID.fullmatch(result) is None:
        raise RetrievalConfigError(f"{name} must use lowercase letters, digits, and hyphens")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalConfigError(f"{name} must be a non-blank string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise RetrievalConfigError(f"{name} must be valid UTF-8") from error
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise RetrievalConfigError(f"{name} must be a boolean")
    return value


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise RetrievalConfigError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _number(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetrievalConfigError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise RetrievalConfigError(
            f"{name} must be a finite number between {minimum} and {maximum}"
        )
    return result
