from __future__ import annotations

import copy
import math
import shutil
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest
import yaml

from valkeyrie.evaluations import load_evaluation_suite
from valkeyrie.retrieval_config import (
    CandidateScore,
    RetrievalConfigError,
    compute_config_revision,
    load_retrieval_config,
    select_candidate,
    validate_frozen_retrieval_configuration,
    validate_retrieval_config,
)
from valkeyrie.sources import load_yaml_mapping

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "retrieval-config.yaml"
REVISION = "sha256:b0db86c993c9147ff0078a78c55163b8a6ba462e400ce4fad9ab4b3263d48223"
FIXTURE_REVISION = "sha256:94713f4d1c98d2049d04c93a514c1db6f13e282e48ecf61e9dba55117e70d805"
SELECTED = "titan-v2-1024-fixed-300-20"


def _document() -> dict[str, object]:
    return copy.deepcopy(load_yaml_mapping(CONFIG))


def _candidates(document: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], document["candidates"])


def _candidate(document: dict[str, object], candidate_id: str = SELECTED) -> dict[str, object]:
    return next(candidate for candidate in _candidates(document) if candidate["id"] == candidate_id)


def _measurements(candidate: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], candidate["measurements"])


def _measurement(candidate: dict[str, object], fixture_id: str) -> dict[str, object]:
    return next(
        measurement
        for measurement in _measurements(candidate)
        if measurement["fixture_id"] == fixture_id
    )


def _configuration(candidate: dict[str, object]) -> dict[str, dict[str, object]]:
    return cast(dict[str, dict[str, object]], candidate["configuration"])


def _write(tmp_path: Path, document: dict[str, object], *, rehash: bool = False) -> Path:
    if rehash:
        revision = cast(dict[str, object], document["revision"])
        revision["config_revision"] = compute_config_revision(document)
    path = tmp_path / "retrieval-config.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _load(tmp_path: Path, document: dict[str, object]) -> object:
    return load_retrieval_config(_write(tmp_path, document), project_root=ROOT)


def _score(
    candidate_id: str,
    *,
    recall: float,
    ndcg: float,
    latency: float,
    qualifies: bool = True,
) -> CandidateScore:
    return CandidateScore(
        candidate_id=candidate_id,
        qualifies=qualifies,
        overall_recall_at_5=recall,
        overall_ndcg_at_10=ndcg,
        p95_latency_seconds=latency,
        family_recall_at_5=MappingProxyType({"family": recall}),
        failed_gates=() if qualifies else ("gate",),
    )


def test_loads_exact_frozen_selected_configuration_and_local_qualification() -> None:
    artifact = load_retrieval_config(CONFIG)

    assert artifact.config_revision == REVISION
    assert artifact.selected_candidate == SELECTED
    assert artifact.selected.embedding.model_id == "amazon.titan-embed-text-v2:0"
    assert artifact.selected.embedding.dimensions == 1024
    assert artifact.selected.embedding.output_normalization == "model_defined"
    assert artifact.selected.embedding.normalize is True
    document = _document()
    for candidate in _candidates(document):
        embedding = _configuration(candidate)["embedding"]
        assert set(embedding) == {"model_id", "dimensions", "output_normalization"}
        assert embedding["output_normalization"] == "model_defined"
    assert artifact.selected.chunking.strategy == "FIXED_SIZE"
    assert artifact.selected.chunking.max_tokens == 300
    assert artifact.selected.chunking.overlap_percentage == 20
    assert artifact.selected.index.dimensions == 1024
    assert artifact.selected.index.engine == "faiss"
    assert artifact.selected.index.algorithm == "hnsw"
    assert artifact.selected.index.distance_metric == "l2"
    assert artifact.selected.index.vector_field == "bedrock-knowledge-base-default-vector"
    assert artifact.selected.index.text_field == "AMAZON_BEDROCK_TEXT_CHUNK"
    assert artifact.selected.index.metadata_field == "AMAZON_BEDROCK_METADATA"
    assert artifact.selected.retrieval.search_type == "HYBRID"
    assert artifact.selected.retrieval.number_of_results == 10
    assert artifact.selected.retrieval.generation_filter_field == "generation_id"
    assert artifact.selected.retrieval.exact_identifier_route == "deterministic_exact_lookup"
    assert artifact.selected.retrieval.unavailable_generation_behavior == "fail_closed"
    assert artifact.selected.retrieval.reranking is False

    scores = {candidate.candidate_id: candidate.score for candidate in artifact.candidates}
    assert [candidate.candidate_id for candidate in artifact.candidates] == [
        "titan-v2-1024-fixed-300-20",
        "titan-v2-1024-fixed-512-20",
        "titan-v2-512-fixed-256-15",
    ]
    assert scores[SELECTED].qualifies is True
    assert scores[SELECTED].overall_recall_at_5 == 1.0
    assert scores[SELECTED].overall_ndcg_at_10 == 1.0
    assert scores[SELECTED].p95_latency_seconds == 0.56
    assert scores[SELECTED].failed_gates == ()
    assert scores["titan-v2-1024-fixed-512-20"].failed_gates == (
        "normalized_discounted_cumulative_gain_at_10",
    )
    assert scores["titan-v2-512-fixed-256-15"].failed_gates == (
        "canonical_expected_evidence_recall_at_5",
        "family_recall_at_5.modules",
    )
    assert sum(candidate.score.qualifies for candidate in artifact.candidates) == 1

    document = _document()
    qualification = cast(dict[str, object], document["qualification"])
    assert qualification == {
        "label": "bounded_local_synthetic_qualification",
        "claim_scope": "local_synthetic_measurements_not_deployed_production_claims",
        "candidate_limit": 3,
        "fixture_suite": "evals/retrieval.yaml",
        "fixture_revision": FIXTURE_REVISION,
        "criteria": "evals/criteria/retrieval.yaml",
    }


def test_every_candidate_has_every_fixture_once_with_typed_finite_measurements() -> None:
    document = _document()
    suite = load_evaluation_suite(ROOT)
    expected = {fixture.fixture_id for fixture in suite.retrieval_fixtures}

    for candidate in _candidates(document):
        measurements = _measurements(candidate)
        assert {cast(str, item["fixture_id"]) for item in measurements} == expected
        assert len(measurements) == len(expected)
        for measurement in measurements:
            for field in ("route_selected", "generation_filter_present"):
                assert type(measurement[field]) is bool
            for field in (
                "cross_generation_leaks",
                "excluded_path_leaks",
                "prompt_or_evaluation_leaks",
                "malformed_or_unverifiable_metadata",
            ):
                assert type(measurement[field]) is int
            ranked = cast(list[str], measurement["ranked_evidence_ids"])
            assert len(ranked) <= 10
            assert len(ranked) == len(set(ranked))
            assert math.isfinite(cast(float, measurement["measured_latency_seconds"]))
            assert "recalled_evidence_at_5" not in measurement
            assert "ndcg_at_10" not in measurement


def test_supplied_or_faked_aggregate_metrics_are_rejected(tmp_path: Path) -> None:
    for field, value in (("recalled_evidence_at_5", 1), ("ndcg_at_10", 1.0)):
        document = _document()
        _measurements(_candidate(document))[0][field] = value
        with pytest.raises(RetrievalConfigError, match="unknown or missing"):
            _load(tmp_path, document)


def test_full_runtime_validator_rejects_forged_model_with_genuine_revision() -> None:
    artifact = load_retrieval_config(CONFIG)
    validate_frozen_retrieval_configuration(artifact)
    forged_embedding = replace(
        artifact.selected.embedding,
        model_id="forged.unreviewed-model:0",
    )
    forged = replace(
        artifact,
        selected=replace(artifact.selected, embedding=forged_embedding),
    )
    assert forged.config_revision == REVISION
    with pytest.raises(RetrievalConfigError, match="exact reviewed frozen selection"):
        validate_frozen_retrieval_configuration(forged)


def test_canonical_revision_and_selection_ignore_candidate_and_fixture_input_order(
    tmp_path: Path,
) -> None:
    document = _document()
    document["candidates"] = list(reversed(_candidates(document)))
    for candidate in _candidates(document):
        candidate["measurements"] = list(reversed(_measurements(candidate)))

    assert compute_config_revision(document) == REVISION
    reordered = _load(tmp_path, document)
    assert reordered == load_retrieval_config(CONFIG)


def test_quality_first_selection_and_lexical_tie_break_are_input_order_independent() -> None:
    lower_recall = _score("lower-recall", recall=0.96, ndcg=1.0, latency=0.1)
    higher_recall = _score("higher-recall", recall=0.97, ndcg=0.85, latency=1.0)
    assert select_candidate([lower_recall, higher_recall]) == "higher-recall"

    lower_ndcg = _score("lower-ndcg", recall=0.97, ndcg=0.90, latency=0.1)
    higher_ndcg = _score("higher-ndcg", recall=0.97, ndcg=0.91, latency=1.0)
    assert select_candidate([lower_ndcg, higher_ndcg]) == "higher-ndcg"

    slower = _score("slower", recall=0.97, ndcg=0.91, latency=0.5)
    faster = _score("faster", recall=0.97, ndcg=0.91, latency=0.4)
    assert select_candidate([slower, faster]) == "faster"

    lexical_a = _score("candidate-a", recall=0.97, ndcg=0.91, latency=0.4)
    lexical_b = replace(lexical_a, candidate_id="candidate-b")
    assert select_candidate([lexical_b, lexical_a]) == "candidate-a"
    assert select_candidate([lexical_a, lexical_b]) == "candidate-a"


def test_selection_rejects_no_qualifier_duplicates_and_malformed_scores() -> None:
    failed = _score("failed", recall=1, ndcg=1, latency=0, qualifies=False)
    with pytest.raises(RetrievalConfigError, match="no candidate"):
        select_candidate([failed])
    passing = _score("passing", recall=1, ndcg=1, latency=0)
    with pytest.raises(RetrievalConfigError, match="unique"):
        select_candidate([passing, passing])
    with pytest.raises(RetrievalConfigError, match="malformed"):
        select_candidate(cast(list[CandidateScore], [object()]))
    with pytest.raises(RetrievalConfigError, match="bounded non-empty"):
        select_candidate([])
    with pytest.raises(RetrievalConfigError, match="finite number"):
        select_candidate([replace(passing, overall_recall_at_5=float("nan"))])
    with pytest.raises(RetrievalConfigError, match="boolean"):
        select_candidate([replace(passing, qualifies=cast(bool, 1))])
    with pytest.raises(RetrievalConfigError, match="conflicts"):
        select_candidate([replace(passing, failed_gates=("gate",))])


@pytest.mark.parametrize(
    "content",
    [
        "api_version: one\napi_version: two\n",
        "base: &base\n  api_version: one\nmerged:\n  <<: *base\n",
        "- not-a-mapping\n",
        "1: non-string-key\n",
    ],
)
def test_yaml_loader_rejects_duplicate_merge_nonmapping_and_nonstring_keys(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "retrieval-config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(RetrievalConfigError, match="parse|mapping|merge|duplicate|non-string"):
        load_retrieval_config(path, project_root=ROOT)


def test_loader_rejects_symlink_non_utf8_crlf_empty_and_oversized_files(tmp_path: Path) -> None:
    target = tmp_path / "target.yaml"
    target.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    link = tmp_path / "retrieval-config.yaml"
    link.symlink_to(target)
    with pytest.raises(RetrievalConfigError, match="regular file"):
        load_retrieval_config(link, project_root=ROOT)

    link.unlink()
    link.write_bytes(b"\xff")
    with pytest.raises(RetrievalConfigError, match="UTF-8"):
        load_retrieval_config(link, project_root=ROOT)

    link.write_bytes(CONFIG.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(RetrievalConfigError, match="LF newlines"):
        load_retrieval_config(link, project_root=ROOT)

    link.write_bytes(b"")
    with pytest.raises(RetrievalConfigError, match="file bound"):
        load_retrieval_config(link, project_root=ROOT)

    link.write_bytes(b"x" * (512 * 1024 + 1))
    with pytest.raises(RetrievalConfigError, match="file bound"):
        load_retrieval_config(link, project_root=ROOT)


@pytest.mark.parametrize(
    ("owner", "mutation"),
    [
        ("top", "unknown"),
        ("top", "missing"),
        ("qualification", "unknown"),
        ("selection", "missing"),
        ("freeze", "unknown"),
        ("revision", "missing"),
        ("candidate", "unknown"),
        ("configuration", "missing"),
        ("embedding", "unknown"),
        ("chunking", "missing"),
        ("index", "unknown"),
        ("retrieval", "missing"),
        ("measurement", "unknown"),
    ],
)
def test_unknown_and_missing_fields_fail_closed(tmp_path: Path, owner: str, mutation: str) -> None:
    document = _document()
    candidate = _candidate(document)
    configuration = _configuration(candidate)
    measurement = _measurements(candidate)[0]
    owners: dict[str, dict[str, object]] = {
        "top": document,
        "qualification": cast(dict[str, object], document["qualification"]),
        "selection": cast(dict[str, object], document["selection"]),
        "freeze": cast(dict[str, object], document["freeze"]),
        "revision": cast(dict[str, object], document["revision"]),
        "candidate": candidate,
        "configuration": cast(dict[str, object], candidate["configuration"]),
        "embedding": configuration["embedding"],
        "chunking": configuration["chunking"],
        "index": configuration["index"],
        "retrieval": configuration["retrieval"],
        "measurement": measurement,
    }
    target = owners[owner]
    if mutation == "unknown":
        target["unexpected"] = True
    else:
        target.pop(next(iter(target)))
    with pytest.raises(RetrievalConfigError, match="unknown or missing"):
        _load(tmp_path, document)


def test_candidate_and_fixture_coverage_rejects_count_duplicate_missing_and_unknown(
    tmp_path: Path,
) -> None:
    document = _document()
    _candidates(document).pop()
    with pytest.raises(RetrievalConfigError, match="candidate_limit"):
        _load(tmp_path, document)

    document = _document()
    _candidates(document)[1]["id"] = _candidates(document)[0]["id"]
    with pytest.raises(RetrievalConfigError, match="candidate IDs must be unique"):
        _load(tmp_path, document)

    document = _document()
    selected = _candidate(document)
    _measurements(selected).append(copy.deepcopy(_measurements(selected)[0]))
    with pytest.raises(RetrievalConfigError, match="repeats fixture"):
        _load(tmp_path, document)

    document = _document()
    _measurements(_candidate(document)).pop()
    with pytest.raises(RetrievalConfigError, match="every approved fixture"):
        _load(tmp_path, document)

    document = _document()
    _measurements(_candidate(document))[0]["fixture_id"] = "retrieval-unknown"
    with pytest.raises(RetrievalConfigError, match="unknown fixture"):
        _load(tmp_path, document)


def test_candidate_configurations_must_be_unique(tmp_path: Path) -> None:
    document = _document()
    _candidates(document)[1]["configuration"] = copy.deepcopy(
        _candidates(document)[0]["configuration"]
    )
    with pytest.raises(RetrievalConfigError, match="configurations must be unique"):
        _load(tmp_path, document)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("route_selected", 1, "boolean"),
        ("generation_filter_present", "true", "boolean"),
        ("cross_generation_leaks", True, "integer"),
        ("excluded_path_leaks", -1, "integer"),
        ("measured_latency_seconds", True, "finite number"),
        ("measured_latency_seconds", float("nan"), "finite number"),
        ("measured_latency_seconds", float("inf"), "finite number"),
        ("measured_latency_seconds", -0.01, "finite number"),
        ("measured_latency_seconds", 60.01, "finite number"),
        ("ranked_evidence_ids", "not-a-list", "list"),
        ("ranked_evidence_ids", ["duplicate", "duplicate"], "unique evidence IDs"),
        ("ranked_evidence_ids", [f"evidence-{index}" for index in range(11)], "at most ten"),
    ],
)
def test_measurement_runtime_types_finiteness_and_bounds_are_strict(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    document = _document()
    _measurements(_candidate(document))[0][field] = value
    with pytest.raises(RetrievalConfigError, match=error):
        _load(tmp_path, document)


def test_exact_lookup_and_generation_availability_metric_shapes_are_strict(
    tmp_path: Path,
) -> None:
    document = _document()
    _measurement(_candidate(document), "retrieval-core-versioned-command")[
        "exact_identifier_found"
    ] = True
    with pytest.raises(RetrievalConfigError, match="inapplicable exact lookup"):
        _load(tmp_path, document)

    document = _document()
    _measurement(_candidate(document), "retrieval-exact-release-digest")[
        "exact_identifier_found"
    ] = None
    with pytest.raises(RetrievalConfigError, match="boolean"):
        _load(tmp_path, document)

    document = _document()
    _measurement(_candidate(document), "retrieval-core-versioned-command")["failed_closed"] = False
    with pytest.raises(RetrievalConfigError, match="inapplicable fail-closed"):
        _load(tmp_path, document)

    document = _document()
    _measurement(_candidate(document), "retrieval-missing-generation")["ranked_evidence_ids"] = [
        "synthetic-unavailable-evidence"
    ]
    with pytest.raises(RetrievalConfigError, match="unavailable-generation evidence"):
        _load(tmp_path, document)


@pytest.mark.parametrize(
    ("fixture_id", "field", "value"),
    [
        ("retrieval-core-versioned-command", "route_selected", False),
        ("retrieval-core-versioned-command", "generation_filter_present", False),
        ("retrieval-core-versioned-command", "cross_generation_leaks", 1),
        ("retrieval-core-versioned-command", "excluded_path_leaks", 1),
        ("retrieval-core-versioned-command", "prompt_or_evaluation_leaks", 1),
        ("retrieval-core-versioned-command", "malformed_or_unverifiable_metadata", 1),
        ("retrieval-exact-release-digest", "exact_identifier_found", False),
        ("retrieval-missing-generation", "failed_closed", False),
    ],
)
def test_every_approved_hard_gate_blocks_the_selected_candidate(
    tmp_path: Path, fixture_id: str, field: str, value: object
) -> None:
    document = _document()
    _measurement(_candidate(document), fixture_id)[field] = value
    with pytest.raises(RetrievalConfigError, match="exactly one qualifying"):
        _load(tmp_path, document)


def test_overall_family_ndcg_and_p95_quality_gates_are_derived_from_observations(
    tmp_path: Path,
) -> None:
    document = _document()
    _measurement(_candidate(document), "retrieval-module-public-command")["ranked_evidence_ids"] = [
        "synthetic-module-json-set"
    ]
    with pytest.raises(RetrievalConfigError, match="exactly one qualifying"):
        _load(tmp_path, document)

    document = _document()
    for measurement in _measurements(_candidate(document)):
        ranked = cast(list[str], measurement["ranked_evidence_ids"])
        if ranked:
            measurement["ranked_evidence_ids"] = [
                "synthetic-low-rank-a",
                "synthetic-low-rank-b",
                *ranked,
            ]
    with pytest.raises(RetrievalConfigError, match="exactly one qualifying"):
        _load(tmp_path, document)

    document = _document()
    _measurement(_candidate(document), "retrieval-automation-interface")[
        "measured_latency_seconds"
    ] = 2.01
    with pytest.raises(RetrievalConfigError, match="exactly one qualifying"):
        _load(tmp_path, document)


def test_latency_threshold_is_inclusive_at_exact_boundary() -> None:
    document = _document()
    for measurement in _measurements(_candidate(document)):
        measurement["measured_latency_seconds"] = 2.0

    revision = cast(dict[str, object], document["revision"])
    revision["config_revision"] = compute_config_revision(document)
    suite = load_evaluation_suite(ROOT)
    with pytest.raises(RetrievalConfigError, match="separately approved full-index migration"):
        validate_retrieval_config(document, suite)


def test_exactly_one_candidate_must_qualify_and_selection_must_match(tmp_path: Path) -> None:
    document = _document()
    compact = _candidate(document, "titan-v2-512-fixed-256-15")
    _measurement(compact, "retrieval-module-public-command")["ranked_evidence_ids"] = [
        "synthetic-module-json-set",
        "synthetic-module-public-doc",
    ]
    with pytest.raises(RetrievalConfigError, match="exactly one qualifying"):
        _load(tmp_path, document)

    document = _document()
    selection = cast(dict[str, object], document["selection"])
    selection["selected_candidate"] = "titan-v2-1024-fixed-512-20"
    with pytest.raises(RetrievalConfigError, match="does not match"):
        _load(tmp_path, document)


def test_revision_must_match_and_excludes_only_its_own_value(tmp_path: Path) -> None:
    document = _document()
    assert compute_config_revision(document) == REVISION
    revision = cast(dict[str, object], document["revision"])
    revision["config_revision"] = "sha256:" + "0" * 64
    assert compute_config_revision(document) == REVISION
    with pytest.raises(RetrievalConfigError, match="does not match canonical semantics"):
        _load(tmp_path, document)

    document = _document()
    cast(dict[str, object], document["qualification"])["candidate_limit"] = 4
    assert compute_config_revision(document) != REVISION


def test_routine_semantic_change_with_recomputed_revision_requires_migration(
    tmp_path: Path,
) -> None:
    document = _document()
    _configuration(_candidate(document))["retrieval"]["number_of_results"] = 11
    path = _write(tmp_path, document, rehash=True)
    with pytest.raises(RetrievalConfigError, match="separately approved full-index migration"):
        load_retrieval_config(path, project_root=ROOT)


def test_selection_freeze_and_revision_semantics_cannot_be_relaxed(tmp_path: Path) -> None:
    changes = [
        ("selection", "tie_break", "candidate_order"),
        ("freeze", "routine_change", "allowed"),
        ("revision", "config_revision_is_not_part_of_preimage", False),
        ("qualification", "claim_scope", "production_measurements"),
    ]
    for owner, field, value in changes:
        document = _document()
        cast(dict[str, object], document[owner])[field] = value
        with pytest.raises(RetrievalConfigError):
            _load(tmp_path, document)


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("embedding", "dimensions", True, "integer"),
        ("embedding", "dimensions", 768, "unsupported"),
        ("embedding", "output_normalization", "caller_defined", "model-defined invariant"),
        ("chunking", "max_tokens", 19, "integer"),
        ("chunking", "overlap_percentage", 100, "integer"),
        ("index", "dimensions", 512, "incompatible"),
        ("retrieval", "number_of_results", 0, "integer"),
        ("retrieval", "reranking", 0, "boolean"),
        ("retrieval", "reranking", True, "reviewed boundary"),
    ],
)
def test_configuration_runtime_types_bounds_and_cross_field_compatibility(
    tmp_path: Path, section: str, field: str, value: object, error: str
) -> None:
    document = _document()
    _configuration(_candidate(document))[section][field] = value
    with pytest.raises(RetrievalConfigError, match=error):
        _load(tmp_path, document)


def test_candidate_limit_rejects_boolean_and_out_of_bounds(tmp_path: Path) -> None:
    for value in (True, 1, 9):
        document = _document()
        cast(dict[str, object], document["qualification"])["candidate_limit"] = value
        with pytest.raises(RetrievalConfigError, match="integer between 2 and 8"):
            _load(tmp_path, document)


def test_changed_query_or_relevance_judgment_cannot_reuse_frozen_revision(
    tmp_path: Path,
) -> None:
    shutil.copytree(ROOT / "evals", tmp_path / "evals")
    path = tmp_path / "retrieval-config.yaml"
    path.write_bytes(CONFIG.read_bytes())
    fixtures = tmp_path / "evals" / "retrieval.yaml"
    fixtures.write_text(
        fixtures.read_text(encoding="utf-8").replace(
            "which SET behavior applies", "which GET behavior applies", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(RetrievalConfigError, match="exact reviewed synthetic fixture"):
        load_retrieval_config(path, project_root=tmp_path)


def test_tampered_retrieval_criteria_are_not_accepted(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "evals", tmp_path / "evals")
    path = tmp_path / "retrieval-config.yaml"
    path.write_bytes(CONFIG.read_bytes())
    criteria = tmp_path / "evals" / "criteria" / "retrieval.yaml"
    criteria.write_text(
        criteria.read_text(encoding="utf-8").replace("    minimum: 0.95", "    minimum: 0.94", 1),
        encoding="utf-8",
    )
    with pytest.raises(RetrievalConfigError, match="approved retrieval inputs"):
        load_retrieval_config(path, project_root=tmp_path)


def test_validation_is_offline_and_has_no_aws_network_or_model_runtime_imports() -> None:
    source = (ROOT / "src" / "valkeyrie" / "retrieval_config.py").read_text(encoding="utf-8")
    for prohibited in (
        "boto3",
        "botocore",
        "requests",
        "urllib.request",
        "aws_cdk",
        "sentence_transformers",
        "transformers",
    ):
        assert prohibited not in source
