from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pytest

from tests.test_evaluations import _passing_model_runs, _passing_retrieval_results
from valkeyrie.answer_models import (
    AnswerModelError,
    AnswerModelProfile,
    InferenceConfiguration,
    create_answer_model_profile,
    create_candidate_profiles,
    load_answer_model_inventory,
    select_answer_model,
    validate_answer_model_inventory,
)
from valkeyrie.evaluations import (
    EvaluationSuite,
    ModelRun,
    evaluate_candidate,
    load_evaluation_suite,
)

ROOT = Path(__file__).resolve().parents[1]
PROMPT_REVISION = "sha256:398f37c086acc551a0e95be320099dfe9795625a6cb5282da272b82dc9bdc108"
CORPUS_GENERATION = f"sha256:{'c' * 64}"
STARTED = "2026-08-19T03:00:00Z"
COMPLETED = "2026-08-19T03:10:00Z"
INVENTORY_DOCUMENT: dict[str, object] = {
    "api_version": "valkeyrie.io/answer-models/1",
    "kind": "AnswerModelInventory",
    "candidates": [
        {
            "model_id": "us.anthropic.claude-fable-5",
            "inference": {
                "maximum_output_tokens": 2048,
                "temperature": None,
                "top_p": None,
                "reasoning_effort": "low",
            },
        },
        {
            "model_id": "amazon.nova-pro-v1:0",
            "inference": {
                "maximum_output_tokens": 1200,
                "temperature": 0.0,
                "top_p": 1.0,
                "reasoning_effort": None,
            },
        },
        {
            "model_id": "us.anthropic.claude-opus-5",
            "inference": {
                "maximum_output_tokens": 2048,
                "temperature": None,
                "top_p": None,
                "reasoning_effort": "low",
            },
        },
    ],
}


@pytest.fixture(scope="module")
def suite() -> EvaluationSuite:
    return load_evaluation_suite(ROOT)


@pytest.fixture(scope="module")
def pro(suite: EvaluationSuite) -> AnswerModelProfile:
    return create_answer_model_profile(
        "amazon.nova-pro-v1:0",
        maximum_output_tokens=1200,
        temperature=0.0,
        top_p=1.0,
        reasoning_effort=None,
        prompt_revision=PROMPT_REVISION,
        corpus_generation=CORPUS_GENERATION,
        evaluation_suite_revision=suite.revision,
    )


@pytest.fixture(scope="module")
def lite(suite: EvaluationSuite) -> AnswerModelProfile:
    return create_answer_model_profile(
        "amazon.nova-lite-v1:0",
        maximum_output_tokens=1200,
        temperature=0.0,
        top_p=1.0,
        reasoning_effort=None,
        prompt_revision=PROMPT_REVISION,
        corpus_generation=CORPUS_GENERATION,
        evaluation_suite_revision=suite.revision,
    )


def _report(
    suite: EvaluationSuite,
    profile: AnswerModelProfile,
    *,
    runs: list[ModelRun] | None = None,
) -> dict[str, object]:
    return evaluate_candidate(
        suite,
        candidate_revision=profile.profile_revision,
        started_at=STARTED,
        completed_at=COMPLETED,
        model_runs=runs if runs is not None else _passing_model_runs(suite),
        retrieval_results=_passing_retrieval_results(suite),
    )


def test_profile_has_separate_exact_model_config_and_combined_revisions(
    pro: AnswerModelProfile,
) -> None:
    assert pro.model_revision == "amazon.nova-pro-v1:0"
    assert pro.inference.maximum_output_tokens == 1200
    assert pro.inference.temperature == 0.0
    assert pro.inference.top_p == 1.0
    assert pro.inference_config_revision.startswith("sha256:")
    assert pro.profile_revision.startswith("sha256:")
    assert pro.profile_revision != pro.inference_config_revision
    assert pro == create_answer_model_profile(
        pro.model_revision,
        maximum_output_tokens=1200,
        temperature=0,
        top_p=1,
        reasoning_effort=None,
        prompt_revision=pro.prompt_revision,
        corpus_generation=pro.corpus_generation,
        evaluation_suite_revision=pro.evaluation_suite_revision,
    )
    assert (
        create_answer_model_profile(
            pro.model_revision,
            maximum_output_tokens=1201,
            temperature=0,
            top_p=1,
            reasoning_effort=None,
            prompt_revision=pro.prompt_revision,
            corpus_generation=pro.corpus_generation,
            evaluation_suite_revision=pro.evaluation_suite_revision,
        ).inference_config_revision
        != pro.inference_config_revision
    )


def test_nullable_inference_is_serialized_exactly_and_changes_both_revisions(
    suite: EvaluationSuite,
    pro: AnswerModelProfile,
) -> None:
    fable = create_answer_model_profile(
        "us.anthropic.claude-fable-5",
        maximum_output_tokens=2048,
        temperature=None,
        top_p=None,
        reasoning_effort="low",
        prompt_revision=PROMPT_REVISION,
        corpus_generation=CORPUS_GENERATION,
        evaluation_suite_revision=suite.revision,
    )
    assert asdict(fable.inference) == {
        "maximum_output_tokens": 2048,
        "temperature": None,
        "top_p": None,
        "reasoning_effort": "low",
    }
    default_effort = create_answer_model_profile(
        fable.model_revision,
        maximum_output_tokens=2048,
        temperature=None,
        top_p=None,
        reasoning_effort=None,
        prompt_revision=PROMPT_REVISION,
        corpus_generation=CORPUS_GENERATION,
        evaluation_suite_revision=suite.revision,
    )
    assert fable.inference_config_revision != default_effort.inference_config_revision
    assert fable.profile_revision != default_effort.profile_revision
    assert fable.inference_config_revision != pro.inference_config_revision
    tampered = replace(fable, inference=replace(fable.inference, reasoning_effort=None))
    with pytest.raises(AnswerModelError, match="profile revisions"):
        select_answer_model(suite, (tampered,), (_report(suite, fable),))


def test_each_fixed_evaluation_input_is_bound_to_candidate_identity(
    suite: EvaluationSuite,
    pro: AnswerModelProfile,
) -> None:
    changes = (
        {"prompt_revision": f"sha256:{'1' * 64}"},
        {"corpus_generation": f"sha256:{'2' * 64}"},
        {"evaluation_suite_revision": f"sha256:{'3' * 64}"},
    )
    for change in changes:
        changed = create_answer_model_profile(
            pro.model_revision,
            maximum_output_tokens=pro.inference.maximum_output_tokens,
            temperature=pro.inference.temperature,
            top_p=pro.inference.top_p,
            reasoning_effort=pro.inference.reasoning_effort,
            prompt_revision=str(change.get("prompt_revision", pro.prompt_revision)),
            corpus_generation=str(change.get("corpus_generation", pro.corpus_generation)),
            evaluation_suite_revision=str(
                change.get("evaluation_suite_revision", pro.evaluation_suite_revision)
            ),
        )
        assert changed.profile_revision != pro.profile_revision
        with pytest.raises(AnswerModelError, match="unknown profile|suite revision"):
            select_answer_model(suite, (changed,), (_report(suite, pro),))


def test_highest_quality_passing_candidate_wins_even_when_more_expensive(
    suite: EvaluationSuite,
    pro: AnswerModelProfile,
    lite: AnswerModelProfile,
) -> None:
    lower_quality = _passing_model_runs(suite)
    supported_indexes = [
        index
        for index, run in enumerate(lower_quality)
        if run.metrics["materially_correct"] is True
    ]
    for index in supported_indexes[:3]:
        lower_quality[index] = replace(
            lower_quality[index],
            metrics={**lower_quality[index].metrics, "materially_correct": False},
            cost_usd=0.0001,
        )
    expensive = [replace(run, cost_usd=1.0) for run in _passing_model_runs(suite)]

    pro_report = _report(suite, pro, runs=expensive)
    lite_report = _report(suite, lite, runs=lower_quality)
    selected = select_answer_model(
        suite,
        (pro, lite),
        (pro_report, lite_report),
    )

    assert selected.profile == pro
    assert selected.evaluation_report_id == pro_report["report_id"]
    assert selected.evaluation_suite_revision == suite.revision
    assert selected.recorded_cost_usd > float(lite_report["summary"]["recorded_cost_usd"])  # type: ignore[index]


def test_failed_hard_gate_is_never_eligible(
    suite: EvaluationSuite,
    pro: AnswerModelProfile,
    lite: AnswerModelProfile,
) -> None:
    failed_runs = _passing_model_runs(suite)
    failed_runs[0] = replace(
        failed_runs[0],
        metrics={**failed_runs[0].metrics, "fabricated_citations_or_links": 1},
    )
    selected = select_answer_model(
        suite,
        (pro, lite),
        (_report(suite, pro, runs=failed_runs), _report(suite, lite)),
    )
    assert selected.profile == lite

    with pytest.raises(AnswerModelError, match="no answer-model profile"):
        select_answer_model(
            suite,
            (pro,),
            (_report(suite, pro, runs=failed_runs),),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_revision": "bad model"},
        {"maximum_output_tokens": 0},
        {"maximum_output_tokens": True},
        {"temperature": -0.1},
        {"temperature": float("nan")},
        {"temperature": "null"},
        {"temperature": False},
        {"top_p": 0},
        {"top_p": float("inf")},
        {"top_p": "null"},
        {"top_p": True},
        {"reasoning_effort": True},
        {"reasoning_effort": "unknown"},
        {"reasoning_effort": "low"},
    ],
)
def test_malformed_profiles_fail_closed(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "model_revision": "amazon.nova-pro-v1:0",
        "maximum_output_tokens": 1200,
        "temperature": 0.0,
        "top_p": 1.0,
        "reasoning_effort": None,
        "prompt_revision": PROMPT_REVISION,
        "corpus_generation": CORPUS_GENERATION,
        "evaluation_suite_revision": f"sha256:{'e' * 64}",
    }
    values.update(kwargs)
    with pytest.raises(AnswerModelError):
        create_answer_model_profile(**values)  # type: ignore[arg-type]


def test_tampered_profile_and_report_bindings_fail_closed(
    suite: EvaluationSuite,
    pro: AnswerModelProfile,
    lite: AnswerModelProfile,
) -> None:
    with pytest.raises(AnswerModelError, match="profile revisions"):
        select_answer_model(
            suite,
            (replace(pro, inference_config_revision=f"sha256:{'0' * 64}"),),
            (_report(suite, pro),),
        )
    with pytest.raises(AnswerModelError, match="exactly one"):
        select_answer_model(suite, (pro, lite), (_report(suite, pro),))
    with pytest.raises(AnswerModelError, match="unknown profile"):
        select_answer_model(suite, (pro,), (_report(suite, lite),))
    with pytest.raises(AnswerModelError, match="unique"):
        select_answer_model(
            suite,
            (pro, pro),
            (_report(suite, pro), _report(suite, pro)),
        )


def test_report_must_be_suite_bound_and_content_verified(
    suite: EvaluationSuite,
    pro: AnswerModelProfile,
) -> None:
    report = _report(suite, pro)
    report["report_id"] = f"eval_{'0' * 64}"
    with pytest.raises(AnswerModelError, match="candidate evaluation report is invalid"):
        select_answer_model(suite, (pro,), (report,))


def test_inputs_must_be_nonempty_immutable_exact_sets(
    suite: EvaluationSuite,
    pro: AnswerModelProfile,
) -> None:
    with pytest.raises(AnswerModelError, match="immutable tuple"):
        select_answer_model(suite, [pro], [_report(suite, pro)])
    with pytest.raises(AnswerModelError, match="immutable tuple"):
        select_answer_model(suite, (), ())


def test_inventory_binds_the_exact_approved_candidates() -> None:
    candidates = load_answer_model_inventory(ROOT / "answer-models.yaml")
    assert [candidate.model_revision for candidate in candidates] == [
        "us.anthropic.claude-fable-5",
        "amazon.nova-pro-v1:0",
        "us.anthropic.claude-opus-5",
    ]
    assert candidates[0].inference == InferenceConfiguration(2048, None, None, "low")
    assert candidates[1].inference == InferenceConfiguration(1200, 0.0, 1.0, None)
    assert candidates[2].inference == InferenceConfiguration(2048, None, None, "low")
    assert validate_answer_model_inventory(INVENTORY_DOCUMENT) == candidates


def test_candidate_profiles_are_constructed_only_from_the_inventory() -> None:
    candidates = load_answer_model_inventory(ROOT / "answer-models.yaml")
    suite_revision = f"sha256:{'e' * 64}"
    profiles = create_candidate_profiles(
        candidates,
        prompt_revision=PROMPT_REVISION,
        corpus_generation=CORPUS_GENERATION,
        evaluation_suite_revision=suite_revision,
    )
    assert profiles == tuple(
        create_answer_model_profile(
            candidate.model_revision,
            maximum_output_tokens=candidate.inference.maximum_output_tokens,
            temperature=candidate.inference.temperature,
            top_p=candidate.inference.top_p,
            reasoning_effort=candidate.inference.reasoning_effort,
            prompt_revision=PROMPT_REVISION,
            corpus_generation=CORPUS_GENERATION,
            evaluation_suite_revision=suite_revision,
        )
        for candidate in candidates
    )
    with pytest.raises(AnswerModelError, match="immutable tuple"):
        create_candidate_profiles(
            list(candidates),  # type: ignore[arg-type]
            prompt_revision=PROMPT_REVISION,
            corpus_generation=CORPUS_GENERATION,
            evaluation_suite_revision=suite_revision,
        )
    with pytest.raises(AnswerModelError, match="immutable tuple"):
        create_candidate_profiles(
            (),
            prompt_revision=PROMPT_REVISION,
            corpus_generation=CORPUS_GENERATION,
            evaluation_suite_revision=suite_revision,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document.pop("candidates"),
        lambda document: document.update({"unreviewed_extension": True}),
        lambda document: document.update({"api_version": "valkeyrie.io/incompatible/2"}),
        lambda document: document.update({"kind": "SomethingElse"}),
        lambda document: document.update({"candidates": []}),
        lambda document: document.update(
            {"candidates": cast("list[object]", document["candidates"]) * 5}
        ),
        lambda document: cast("list[dict[str, object]]", document["candidates"])[0].pop(
            "inference"
        ),
        lambda document: cast("list[dict[str, object]]", document["candidates"])[0].update(
            {"qualified": True}
        ),
        lambda document: cast("list[dict[str, object]]", document["candidates"])[0].update(
            {"model_id": "Bad Model"}
        ),
        lambda document: cast(
            "dict[str, object]",
            cast("list[dict[str, object]]", document["candidates"])[0]["inference"],
        ).update({"maximum_output_tokens": True}),
        lambda document: cast(
            "dict[str, object]",
            cast("list[dict[str, object]]", document["candidates"])[0]["inference"],
        ).update({"temperature": "null"}),
        lambda document: cast(
            "dict[str, object]",
            cast("list[dict[str, object]]", document["candidates"])[0]["inference"],
        ).update({"temperature": 2.0}),
        lambda document: cast(
            "dict[str, object]",
            cast("list[dict[str, object]]", document["candidates"])[0]["inference"],
        ).update({"top_p": 0}),
        lambda document: cast(
            "dict[str, object]",
            cast("list[dict[str, object]]", document["candidates"])[0]["inference"],
        ).pop("top_p"),
        lambda document: cast(
            "dict[str, object]",
            cast("list[dict[str, object]]", document["candidates"])[0]["inference"],
        ).pop("reasoning_effort"),
        lambda document: cast(
            "dict[str, object]",
            cast("list[dict[str, object]]", document["candidates"])[0]["inference"],
        ).update({"reasoning_effort": True}),
        lambda document: cast(
            "dict[str, object]",
            cast("list[dict[str, object]]", document["candidates"])[0]["inference"],
        ).update({"reasoning_effort": "unknown"}),
        lambda document: cast(
            "dict[str, object]",
            cast("list[dict[str, object]]", document["candidates"])[1]["inference"],
        ).update({"reasoning_effort": "low"}),
        lambda document: cast(
            "dict[str, object]",
            cast("list[dict[str, object]]", document["candidates"])[0]["inference"],
        ).update({"temperature": 0.0}),
        lambda document: cast("list[dict[str, object]]", document["candidates"])[1].update(
            {"model_id": "us.anthropic.claude-fable-5"}
        ),
    ],
)
def test_malformed_inventory_documents_fail_closed(
    mutate: Callable[[dict[str, object]], object],
) -> None:
    document = deepcopy(INVENTORY_DOCUMENT)
    mutate(document)
    with pytest.raises(AnswerModelError):
        validate_answer_model_inventory(document)


def test_inventory_file_loading_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(AnswerModelError, match="cannot read"):
        load_answer_model_inventory(tmp_path / "answer-models.yaml")

    crlf = tmp_path / "crlf.yaml"
    crlf.write_bytes(b"api_version: valkeyrie.io/answer-models/1\r\n")
    with pytest.raises(AnswerModelError, match="LF newlines"):
        load_answer_model_inventory(crlf)

    empty = tmp_path / "empty.yaml"
    empty.write_bytes(b"")
    with pytest.raises(AnswerModelError, match="file bound"):
        load_answer_model_inventory(empty)

    merged = tmp_path / "merged.yaml"
    merged.write_text(
        "api_version: valkeyrie.io/answer-models/1\n"
        "kind: AnswerModelInventory\n"
        "candidates:\n"
        "- &base\n"
        "  model_id: amazon.nova-pro-v1:0\n"
        "  inference:\n"
        "    maximum_output_tokens: 1200\n"
        "    temperature: 0.0\n"
        "    top_p: 1.0\n"
        "- <<: *base\n",
        encoding="utf-8",
    )
    with pytest.raises(AnswerModelError, match="cannot parse"):
        load_answer_model_inventory(merged)
