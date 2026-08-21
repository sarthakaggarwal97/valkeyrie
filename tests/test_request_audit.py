from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from tests.helpers import load_yaml
from tests.test_evaluations import _passing_model_runs, _passing_retrieval_results
from tests.test_evidence import _bundle, _package
from valkeyrie.answer_models import (
    AnswerModelCandidate,
    AnswerModelProfile,
    AnswerModelSelection,
    create_candidate_profiles,
    load_answer_model_inventory,
    select_answer_model,
)
from valkeyrie.evaluations import EvaluationSuite, evaluate_candidate, load_evaluation_suite
from valkeyrie.evidence import EvidencePackage
from valkeyrie.generation import GenerationBundle
from valkeyrie.prompts import PromptPackage, load_prompt_package
from valkeyrie.request_audit import (
    LiveObservation,
    RequestAuditError,
    create_live_observation,
    live_observation_value,
)
from valkeyrie.retrieval_config import load_retrieval_config

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = cast(dict[str, Any], load_yaml(ROOT / "src/valkeyrie/schemas/contracts.schema.json"))
REQUEST = "req_a-05-example"
QUESTION = "What does the GET command return in Valkey 9.0.0?"
APPLICATION = f"sha256:{'ab' * 32}"
CURRENT_APPLICATION = f"sha256:{'cd' * 32}"
STARTED = "2026-08-19T04:00:00Z"
EARLY = "2026-08-19T04:04:59Z"
RETRIED = "2026-08-19T04:05:00Z"
COMPLETED = "2026-08-19T04:10:00Z"
OBSERVED = "2026-08-19T03:59:00Z"
OBSERVATION_URL = "https://api.github.com/repos/valkey-io/valkey/releases/latest"
PAYLOAD: dict[str, object] = {
    "tag_name": "9.0.0",
    "assets": [{"name": "valkey-9.0.0.tar.gz", "size": 1234}],
}
EVALUATION_STARTED = "2026-08-19T03:00:00Z"
EVALUATION_COMPLETED = "2026-08-19T03:10:00Z"


@pytest.fixture(scope="module")
def bundle() -> GenerationBundle:
    return _bundle(load_retrieval_config(ROOT / "retrieval-config.yaml"))


@pytest.fixture(scope="module")
def package(bundle: GenerationBundle) -> EvidencePackage:
    return _package(bundle)


@pytest.fixture(scope="module")
def suite() -> EvaluationSuite:
    return load_evaluation_suite(ROOT)


@pytest.fixture(scope="module")
def prompt_package() -> PromptPackage:
    return load_prompt_package(ROOT)


@pytest.fixture(scope="module")
def candidates() -> tuple[AnswerModelCandidate, ...]:
    return load_answer_model_inventory(ROOT / "answer-models.yaml")


@pytest.fixture(scope="module")
def profiles(
    suite: EvaluationSuite,
    prompt_package: PromptPackage,
    bundle: GenerationBundle,
    candidates: tuple[AnswerModelCandidate, ...],
) -> tuple[AnswerModelProfile, ...]:
    return create_candidate_profiles(
        candidates,
        prompt_revision=prompt_package.prompt_revision,
        corpus_generation=bundle.generation_id,
        evaluation_suite_revision=suite.revision,
    )


@pytest.fixture(scope="module")
def reports(
    suite: EvaluationSuite,
    profiles: tuple[AnswerModelProfile, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(_report(suite, profile) for profile in profiles)


@pytest.fixture(scope="module")
def selection(
    suite: EvaluationSuite,
    profiles: tuple[AnswerModelProfile, ...],
    reports: tuple[dict[str, object], ...],
) -> AnswerModelSelection:
    return select_answer_model(suite, profiles, reports)


@pytest.fixture(scope="module")
def current_bundle() -> GenerationBundle:
    return _bundle(load_retrieval_config(ROOT / "retrieval-config.yaml"), commit="4" * 40)


@pytest.fixture(scope="module")
def current_package(current_bundle: GenerationBundle) -> EvidencePackage:
    return _package(current_bundle)


def _report(suite: EvaluationSuite, profile: AnswerModelProfile) -> dict[str, object]:
    return evaluate_candidate(
        suite,
        candidate_revision=profile.profile_revision,
        started_at=EVALUATION_STARTED,
        completed_at=EVALUATION_COMPLETED,
        model_runs=_passing_model_runs(suite),
        retrieval_results=_passing_retrieval_results(suite),
    )


def _observation(payload: object = PAYLOAD) -> LiveObservation:
    return create_live_observation(
        observed_at=OBSERVED,
        source_url=OBSERVATION_URL,
        object_type="release",
        payload=payload,
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


def test_live_observation_retains_exact_canonical_payload_and_fresh_exports() -> None:
    observation = _observation({"z": [1, {"ok": True}], "a": "value"})
    same = _observation({"a": "value", "z": [1, {"ok": True}]})
    assert observation == same
    assert observation.canonical_payload == b'{"a":"value","z":[1,{"ok":true}]}'
    assert (
        observation.payload_digest
        == "sha256:" + hashlib.sha256(observation.canonical_payload).hexdigest()
    )
    assert observation.observation_id.startswith("obs_")
    assert len(observation.observation_id) == len("obs_") + 64

    first = live_observation_value(observation)
    _validator("live_observation").validate(first)
    payload = cast(dict[str, object], first["payload"])
    cast(list[object], payload["z"]).append("mutated export")
    second = live_observation_value(observation)
    assert second["payload"] == {"a": "value", "z": [1, {"ok": True}]}
    assert observation.canonical_payload == b'{"a":"value","z":[1,{"ok":true}]}'


def test_live_observation_payload_has_exact_positive_256_kib_bound() -> None:
    exact = _observation("x" * (256 * 1024 - 2))
    assert len(exact.canonical_payload) == 256 * 1024
    with pytest.raises(RequestAuditError, match="262144-byte bound"):
        _observation("x" * (256 * 1024 - 1))


@pytest.mark.parametrize(
    "payload",
    [
        ("tuple",),
        {"bytes": b"no"},
        {1: "non-text key"},
        {"set": {1}},
        {"number": float("nan")},
        {"number": float("inf")},
        {"text": "\ud800"},
    ],
)
def test_non_json_non_utf8_and_non_finite_payloads_fail_closed(payload: object) -> None:
    with pytest.raises(RequestAuditError, match="live observation"):
        _observation(payload)


def test_cyclic_payload_fails_closed() -> None:
    payload: list[object] = []
    payload.append(payload)
    with pytest.raises(RequestAuditError, match="cyclic JSON"):
        _observation(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_url", "http://api.github.com/repos/valkey-io/valkey"),
        ("source_url", "https://"),
        ("source_url", "https://api.github.com/a b"),
        ("source_url", "https://example.com/repos/valkey-io/valkey/releases/latest"),
        ("source_url", "https://api.github.com/repos/other/valkey/releases/latest"),
        ("source_url", "https://user@api.github.com/repos/valkey-io/valkey/releases/latest"),
        ("source_url", "https://api.github.com/repos/valkey-io/../other/releases/latest"),
        ("source_url", "https://github.com/other/valkey/issues/1"),
        ("observed_at", "2026-08-19 04:00:00Z"),
        ("observed_at", "2026-02-30T04:00:00Z"),
        ("object_type", "gist"),
    ],
)
def test_malformed_observation_metadata_fails_closed(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "observed_at": OBSERVED,
        "source_url": OBSERVATION_URL,
        "object_type": "release",
        "payload": PAYLOAD,
    }
    kwargs[field] = value
    with pytest.raises(RequestAuditError):
        create_live_observation(**cast(dict[str, Any], kwargs))
