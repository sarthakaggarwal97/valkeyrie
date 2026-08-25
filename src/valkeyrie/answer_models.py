"""Deterministic answer-model candidate inventory, identity, and selection."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import yaml

from valkeyrie.evaluations import EvaluationSuite, verify_evaluation_report
from valkeyrie.sources import _load_yaml_mapping

ReasoningEffort: TypeAlias = Literal["low", "medium", "high"]


class AnswerModelError(ValueError):
    """An answer-model profile or qualification set is invalid."""


@dataclass(frozen=True)
class InferenceConfiguration:
    """One exact bounded Bedrock inference configuration."""

    maximum_output_tokens: int
    temperature: float | None
    top_p: float | None
    reasoning_effort: ReasoningEffort | None


@dataclass(frozen=True)
class AnswerModelProfile:
    """One immutable answer-model qualification candidate."""

    model_revision: str
    inference: InferenceConfiguration
    inference_config_revision: str
    prompt_revision: str
    corpus_generation: str
    evaluation_suite_revision: str
    profile_revision: str


@dataclass(frozen=True)
class AnswerModelCandidate:
    """One approved candidate model identity and its exact inference settings."""

    model_revision: str
    inference: InferenceConfiguration


@dataclass(frozen=True)
class AnswerModelSelection:
    """Highest-quality passing candidate and its complete report evidence."""

    profile: AnswerModelProfile
    evaluation_report_id: str
    evaluation_suite_revision: str
    recorded_cost_usd: float


_MODEL = re.compile(r"^[a-z0-9][a-z0-9._:-]{2,255}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_QUALITY_GATE = "model.answerable_cases_materially_correct"
_FAMILY_PREFIX = "model.family_correctness."
_RELIABILITY_GATE = "model.candidate_request_success"
_LATENCY_GATE = "model.candidate_p95_latency_seconds"
_MAX_INVENTORY_BYTES = 64 * 1024
_MAX_INVENTORY_CANDIDATES = 8
_INVENTORY_IDENTITY = ("valkeyrie.io/answer-models/1", "AnswerModelInventory")
_INVENTORY_FIELDS = {"api_version", "kind", "candidates"}
_CANDIDATE_FIELDS = {"model_id", "inference"}
_INFERENCE_FIELDS = {
    "maximum_output_tokens",
    "temperature",
    "top_p",
    "reasoning_effort",
}
_FABLE_MODEL = "us.anthropic.claude-fable-5"
_OPUS_MODEL = "us.anthropic.claude-opus-5"
# Both Claude revisions take reasoning_effort and reject sampling controls. Immutable
# Lambda version 8 ran Opus with reasoning_effort "low" and null temperature/top_p, so
# the constraint is a property of the Claude family rather than of Fable alone.
_REASONING_EFFORT_MODELS = frozenset({_FABLE_MODEL, _OPUS_MODEL})
_REASONING_EFFORTS = frozenset({"low", "medium", "high"})


def load_answer_model_inventory(path: Path) -> tuple[AnswerModelCandidate, ...]:
    """Load the approved answer-model candidate inventory or fail closed."""
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise AnswerModelError("answer-model inventory must be a regular file")
        if metadata.st_size < 1 or metadata.st_size > _MAX_INVENTORY_BYTES:
            raise AnswerModelError("answer-model inventory exceeds its local file bound")
        content = path.read_bytes()
    except AnswerModelError:
        raise
    except OSError as error:
        raise AnswerModelError(f"cannot read answer-model inventory: {error}") from error
    if len(content) < 1 or len(content) > _MAX_INVENTORY_BYTES:
        raise AnswerModelError("answer-model inventory exceeds its local file bound")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AnswerModelError("answer-model inventory is not UTF-8") from error
    if "\r" in text:
        raise AnswerModelError("answer-model inventory must use LF newlines")
    try:
        document = _load_yaml_mapping(text, str(path), reject_merge_keys=True)
    except (UnicodeError, yaml.YAMLError, ValueError) as error:
        raise AnswerModelError(f"cannot parse answer-model inventory: {error}") from error
    return validate_answer_model_inventory(document)


def validate_answer_model_inventory(
    document: Mapping[str, object],
) -> tuple[AnswerModelCandidate, ...]:
    """Validate one already-loaded inventory document into immutable candidates."""
    if not isinstance(document, Mapping) or set(document) != _INVENTORY_FIELDS:
        raise AnswerModelError("answer-model inventory has an unknown or missing field")
    if (document.get("api_version"), document.get("kind")) != _INVENTORY_IDENTITY:
        raise AnswerModelError("answer-model inventory has an incompatible identity")
    raw = document.get("candidates")
    if not isinstance(raw, list) or not 1 <= len(raw) <= _MAX_INVENTORY_CANDIDATES:
        raise AnswerModelError(
            f"answer-model inventory must list 1 through {_MAX_INVENTORY_CANDIDATES} candidates"
        )
    candidates: list[AnswerModelCandidate] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, Mapping) or set(value) != _CANDIDATE_FIELDS:
            raise AnswerModelError("inventory candidate has an unknown or missing field")
        model = _validated_model_revision(value.get("model_id"))
        inference_value = value.get("inference")
        if not isinstance(inference_value, Mapping) or set(inference_value) != _INFERENCE_FIELDS:
            raise AnswerModelError("inventory candidate inference has an unknown or missing field")
        inference = _validated_inference(
            model,
            inference_value.get("maximum_output_tokens"),
            inference_value.get("temperature"),
            inference_value.get("top_p"),
            inference_value.get("reasoning_effort"),
        )
        if model in seen:
            raise AnswerModelError("inventory candidate model IDs must be unique")
        seen.add(model)
        candidates.append(AnswerModelCandidate(model, inference))
    return tuple(candidates)


def create_candidate_profiles(
    candidates: tuple[AnswerModelCandidate, ...],
    *,
    prompt_revision: str,
    corpus_generation: str,
    evaluation_suite_revision: str,
) -> tuple[AnswerModelProfile, ...]:
    """Construct the exact qualification candidates from the approved inventory."""
    if not isinstance(candidates, tuple) or not candidates:
        raise AnswerModelError("candidates must be a non-empty immutable tuple")
    return tuple(
        create_answer_model_profile(
            candidate.model_revision,
            maximum_output_tokens=candidate.inference.maximum_output_tokens,
            temperature=candidate.inference.temperature,
            top_p=candidate.inference.top_p,
            reasoning_effort=candidate.inference.reasoning_effort,
            prompt_revision=prompt_revision,
            corpus_generation=corpus_generation,
            evaluation_suite_revision=evaluation_suite_revision,
        )
        for candidate in candidates
    )


def create_answer_model_profile(
    model_revision: str,
    *,
    maximum_output_tokens: int,
    temperature: float | None,
    top_p: float | None,
    reasoning_effort: ReasoningEffort | None,
    prompt_revision: str,
    corpus_generation: str,
    evaluation_suite_revision: str,
) -> AnswerModelProfile:
    """Create a content-addressed candidate bound to every fixed evaluation input."""
    model = _validated_model_revision(model_revision)
    for value, field in (
        (prompt_revision, "prompt revision"),
        (corpus_generation, "corpus generation"),
        (evaluation_suite_revision, "evaluation suite revision"),
    ):
        if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
            raise AnswerModelError(f"{field} must be a sha256 digest")
    inference = _validated_inference(
        model, maximum_output_tokens, temperature, top_p, reasoning_effort
    )
    inference_value = {
        "maximum_output_tokens": inference.maximum_output_tokens,
        "temperature": inference.temperature,
        "top_p": inference.top_p,
        "reasoning_effort": inference.reasoning_effort,
    }
    inference_revision = _revision("valkeyrie-inference-config/1", inference_value)
    profile_revision = _revision(
        "valkeyrie-answer-model-profile/1",
        {
            "model_revision": model,
            "inference_config_revision": inference_revision,
            "prompt_revision": prompt_revision,
            "corpus_generation": corpus_generation,
            "evaluation_suite_revision": evaluation_suite_revision,
        },
    )
    return AnswerModelProfile(
        model,
        inference,
        inference_revision,
        prompt_revision,
        corpus_generation,
        evaluation_suite_revision,
        profile_revision,
    )


def _validated_model_revision(value: object) -> str:
    if not isinstance(value, str) or _MODEL.fullmatch(value) is None:
        raise AnswerModelError("model revision is malformed")
    return value


def _validated_inference(
    model_revision: str,
    maximum_output_tokens: object,
    temperature: object,
    top_p: object,
    reasoning_effort: object,
) -> InferenceConfiguration:
    if (
        not isinstance(maximum_output_tokens, int)
        or isinstance(maximum_output_tokens, bool)
        or not 1 <= maximum_output_tokens <= 4096
    ):
        raise AnswerModelError("maximum output tokens must be an integer from 1 through 4096")
    temperature_value = _optional_finite(temperature, "temperature")
    top_p_value = _optional_finite(top_p, "top_p")
    if temperature_value is not None and not 0.0 <= temperature_value <= 1.0:
        raise AnswerModelError("temperature must be null or from 0 through 1")
    if top_p_value is not None and not 0.0 < top_p_value <= 1.0:
        raise AnswerModelError("top_p must be null or greater than 0 and at most 1")
    if reasoning_effort is None:
        reasoning_effort_value = None
    elif not isinstance(reasoning_effort, str) or reasoning_effort not in _REASONING_EFFORTS:
        raise AnswerModelError("reasoning_effort must be null, low, medium, or high")
    else:
        reasoning_effort_value = cast(ReasoningEffort, reasoning_effort)
    if model_revision in _REASONING_EFFORT_MODELS and (
        temperature_value is not None or top_p_value is not None
    ):
        raise AnswerModelError("Claude sampling temperature and top_p must be null")
    if model_revision not in _REASONING_EFFORT_MODELS and reasoning_effort_value is not None:
        raise AnswerModelError("reasoning_effort is supported only for Claude revisions")
    return InferenceConfiguration(
        maximum_output_tokens, temperature_value, top_p_value, reasoning_effort_value
    )


def _optional_finite(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _finite(value, field)


def select_answer_model(
    suite: EvaluationSuite,
    profiles: Sequence[AnswerModelProfile],
    reports: Sequence[Mapping[str, object]],
) -> AnswerModelSelection:
    """Select quality first from exact complete passing candidate reports."""
    if not isinstance(profiles, tuple) or not profiles:
        raise AnswerModelError("profiles must be a non-empty immutable tuple")
    if not isinstance(reports, tuple) or len(reports) != len(profiles):
        raise AnswerModelError("reports must contain exactly one item per profile")
    by_revision: dict[str, AnswerModelProfile] = {}
    for profile in profiles:
        _validate_profile(profile)
        if profile.evaluation_suite_revision != suite.revision:
            raise AnswerModelError("profile evaluation suite revision does not match the suite")
        if profile.profile_revision in by_revision:
            raise AnswerModelError("answer-model profile revisions must be unique")
        by_revision[profile.profile_revision] = profile

    candidates: list[tuple[tuple[float, float, float, float, str], AnswerModelSelection]] = []
    seen: set[str] = set()
    for report in reports:
        try:
            verify_evaluation_report(report, suite)
        except ValueError as error:
            raise AnswerModelError(f"candidate evaluation report is invalid: {error}") from error
        revision = cast(str, report["candidate_revision"])
        profile = by_revision.get(revision)
        if profile is None:
            raise AnswerModelError("evaluation report identifies an unknown profile revision")
        if revision in seen:
            raise AnswerModelError("multiple evaluation reports identify the same profile")
        seen.add(revision)
        if report["result"] != "pass":
            continue
        quality = _gate_actual(report, _QUALITY_GATE)
        family_quality = min(
            _gate_actual(report, f"{_FAMILY_PREFIX}{family}") for family in suite.required_families
        )
        reliability = _gate_actual(report, _RELIABILITY_GATE)
        latency = _gate_actual(report, _LATENCY_GATE)
        summary = cast(Mapping[str, object], report["summary"])
        cost = _finite(summary["recorded_cost_usd"], "recorded cost")
        selection = AnswerModelSelection(
            profile,
            cast(str, report["report_id"]),
            suite.revision,
            cost,
        )
        candidates.append(
            ((quality, family_quality, reliability, -latency, profile.profile_revision), selection)
        )

    if seen != set(by_revision):
        raise AnswerModelError("every profile must have one exact evaluation report")
    if not candidates:
        raise AnswerModelError("no answer-model profile passed every approved hard gate")
    return max(candidates, key=lambda candidate: candidate[0])[1]


def _validate_profile(profile: AnswerModelProfile) -> None:
    if not isinstance(profile, AnswerModelProfile):
        raise AnswerModelError("profile has an invalid type")
    expected = create_answer_model_profile(
        profile.model_revision,
        maximum_output_tokens=profile.inference.maximum_output_tokens,
        temperature=profile.inference.temperature,
        top_p=profile.inference.top_p,
        reasoning_effort=profile.inference.reasoning_effort,
        prompt_revision=profile.prompt_revision,
        corpus_generation=profile.corpus_generation,
        evaluation_suite_revision=profile.evaluation_suite_revision,
    )
    if profile != expected:
        raise AnswerModelError("answer-model profile revisions do not match its content")


def _gate_actual(report: Mapping[str, object], name: str) -> float:
    gates = cast(Sequence[Mapping[str, object]], report["gates"])
    matches = [gate for gate in gates if gate["name"] == name]
    if len(matches) != 1:
        raise AnswerModelError(f"candidate report is missing exact quality gate {name}")
    return _finite(matches[0]["actual"], f"{name} actual")


def _revision(domain: str, value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    for part in (domain.encode("utf-8"), encoded):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return f"sha256:{digest.hexdigest()}"


def _finite(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AnswerModelError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AnswerModelError(f"{field} must be a finite number")
    return result
