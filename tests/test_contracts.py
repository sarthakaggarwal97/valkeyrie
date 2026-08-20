from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

from tests.helpers import load_yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "src" / "valkeyrie" / "schemas" / "contracts.schema.json"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "contracts" / "valid.json"
SCHEMA_BUNDLE = cast(dict[str, Any], load_yaml(SCHEMA_PATH))
FIXTURES = cast(dict[str, dict[str, Any]], load_yaml(FIXTURE_PATH))
CONTRACTS = (
    "source_inventory",
    "normalized_document",
    "metadata_sidecar",
    "generation_manifest",
    "generation_completion",
    "candidate_state",
    "generation_availability",
    "active_generation",
    "protected_approval",
    "evidence_record",
    "model_output",
    "request_audit",
    "live_observation",
    "evaluation_report",
)
REQUIRED_FIELD = {
    "source_inventory": "repositories",
    "normalized_document": "content",
    "metadata_sidecar": "generation_id",
    "generation_manifest": "documents",
    "generation_completion": "manifest_digest",
    "candidate_state": "fence",
    "generation_availability": "retrievable",
    "active_generation": "revision",
    "protected_approval": "approver",
    "evidence_record": "source",
    "model_output": "claims",
    "request_audit": "prompt_revision",
    "live_observation": "observed_at",
    "evaluation_report": "summary",
}


def _instance(contract: str) -> dict[str, Any]:
    if contract == "source_inventory":
        return cast(dict[str, Any], load_yaml(ROOT / "sources.yaml"))
    return deepcopy(FIXTURES[contract])


def _validator(contract: str) -> Draft202012Validator:
    schema = {
        "$schema": SCHEMA_BUNDLE["$schema"],
        "$defs": SCHEMA_BUNDLE["$defs"],
        "$ref": f"#/$defs/{contract}",
    }
    return Draft202012Validator(schema, format_checker=FormatChecker())


def test_schema_bundle_contains_exact_shared_contracts() -> None:
    assert set(SCHEMA_BUNDLE) == {"$schema", "$id", "title", "$defs"}
    definitions = cast(dict[str, object], SCHEMA_BUNDLE["$defs"])
    assert set(definitions) == {
        "digest",
        "commit",
        "timestamp",
        "https_url",
        "repository_path",
        "non_blank",
        "path_rule",
        "path_policy",
        "hard_exclusion_group",
        "repository",
        *CONTRACTS,
    }
    Draft202012Validator.check_schema(SCHEMA_BUNDLE)
    assert set(FIXTURES) == set(CONTRACTS) - {"source_inventory"}


def test_evaluation_report_golden_fixture_has_coherent_split_totals() -> None:
    report = FIXTURES["evaluation_report"]
    summary = cast(dict[str, Any], report["summary"])
    cases = cast(list[dict[str, Any]], report["cases"])
    assert summary["total"] == len(cases)
    assert summary["passed"] == sum(case["status"] == "pass" for case in cases)
    assert summary["failed"] == sum(case["status"] == "fail" for case in cases)
    assert summary["public_cases"] == sum(case["split"] == "public" for case in cases)
    assert summary["holdout_cases"] == sum(case["split"] == "holdout" for case in cases)
    assert summary["retrieval_fixtures"] == sum(case["scope"] == "retrieval" for case in cases)
    assert summary["model_runs"] == sum(
        cast(int, case["runs"]) for case in cases if case["scope"] == "model"
    )


@pytest.mark.parametrize("contract", CONTRACTS)
def test_golden_contract_fixtures_are_valid(contract: str) -> None:
    _validator(contract).validate(_instance(contract))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_contracts_reject_incompatible_versions(contract: str) -> None:
    instance = _instance(contract)
    instance["api_version"] = "valkeyrie.io/incompatible/2"
    with pytest.raises(ValidationError):
        _validator(contract).validate(instance)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_contracts_reject_missing_required_fields(contract: str) -> None:
    instance = _instance(contract)
    del instance[REQUIRED_FIELD[contract]]
    with pytest.raises(ValidationError):
        _validator(contract).validate(instance)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_contracts_reject_unknown_fields(contract: str) -> None:
    instance = _instance(contract)
    instance["unreviewed_extension"] = True
    with pytest.raises(ValidationError):
        _validator(contract).validate(instance)


def test_source_contract_preserves_fail_closed_defaults_and_reasons() -> None:
    inventory = _instance("source_inventory")
    defaults = cast(dict[str, Any], inventory["defaults"])
    defaults["unknown_repository_behavior"] = "include"
    with pytest.raises(ValidationError):
        _validator("source_inventory").validate(inventory)

    inventory = _instance("source_inventory")
    repositories = cast(list[dict[str, Any]], inventory["repositories"])
    excluded = next(item for item in repositories if item["classification"] == "excluded")
    del excluded["reason"]
    with pytest.raises(ValidationError):
        _validator("source_inventory").validate(inventory)


def test_source_identity_contracts_reject_unsafe_repository_paths() -> None:
    for unsafe_path in ("/README.md", "../README.md", "docs/../README.md", "docs\\README.md"):
        sidecar = _instance("metadata_sidecar")
        sidecar["path"] = unsafe_path
        with pytest.raises(ValidationError):
            _validator("metadata_sidecar").validate(sidecar)

        evidence = _instance("evidence_record")
        source = cast(dict[str, Any], evidence["source"])
        source["path"] = unsafe_path
        with pytest.raises(ValidationError):
            _validator("evidence_record").validate(evidence)


def test_model_output_contract_supports_only_bounded_outcomes() -> None:
    clarification = {
        "api_version": "valkeyrie.io/model-output/1",
        "kind": "ModelOutput",
        "outcome": "clarification",
        "question": "Which Valkey version do you mean?",
    }
    abstention = {
        "api_version": "valkeyrie.io/model-output/1",
        "kind": "ModelOutput",
        "outcome": "abstention",
        "reason": "The reviewed evidence does not support an answer.",
    }
    _validator("model_output").validate(clarification)
    _validator("model_output").validate(abstention)

    invalid = deepcopy(clarification)
    invalid["outcome"] = "write"
    with pytest.raises(ValidationError):
        _validator("model_output").validate(invalid)


def test_model_output_answer_requires_per_claim_evidence_not_global_ids() -> None:
    legacy = {
        "api_version": "valkeyrie.io/model-output/1",
        "kind": "ModelOutput",
        "outcome": "answer",
        "answer": "GET returns the value stored at a key.",
        "evidence_ids": ["ev_get-command"],
    }
    with pytest.raises(ValidationError):
        _validator("model_output").validate(legacy)

    answer = _instance("model_output")
    claims = cast(list[dict[str, Any]], answer["claims"])
    claims[0]["evidence_ids"] = []
    with pytest.raises(ValidationError):
        _validator("model_output").validate(answer)

    answer = _instance("model_output")
    claims = cast(list[dict[str, Any]], answer["claims"])
    del claims[0]["claim_id"]
    with pytest.raises(ValidationError):
        _validator("model_output").validate(answer)

    answer = _instance("model_output")
    claims = cast(list[dict[str, Any]], answer["claims"])
    claims[0]["source_url"] = "https://example.invalid"
    with pytest.raises(ValidationError):
        _validator("model_output").validate(answer)


def test_request_audit_embeds_full_exact_live_observations() -> None:
    audit = _instance("request_audit")
    observations = cast(list[dict[str, Any]], audit["live_observations"])
    _validator("live_observation").validate(observations[0])
    assert observations[0]["payload"] == {"tag_name": "9.0.0"}

    legacy = _instance("request_audit")
    legacy["live_observation_ids"] = [cast(str, observations[0]["observation_id"])]
    with pytest.raises(ValidationError):
        _validator("request_audit").validate(legacy)

    prefix_only = _instance("request_audit")
    prefix_only["live_observations"] = [observations[0]["observation_id"]]
    with pytest.raises(ValidationError):
        _validator("request_audit").validate(prefix_only)

    for missing_field in ("payload", "payload_digest"):
        missing = _instance("request_audit")
        incomplete = cast(list[dict[str, Any]], missing["live_observations"])
        del incomplete[0][missing_field]
        with pytest.raises(ValidationError):
            _validator("request_audit").validate(missing)

    non_canonical_digest = _instance("request_audit")
    non_canonical_digest["request_content_digest"] = "md5:abc"
    with pytest.raises(ValidationError):
        _validator("request_audit").validate(non_canonical_digest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observation_id", "obs_caller-prefix"),
        ("observation_id", f"obs_{'A' * 64}"),
        ("observation_id", f"obs_{'a' * 63}"),
        ("complete", False),
        ("truncated", True),
    ],
)
def test_live_observation_identity_and_status_are_exact(field: str, value: object) -> None:
    observation = _instance("live_observation")
    observation[field] = value
    with pytest.raises(ValidationError):
        _validator("live_observation").validate(observation)


def test_formats_are_enforced_not_annotations_only() -> None:
    observation = _instance("live_observation")
    observation["observed_at"] = "not-a-timestamp"
    with pytest.raises(ValidationError):
        _validator("live_observation").validate(observation)

    evidence = _instance("evidence_record")
    source = cast(dict[str, Any], evidence["source"])
    source["immutable_url"] = "http://example.invalid/evidence"
    with pytest.raises(ValidationError):
        _validator("evidence_record").validate(evidence)
