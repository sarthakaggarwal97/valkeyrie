from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from tests.helpers import load_yaml

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "docs" / "revision-action-policy.yaml"
SCHEMA_PATH = ROOT / "src" / "valkeyrie" / "schemas" / "contracts.schema.json"
THREAT_MODEL_PATH = ROOT / "docs" / "security" / "threat-model.yaml"

PINNED_FIELDS = [
    "generation_id",
    "prompt_revision",
    "application_revision",
    "model_revision",
    "inference_config_revision",
]
EXPECTED_LIFETIME_COMPONENTS = {
    "queue_retry_lifetime",
    "dead_letter_redrive_lifetime",
    "expired_claim_recovery_lifetime",
    "terminal_unconfirmed_reconciliation_lifetime",
}
EXPECTED_ARTIFACTS = {
    "corpus_generation": (
        {"generation_id"},
        "sealed_generation_and_retrievable_index_state",
        "pinned generation filter",
    ),
    "prompt_package": (
        {"prompt_revision"},
        "immutable_prompt_package",
        "pinned prompt package",
    ),
    "application_package": (
        {"application_revision"},
        "immutable_routable_worker_artifact",
        "execute the pinned worker artifact",
    ),
    "answer_model_profile": (
        {"model_revision", "inference_config_revision"},
        "immutable_model_profile_and_inference_configuration",
        "without fallback substitution",
    ),
}
EXPECTED_GATES = {
    "D-01": (
        "deploy_development_knowledge_plane",
        {
            "application_deployment",
            "slack_deployment_or_traffic",
            "live_github_credentials",
            "public_beta",
        },
    ),
    "D-02": (
        "deploy_development_application_and_prompt_revision",
        {
            "corpus_activation",
            "slack_deployment_or_traffic",
            "live_github_credentials",
            "public_beta",
        },
    ),
    "P2-D1": (
        "deploy_disabled_slack_stack_and_run_bounded_synthetic_window",
        {"real_slack_pilot", "live_github_credentials", "public_beta"},
    ),
    "P2-D5": (
        "enable_timeboxed_real_slack_pilot",
        {"unlisted_slack_scope", "live_github_credentials", "public_beta"},
    ),
    "P3-D1": (
        "provision_live_github_credential_and_deploy_disabled_live_state",
        {"live_capability_enablement", "project_state_writes", "public_beta"},
    ),
    "P3-11": (
        "enable_read_only_public_beta",
        {"private_sources", "expanded_slack_scope", "project_state_writes"},
    ),
}
EXPECTED_ROUTINE_VALIDATION = {
    "source_or_authority_policy": {
        "source_schema",
        "inventory_partition",
        "path_and_authority_tests",
        "affected_evaluations",
    },
    "prompt_or_routing": {
        "prompt_contract",
        "complete_visible_and_holdout_suite",
        "safety_and_injection_cases",
        "answer_diff",
    },
    "evaluation_contract": {
        "manifest_and_schema",
        "exact_suite_semantics",
        "holdout_custody",
        "threshold_regressions",
    },
    "answer_model_or_inference_configuration": {
        "complete_model_qualification",
        "safety_and_reliability_gates",
        "immutable_profile",
    },
    "embedding_chunking_or_index_configuration": {
        "retrieval_qualification",
        "compatibility_preflight",
        "deliberate_full_index_migration",
    },
    "corpus_generation": {
        "exact_revision_resolution",
        "reproducible_build",
        "complete_manifest",
        "evaluation_and_smoke_tests",
    },
}


def _document() -> dict[str, object]:
    return load_yaml(POLICY_PATH)


def _records(container: dict[str, object], field: str) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], container[field])


def _by_id(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    indexed = {cast(str, record["id"]): record for record in records}
    assert len(indexed) == len(records)
    return indexed


def test_policy_identity_scope_and_references_are_exact() -> None:
    document = _document()
    assert set(document) == {
        "api_version",
        "kind",
        "status",
        "scope",
        "design_references",
        "request_revision_contract",
        "artifact_retention",
        "external_action_policy",
        "routine_change_policy",
    }
    assert document["api_version"] == "valkeyrie.io/revision-action-policy/1"
    assert document["kind"] == "RevisionAndExternalActionPolicy"
    assert document["status"] == "approved_boundary_design_not_deployed"
    assert document["scope"] == {
        "current_authorization": "local_non_public_phase_1",
        "external_actions_default": "deny_unless_named_gate_or_authorized_protected_path",
        "routine_change_default": "normal_review_plus_automated_validation",
        "project_state_writes": "prohibited",
    }

    references = cast(list[str], document["design_references"])
    assert references and len(references) == len(set(references))
    for reference in references:
        if reference.startswith("task:"):
            assert re.fullmatch(r"task:[A-Z0-9-]+", reference)
            continue
        relative, separator, section = reference.partition("#")
        path = ROOT / relative
        assert path.is_file()
        if separator:
            assert section


def test_first_claim_pins_the_complete_request_audit_revision_set() -> None:
    contract = cast(dict[str, object], _document()["request_revision_contract"])
    assert contract["pin_on"] == "first_idempotent_claim"
    assert contract["required_fields"] == PINNED_FIELDS

    schema = load_yaml(SCHEMA_PATH)
    definitions = cast(dict[str, dict[str, object]], schema["$defs"])
    request_audit = definitions["request_audit"]
    schema_required = set(cast(list[str], request_audit["required"]))
    assert set(PINNED_FIELDS) <= schema_required

    rules = cast(list[str], contract["rules"])
    assert len(rules) == 4
    combined = " ".join(rules)
    for phrase in (
        "first successful claim",
        "executes the pinned application revision and configuration",
        "merely recording old identifiers",
        (
            "No attempt substitutes the current active generation, prompt, application, "
            "model, or inference configuration"
        ),
        "Static evidence and live-observation references",
    ):
        assert phrase in combined


def test_artifacts_remain_addressable_for_the_maximum_recovery_lifetime() -> None:
    retention = cast(dict[str, object], _document()["artifact_retention"])
    assert retention["retention_floor"] == (
        "maximum_retry_redrive_recovery_and_reconciliation_lifetime"
    )
    assert set(cast(list[str], retention["lifetime_components"])) == (EXPECTED_LIFETIME_COMPONENTS)
    artifacts = _by_id(_records(retention, "artifacts"))
    assert set(artifacts) == set(EXPECTED_ARTIFACTS)
    covered_fields: set[str] = set()
    for artifact_id, (fields, retained_object, execution_phrase) in EXPECTED_ARTIFACTS.items():
        artifact = artifacts[artifact_id]
        assert set(artifact) == {
            "id",
            "revision_fields",
            "retained_object",
            "execution_requirement",
        }
        actual_fields = set(cast(list[str], artifact["revision_fields"]))
        assert actual_fields == fields
        covered_fields.update(actual_fields)
        assert artifact["retained_object"] == retained_object
        assert execution_phrase in cast(str, artifact["execution_requirement"])
    assert covered_fields == set(PINNED_FIELDS)

    assert retention["deletion_requires"] == [
        (
            "No nonterminal request, queue, redrive, expired claim, send intent, or "
            "reconciliation record references the artifact."
        ),
        (
            "The maximum configured lifetime has elapsed since the last possible reference "
            "was created."
        ),
        "The artifact is not a retained rollback target.",
    ]
    assert retention["unavailable_provider_model_behavior"] == (
        "explicit_unavailable_outcome_no_model_or_configuration_substitution"
    )


def test_only_six_external_actions_require_bespoke_confirmation() -> None:
    document = _document()
    policy = cast(dict[str, object], document["external_action_policy"])
    gates = _by_id(_records(policy, "gates"))
    assert set(gates) == set(EXPECTED_GATES)
    for gate_id, (action, exclusions) in EXPECTED_GATES.items():
        gate = gates[gate_id]
        assert set(gate) == {"id", "action", "authorizes", "does_not_authorize"}
        assert gate["action"] == action
        assert cast(str, gate["authorizes"]).strip()
        assert set(cast(list[str], gate["does_not_authorize"])) == exclusions

    threat_scope = cast(dict[str, object], load_yaml(THREAT_MODEL_PATH)["scope"])
    assert set(cast(list[str], threat_scope["external_action_gates"])) == set(gates)
    assert "only its named action" in cast(str, policy["non_transference"])
    assert "exact reviewed revision" in cast(str, policy["approval_binding"])


def test_routine_changes_use_review_and_class_specific_validation_not_new_gates() -> None:
    routine = cast(dict[str, object], _document()["routine_change_policy"])
    assert routine["confirmation"] == (
        "no_new_bespoke_gate_when_the_approved_boundary_is_unchanged"
    )
    assert routine["review"] == "normal_codeowners_review"
    classes = _by_id(_records(routine, "classes"))
    assert set(classes) == set(EXPECTED_ROUTINE_VALIDATION)
    for class_id, validation in EXPECTED_ROUTINE_VALIDATION.items():
        record = classes[class_id]
        assert set(record) == {"id", "automated_validation"}
        assert set(cast(list[str], record["automated_validation"])) == validation

    rules = cast(list[str], routine["rules"])
    combined = " ".join(rules)
    assert "never grants a credential or executes an external action" in combined
    assert "already authorized target and boundary" in combined
    assert "not an implicit seventh gate" in combined
