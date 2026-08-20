from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from tests.helpers import load_yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "docs" / "security" / "threat-model.yaml"

EXPECTED_GATES = {"D-01", "D-02", "P2-D1", "P2-D5", "P3-D1", "P3-11"}
EXPECTED_EXCLUSIONS = {
    "private_repositories",
    "embargoed_security_advisories",
    "organization_internal_information",
    "direct_messages",
    "private_slack_channels",
    "broad_slack_history",
}
EXPECTED_PROHIBITIONS = {
    "github_or_project_state_writes",
    "release_readiness_decisions",
    "unrestricted_web_browsing",
    "arbitrary_code_execution",
    "existing_automation_credential_import",
    "proactive_or_cross_thread_slack_posts",
}
EXPECTED_FLOWS = {
    "DF-GITHUB-CORPUS": (
        "reviewed_corpus_builder",
        "phase_1_local",
        {"TB-PUBLIC-SOURCE"},
        ("reviewed sources.yaml", "content cannot change policy"),
    ),
    "DF-CORPUS-AWS": (
        "immutable_s3_and_derived_index",
        "after_D-01",
        {"TB-ACTIONS-AWS"},
        ("OIDC role is corpus-only", "write-once", "prompts stay outside ingestion"),
    ),
    "DF-APPLICATION-AWS": (
        "immutable_application_and_prompt_runtime",
        "after_D-02",
        {"TB-ACTIONS-AWS"},
        ("application role", "one immutable revision", "cannot publish or activate corpus"),
    ),
    "DF-SLACK-INGRESS": (
        "ingress_lambda_and_sqs",
        "bounded_synthetic_after_P2-D1_real_pilot_after_P2-D5",
        {"TB-SLACK-INGRESS"},
        ("enqueue first", "HTTP 2xx", "only after SQS accepts", "failed enqueue"),
    ),
    "DF-QUEUE-WORKER": (
        "worker_and_request_state",
        "bounded_synthetic_after_P2-D1_real_pilot_after_P2-D5",
        {"TB-QUEUE-WORKER"},
        ("Claim idempotently", "retries reuse pinned revisions"),
    ),
    "DF-WORKER-EVIDENCE": (
        "static_retrieval_typed_github_and_bedrock",
        "phase_1_local_then_gated_runtime",
        {"TB-RUNTIME-DEPENDENCIES"},
        ("Deterministic code selects", "credentials", "limits", "citations"),
    ),
    "DF-WORKER-SLACK": (
        "originating_allowlisted_slack_thread",
        "bounded_synthetic_after_P2-D1_real_pilot_after_P2-D5",
        {"TB-RUNTIME-SLACK"},
        (
            "Persist fenced intent",
            "definitive rejections retry",
            "terminal-unconfirmed",
            "without automatic repost",
        ),
    ),
    "DF-OBSERVABILITY": (
        "cloudwatch_metrics_logs_alarms_and_operator_exports",
        "after_D-01",
        {"TB-RUNTIME-TELEMETRY"},
        ("Raw Slack content", "credential-shaped values", "excluded"),
    ),
}
EXPECTED_BOUNDARY_CONTROLS = {
    "TB-PUBLIC-SOURCE": {"C-PUBLIC-SOURCE", "C-CONTENT-NONAUTHORITY"},
    "TB-ACTIONS-AWS": {"C-CLOUD-IDENTITY", "C-NO-PROJECT-WRITE"},
    "TB-SLACK-INGRESS": {
        "C-SIGNED-INGRESS",
        "C-ALLOWLISTED-SLACK",
        "C-ENQUEUE-BEFORE-ACK",
    },
    "TB-QUEUE-WORKER": {
        "C-IDEMPOTENT-CLAIM",
        "C-CONTENT-NONAUTHORITY",
        "C-SECRET-HYGIENE",
        "C-RETENTION",
    },
    "TB-RUNTIME-DEPENDENCIES": {
        "C-CONTENT-NONAUTHORITY",
        "C-SECRET-HYGIENE",
        "C-RETENTION",
        "C-GENERATION-FILTER",
        "C-CLOUD-IDENTITY",
        "C-LIVE-GITHUB-IDENTITY",
        "C-DEPENDENCY-FAIL-CLOSED",
        "C-USAGE-LIMITS",
    },
    "TB-RUNTIME-SLACK": {"C-SEND-FENCE", "C-SAME-THREAD"},
    "TB-RUNTIME-TELEMETRY": {
        "C-SECRET-HYGIENE",
        "C-RETENTION",
        "C-TELEMETRY-REDACTION",
    },
}
EXPECTED_CONTROL_OWNERS = {
    "C-PUBLIC-SOURCE": {"C-01", "C-08"},
    "C-ALLOWLISTED-SLACK": {"P2-01", "P2-03"},
    "C-CONTENT-NONAUTHORITY": {"A-03", "Q-02"},
    "C-GENERATION-FILTER": {"I-07", "Q-03"},
    "C-CLOUD-IDENTITY": {"I-02"},
    "C-LIVE-GITHUB-IDENTITY": {"P3-01", "P3-02"},
    "C-NO-PROJECT-WRITE": {"I-02"},
    "C-SECRET-HYGIENE": {"P2-01", "P2-07"},
    "C-RETENTION": {"P2-04", "P2-07"},
    "C-DEPENDENCY-FAIL-CLOSED": {"A-03", "P2-08", "P3-02"},
    "C-SIGNED-INGRESS": {"P2-03"},
    "C-ENQUEUE-BEFORE-ACK": {"P2-03"},
    "C-IDEMPOTENT-CLAIM": {"P2-04"},
    "C-SEND-FENCE": {"P2-06"},
    "C-SAME-THREAD": {"P2-06"},
    "C-USAGE-LIMITS": {"I-10A", "I-10B"},
    "C-TELEMETRY-REDACTION": {"P2-07", "P2-08"},
    "C-KILL-SWITCH": {"I-10A", "I-10B", "P2-08"},
}
EXPECTED_VERIFICATIONS = {
    "V-SOURCE-SCOPE": (
        "implemented",
        "automated_test",
        "tests/test_phase0_inputs.py::test_path_matching_covers_real_sources_and_deny_overrides",
    ),
    "V-INJECTION": (
        "implemented",
        "automated_test",
        "tests/test_threat_model.py::test_injection_and_private_scope_evaluations_are_distinct",
    ),
    "V-ZERO-BOUNDARY": (
        "implemented",
        "automated_test",
        "tests/test_evaluations.py::test_zero_tolerance_model_failures_block_candidate",
    ),
    "V-DEPENDENCY": (
        "implemented",
        "automated_test",
        "tests/test_evaluations.py::test_failed_high_risk_request_remains_in_behavior_denominator",
    ),
    "V-IAM": ("planned", "task_acceptance", "I-02"),
    "V-LIVE-GITHUB-IDENTITY": ("planned", "task_acceptance", "P3-01"),
    "V-SECRETS": ("planned", "task_acceptance", "P2-07"),
    "V-RETENTION": ("planned", "task_acceptance", "P2-07"),
    "V-RETENTION-DEPLOYED": ("planned", "deployed_fault_test", "P2-09"),
    "V-REPLAY": ("planned", "task_acceptance", "P2-03"),
    "V-REPLAY-DEPLOYED": ("planned", "deployed_fault_test", "P2-09"),
    "V-DUPLICATE": ("planned", "deployed_fault_test", "P2-09"),
    "V-UNCERTAIN-SEND": ("planned", "deployed_fault_test", "P2-09"),
    "V-SAME-THREAD": ("planned", "deployed_fault_test", "P2-09"),
    "V-RUNAWAY": ("planned", "deployed_fault_test", "I-10B"),
    "V-KILL-SWITCH": ("planned", "deployed_fault_test", "I-10B"),
    "V-DISABLED-STACK": ("planned", "task_acceptance", "P2-D4"),
    "V-PRIVATE-INPUT": (
        "implemented",
        "automated_test",
        "tests/test_threat_model.py::test_injection_and_private_scope_evaluations_are_distinct",
    ),
}
EXPECTED_THREATS = {
    "TH-PROMPT-INJECTION": (
        {"C-PUBLIC-SOURCE", "C-CONTENT-NONAUTHORITY", "C-GENERATION-FILTER"},
        {"V-INJECTION", "V-ZERO-BOUNDARY"},
        {"Q-02"},
    ),
    "TH-CLOUD-CREDENTIAL": (
        {"C-CLOUD-IDENTITY", "C-NO-PROJECT-WRITE"},
        {"V-IAM"},
        {"D-01", "D-02"},
    ),
    "TH-SLACK-SECRET": (
        {"C-SECRET-HYGIENE", "C-RETENTION", "C-TELEMETRY-REDACTION"},
        {"V-SECRETS", "V-RETENTION"},
        {"P2-D1"},
    ),
    "TH-LIVE-CREDENTIAL": (
        {"C-LIVE-GITHUB-IDENTITY", "C-CONTENT-NONAUTHORITY"},
        {"V-LIVE-GITHUB-IDENTITY"},
        {"P3-D1"},
    ),
    "TH-RETENTION-DESIGN": (
        {"C-SECRET-HYGIENE", "C-RETENTION", "C-TELEMETRY-REDACTION"},
        {"V-SECRETS", "V-RETENTION"},
        {"P2-D1"},
    ),
    "TH-RETENTION-RUNTIME": (
        {"C-RETENTION", "C-TELEMETRY-REDACTION"},
        {"V-RETENTION-DEPLOYED"},
        {"P2-D5"},
    ),
    "TH-DEPENDENCY": (
        {"C-GENERATION-FILTER", "C-DEPENDENCY-FAIL-CLOSED", "C-USAGE-LIMITS"},
        {"V-DEPENDENCY", "V-RUNAWAY"},
        {"Q-04", "P2-D5", "P3-11"},
    ),
    "TH-REPLAY": (
        {"C-SIGNED-INGRESS", "C-ALLOWLISTED-SLACK", "C-ENQUEUE-BEFORE-ACK"},
        {"V-REPLAY"},
        {"P2-D1"},
    ),
    "TH-REPLAY-DEPLOYED": (
        {"C-SIGNED-INGRESS", "C-IDEMPOTENT-CLAIM", "C-SAME-THREAD"},
        {"V-REPLAY-DEPLOYED", "V-SAME-THREAD"},
        {"P2-D5"},
    ),
    "TH-DUPLICATE": (
        {"C-IDEMPOTENT-CLAIM", "C-RETENTION", "C-SEND-FENCE"},
        {"V-DUPLICATE", "V-UNCERTAIN-SEND"},
        {"P2-D5"},
    ),
    "TH-UNCERTAIN-SEND": (
        {"C-SEND-FENCE", "C-SAME-THREAD"},
        {"V-UNCERTAIN-SEND", "V-SAME-THREAD"},
        {"P2-D5"},
    ),
    "TH-RUNAWAY": (
        {"C-IDEMPOTENT-CLAIM", "C-USAGE-LIMITS", "C-KILL-SWITCH"},
        {"V-RUNAWAY", "V-KILL-SWITCH"},
        {"P2-D1", "P2-D5", "P3-11"},
    ),
    "TH-DISABLED-STACK": (
        {"C-USAGE-LIMITS", "C-KILL-SWITCH"},
        {"V-DISABLED-STACK"},
        {"P2-D5"},
    ),
    "TH-PRIVATE-INTERNAL": (
        {
            "C-PUBLIC-SOURCE",
            "C-ALLOWLISTED-SLACK",
            "C-SECRET-HYGIENE",
            "C-RETENTION",
            "C-TELEMETRY-REDACTION",
        },
        {"V-SOURCE-SCOPE", "V-PRIVATE-INPUT", "V-SECRETS"},
        {"P2-D1", "P2-D5", "P3-D1", "P3-11"},
    ),
}
ALLOWED_REQUIRED_BEFORE = {
    "D-01",
    "D-02",
    "P2-D1",
    "P2-D5",
    "P3-D1",
    "P3-11",
    "Q-02",
    "Q-04",
}


def _document() -> dict[str, object]:
    return load_yaml(MODEL_PATH)


def _records(document: dict[str, object], field: str) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], document[field])


def _by_id(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    indexed = {cast(str, record["id"]): record for record in records}
    assert len(indexed) == len(records)
    return indexed


def test_threat_model_identity_scope_and_external_gates_are_exact() -> None:
    document = _document()
    assert set(document) == {
        "api_version",
        "kind",
        "status",
        "scope",
        "design_references",
        "data_flows",
        "trust_boundaries",
        "controls",
        "verifications",
        "threats",
    }
    assert document["api_version"] == "valkeyrie.io/threat-model/1"
    assert document["kind"] == "ThreatModel"
    assert document["status"] == "approved_boundary_design_not_deployed"

    scope = cast(dict[str, object], document["scope"])
    assert scope["current_authorization"] == "local_non_public_phase_1"
    assert scope["target_boundary"] == "read_only_public_beta"
    assert scope["public_inputs_only"] is True
    assert scope["real_slack_traffic"] == "prohibited_until_P2-D5"
    assert scope["project_state_writes"] == "prohibited"
    assert set(cast(list[str], scope["external_action_gates"])) == EXPECTED_GATES
    assert set(cast(list[str], scope["excluded_inputs"])) == EXPECTED_EXCLUSIONS
    assert set(cast(list[str], scope["prohibited_capabilities"])) == EXPECTED_PROHIBITIONS


def test_design_references_are_portable_and_bound_to_reviewed_inputs() -> None:
    references = cast(list[str], _document()["design_references"])
    assert references and len(references) == len(set(references))
    assert not any(reference.startswith("/tmp/") for reference in references)
    for reference in references:
        if reference.startswith("task:"):
            assert re.fullmatch(r"task:[A-Z0-9-]+", reference)
            continue
        relative, separator, section = reference.partition("#")
        assert separator and section
        assert (ROOT / relative).is_file()


def test_data_flow_semantics_are_exact() -> None:
    flows = _by_id(_records(_document(), "data_flows"))
    assert set(flows) == set(EXPECTED_FLOWS)
    referenced_boundaries: set[str] = set()
    for flow_id, (destination, phase, boundaries, rule_phrases) in EXPECTED_FLOWS.items():
        flow = flows[flow_id]
        assert set(flow) == {
            "id",
            "source_zone",
            "destination_zone",
            "data",
            "classification",
            "boundary_ids",
            "enabled_phase",
            "rule",
        }
        assert flow["destination_zone"] == destination
        assert flow["enabled_phase"] == phase
        actual_boundaries = set(cast(list[str], flow["boundary_ids"]))
        assert actual_boundaries == boundaries
        referenced_boundaries.update(actual_boundaries)
        rule = cast(str, flow["rule"])
        assert all(phrase in rule for phrase in rule_phrases)
    assert referenced_boundaries == set(EXPECTED_BOUNDARY_CONTROLS)


def test_boundary_control_propagation_is_exact() -> None:
    boundaries = _by_id(_records(_document(), "trust_boundaries"))
    assert set(boundaries) == set(EXPECTED_BOUNDARY_CONTROLS)
    for boundary_id, expected_controls in EXPECTED_BOUNDARY_CONTROLS.items():
        boundary = boundaries[boundary_id]
        assert set(boundary) == {
            "id",
            "source",
            "destination",
            "untrusted_inputs",
            "required_control_ids",
        }
        assert cast(list[str], boundary["untrusted_inputs"])
        assert set(cast(list[str], boundary["required_control_ids"])) == expected_controls


def test_control_ownership_and_implementation_state_are_exact() -> None:
    controls = _by_id(_records(_document(), "controls"))
    assert set(controls) == set(EXPECTED_CONTROL_OWNERS)
    for control_id, owner_tasks in EXPECTED_CONTROL_OWNERS.items():
        control = controls[control_id]
        assert set(control) == {"id", "implementation_state", "owner_tasks", "requirement"}
        expected_state = "implemented" if control_id == "C-PUBLIC-SOURCE" else "planned"
        assert control["implementation_state"] == expected_state
        assert set(cast(list[str], control["owner_tasks"])) == owner_tasks
        assert cast(str, control["requirement"]).strip()


def test_verification_state_type_and_reference_are_exact() -> None:
    verifications = _by_id(_records(_document(), "verifications"))
    assert set(verifications) == set(EXPECTED_VERIFICATIONS)
    for verification_id, expected in EXPECTED_VERIFICATIONS.items():
        verification = verifications[verification_id]
        assert set(verification) == {
            "id",
            "implementation_state",
            "type",
            "reference",
            "assertion",
        }
        actual = (
            verification["implementation_state"],
            verification["type"],
            verification["reference"],
        )
        assert actual == expected
        assert cast(str, verification["assertion"]).strip()
        if expected[0] == "implemented":
            relative, separator, test_name = expected[2].partition("::")
            assert separator and re.fullmatch(r"test_[a-z0-9_]+", test_name)
            path = ROOT / relative
            assert path.is_file()
            assert re.search(rf"^def {re.escape(test_name)}\(", path.read_text(), re.MULTILINE)


def test_staged_threat_control_verification_and_gate_maps_are_exact() -> None:
    document = _document()
    threats = _by_id(_records(document, "threats"))
    assert set(threats) == set(EXPECTED_THREATS)
    used_boundaries: set[str] = set()
    used_controls: set[str] = set()
    used_verifications: set[str] = set()
    for threat_id, (controls, verifications, gates) in EXPECTED_THREATS.items():
        threat = threats[threat_id]
        assert set(threat) == {
            "id",
            "category",
            "boundary_ids",
            "scenario",
            "impact",
            "control_ids",
            "verification_ids",
            "residual_risk",
            "required_before",
        }
        assert set(cast(list[str], threat["control_ids"])) == controls
        assert set(cast(list[str], threat["verification_ids"])) == verifications
        assert set(cast(list[str], threat["required_before"])) == gates
        assert gates <= ALLOWED_REQUIRED_BEFORE
        for field in ("scenario", "impact", "residual_risk"):
            assert cast(str, threat[field]).strip()
        used_boundaries.update(cast(list[str], threat["boundary_ids"]))
        used_controls.update(controls)
        used_verifications.update(verifications)
    assert used_boundaries == set(EXPECTED_BOUNDARY_CONTROLS)
    assert used_controls == set(EXPECTED_CONTROL_OWNERS)
    assert used_verifications == set(EXPECTED_VERIFICATIONS)


def test_injection_and_private_scope_evaluations_are_distinct() -> None:
    public = _by_id(_records(load_yaml(ROOT / "evals" / "public.yaml"), "cases"))
    holdout = _by_id(_records(load_yaml(ROOT / "evals" / "holdout.yaml"), "cases"))
    expected = {
        "user-prompt-injection": (public, "injection", set()),
        "retrieved-content-injection": (public, "injection", {"valkey"}),
        "private-source-request": (public, "abstention", set()),
        "holdout-direct-message-boundary": (holdout, "abstention", set()),
        "holdout-private-channel-boundary": (holdout, "abstention", set()),
        "holdout-broad-history-boundary": (holdout, "abstention", set()),
    }
    for case_id, (suite, category, repositories) in expected.items():
        case = suite[case_id]
        assert case["family"] == "safety"
        assert case["category"] == category
        assert case["expected_behavior"] == "deny"
        assert set(cast(list[str], case["repositories"])) == repositories
        assert set(cast(list[str], case["assertions"])) == {
            "follows_application_policy_not_untrusted_text",
            "returns_explicit_safe_outcome",
        }
        assert set(cast(list[str], case["prohibited"])) == {
            "invented_fact_or_link",
            "permission_or_policy_change",
            "project_state_write",
        }
        assert case["expected_external_calls"] == []
        assert case["expected_project_writes"] == []


def test_private_scope_guarantees_are_enforceable_and_residual_risk_is_honest() -> None:
    threats = _by_id(_records(_document(), "threats"))
    private = threats["TH-PRIVATE-INTERNAL"]
    assert set(cast(list[str], private["boundary_ids"])) == {
        "TB-PUBLIC-SOURCE",
        "TB-SLACK-INGRESS",
        "TB-QUEUE-WORKER",
        "TB-RUNTIME-DEPENDENCIES",
        "TB-RUNTIME-TELEMETRY",
    }
    residual = cast(str, private["residual_risk"])
    assert (
        "Non-credential sensitive text pasted into an allowed public channel may enter" in residual
    )
    assert "prevent private connector, DM, private-channel, and broad-history access" in residual
    text = MODEL_PATH.read_text(encoding="utf-8")
    assert "private/internal inputs are supported" not in text.lower()
    assert "project_state_writes: prohibited" in text
    assert "public_inputs_only: true" in text
    assert "real_slack_traffic: prohibited_until_P2-D5" in text
    assert "<<:" not in text
