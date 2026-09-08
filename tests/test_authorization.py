import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION = (ROOT / "PROTOTYPE_AUTHORIZATION.md").read_text(encoding="utf-8")
NORMALIZED_AUTHORIZATION = " ".join(AUTHORIZATION.split())
RFC_SHA256 = "75510aa51200ebdc6acc6968ebbfe9f0eb73940149de5f4380148c05f19b9012"


def test_authorization_binds_exact_rfc_and_owner() -> None:
    rfc = ROOT / "docs/rfcs/valkeyrie-maintainer-rfc.md"
    assert hashlib.sha256(rfc.read_bytes()).hexdigest() == RFC_SHA256
    required = {
        "Status: Authorized for local private-prototype implementation through Phase 1",
        "Decision date: 2026-08-18",
        "Backlog decisions: P0-01 and P0-03",
        "Sarthak Aggarwal (`sarthakaggarwal97`)",
        "`Final RFC for maintainer review`",
        RFC_SHA256,
        "Any byte change to that artifact requires renewed review",
    }
    for value in required:
        assert value in NORMALIZED_AUTHORIZATION


def test_authorization_records_personal_vehicle_and_target_without_deployment() -> None:
    for value in (
        "https://github.com/sarthakaggarwal97/valkeyrie",
        "0485ace3806e3fad69ea82dcd359ab9f2685dbbc",
        "personal account `968533178160`, region `us-east-1`",
        "Recording an AWS target does not authorize any AWS mutation",
        "not adopted as an official Valkey project",
    ):
        assert value in AUTHORIZATION


def test_slack_authorization_is_recorded_with_its_transport_and_bounds() -> None:
    # The earlier records exclude Slack credentials and traffic. Shipping a Slack app while
    # leaving only those would make the document wrong about what is deployed.
    for value in (
        "recorded in Home thread 338 on 2026-09-08",
        "authorizes a Slack app for this assistant",
        "runs in Socket Mode",
        "needs no inbound endpoint",
        "Nothing is exposed to the internet by this integration",
        "Authorization is bounded to answering questions",
    ):
        assert value in NORMALIZED_AUTHORIZATION


def test_tester_access_path_is_recorded_as_role_assumption_not_shared_credentials() -> None:
    # Sharing a long-lived key would be unattributable and is not what is deployed.
    for value in (
        "access is granted by role assumption",
        "valkeyrie-development-endpoint-caller",
        "trusts account 468997136233",
        "It grants nothing else",
        "Both invoke actions are required",
        "anonymous requests remain refused",
    ):
        assert value in NORMALIZED_AUTHORIZATION


def test_public_endpoint_approval_is_recorded_with_its_exception() -> None:
    # Shipping a public URL while the document still claimed the prototype was non-public
    # would leave the authorization record a lie that still passed its own tests.
    for value in (
        "Build the public URL with Fable",
        "recorded in Home thread 338 on 2026-08-26",
        "An unauthenticated url was attempted and is NOT in place",
        "epoxy-engage_mitigations automatically",
        "scoped to this AWS account and requires",
        "World-accessible exposure of this prototype is treated as prohibited",
    ):
        assert value in NORMALIZED_AUTHORIZATION


def test_p0_03_authorizes_local_phase_1_only() -> None:
    for value in (
        "Simplify the backlog gates and continue further implementation",
        "recorded in Home thread 338 on 2026-08-18",
        "authorizes non-public local Phase 1 engineering",
        "local schemas, validators, evaluation and threat-model work",
        "does not authorize AWS deployment",
    ):
        assert value in NORMALIZED_AUTHORIZATION


@pytest.mark.parametrize(
    "boundary",
    [
        "Slack credentials, app installation, deployment, synthetic traffic, real traffic",
        "live-state credentials or enabled request-time GitHub capabilities",
        "GitHub or project-state writes",
        "pushes, pull requests, workflow execution",
        "AWS provisioning, bootstrap, deployment, update, deletion, IAM mutation",
        "public beta, release-readiness decisions, or invocation of release automation",
    ],
)
def test_authorization_explicitly_excludes_ungated_actions(boundary: str) -> None:
    assert boundary in AUTHORIZATION


def test_official_adoption_is_a_hard_slack_prerequisite() -> None:
    assert "Official Valkey adoption" in NORMALIZED_AUTHORIZATION
    assert "hard prerequisites to P2-D1" in NORMALIZED_AUTHORIZATION


def test_d01_is_the_next_external_action_gate() -> None:
    assert "D-01 must approve" in NORMALIZED_AUTHORIZATION
    assert "before the first AWS mutation" in NORMALIZED_AUTHORIZATION
    assert "D-02 separately protects the later application deployment" in NORMALIZED_AUTHORIZATION
