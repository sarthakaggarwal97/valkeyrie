import json
from pathlib import Path

from infra.finalize_deployed_evaluation import finalize_deployed_evaluation
from valkeyrie.deployed_evaluation import DeployedEvaluationIdentity

ROOT = Path(__file__).resolve().parents[1]
LEDGER = (
    ROOT
    / "evals/deployed-attempts"
    / "624d06a0fde7265214b80cdf0eac3dcc9635fd8441d80eb2af816f19efd2cf2f"
)


def test_finalizes_exact_deployed_ledger_offline(tmp_path: Path) -> None:
    identity = DeployedEvaluationIdentity(
        "valkeyrie-development-application",
        "7",
        "sha256:08079d56dcbde9f2c44f421ae0b3dd7e1b157d3ba457d84d971acf3a5efb7f7e",
        "sha256:64f9a99faefc432eca78ed0bbf68d3fb6da3a0867e558fc07111ced476e66193",
        "sha256:6760b22e3e09ac0e2ac64c9c23b3a77733f5bd1f8f8c9704c66d2c69d1362b60",
        "ONVASJDDNX",
    )
    output, manifest = finalize_deployed_evaluation(ROOT, LEDGER, tmp_path, identity)

    assert manifest["evaluation_result"] == "fail"
    assert manifest["summary"] == {
        "case_count": 10,
        "attempt_count": 10,
        "result_count": 8,
        "passed": 6,
        "failed": 4,
        "timeouts": 2,
        "automatic_retries": 0,
    }
    observations = manifest["observations"]
    assert isinstance(observations, list)
    assert [item["index"] for item in observations] == list(range(1, 11))
    assert [item["verdict"] for item in observations] == [
        "pass",
        "pass",
        "fail",
        "pass",
        "pass",
        "fail",
        "pass",
        "fail",
        "fail",
        "pass",
    ]
    assert observations[2]["outcome"] == "timeout"
    assert observations[8]["outcome"] == "timeout"
    assert observations[5]["outcome"] == "abstention"
    assert observations[7]["outcome"] == "partial"
    assert json.loads(output.read_text()) == manifest
