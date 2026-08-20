from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from valkeyrie.aws_adapters import AwsLambdaInvoker, create_lambda_client
from valkeyrie.deployed_evaluation import (
    DeployedEvaluationError,
    DeployedEvaluationIdentity,
    run_deployed_evaluation,
)


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Invoke the ten real Valkeyrie questions exactly once against an immutable Lambda."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--function-name", required=True)
    parser.add_argument("--function-qualifier", required=True)
    parser.add_argument("--application-revision", required=True)
    parser.add_argument("--evaluation-suite-revision", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--knowledge-base-id", required=True)
    values = parser.parse_args(arguments)
    identity = DeployedEvaluationIdentity(
        function_name=values.function_name,
        function_qualifier=values.function_qualifier,
        application_revision=values.application_revision,
        evaluation_suite_revision=values.evaluation_suite_revision,
        generation_id=values.generation_id,
        knowledge_base_id=values.knowledge_base_id,
    )
    try:
        client = create_lambda_client(values.region)
        result = run_deployed_evaluation(
            values.root,
            values.journal,
            identity,
            AwsLambdaInvoker(client),
            clock=lambda: datetime.now(UTC),
        )
    except (DeployedEvaluationError, OSError, RuntimeError, ValueError) as error:
        print(f"deployed evaluation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "evidence_id": result.evidence_id,
                "manifest_path": result.manifest_path.as_posix(),
                "run_id": result.run_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
