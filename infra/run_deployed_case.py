from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from infra.continue_deployed_evaluation import _bounded_result
from valkeyrie.aws_adapters import AwsLambdaInvoker, create_lambda_client
from valkeyrie.deployed_evaluation import (
    DeployedEvaluationError,
    DeployedEvaluationIdentity,
    DeployedInvocation,
    DeployedInvoker,
    _canonical_bytes,
    _content_id,
    _fsync_directory,
    _real_cases,
    _request_id,
    _sha256,
    _timestamp,
    _validate_inputs,
    _validate_result,
    _write_exclusive,
)


@dataclass(frozen=True)
class DeployedCaseResult:
    case_id: str
    request_id: str
    journal_directory: Path
    result_path: Path
    response: Mapping[str, object]


def run_deployed_case_once(
    root: Path,
    ledger_root: Path,
    journal_directory: Path,
    identity: DeployedEvaluationIdentity,
    case_index: int,
    invoker: DeployedInvoker,
    *,
    clock: Callable[[], datetime],
) -> DeployedCaseResult:
    """Run one previously unattempted real case after checking the shared ledger."""
    _validate_inputs(root, journal_directory, identity, invoker)
    project_root = root.resolve(strict=True)
    ledger = ledger_root.resolve(strict=True)
    try:
        journal_directory.parent.resolve(strict=True).relative_to(ledger)
    except ValueError as error:
        raise DeployedEvaluationError("case journal must remain under its ledger root") from error
    cases = _real_cases(project_root)
    if not isinstance(case_index, int) or isinstance(case_index, bool) or not 1 <= case_index <= 10:
        raise DeployedEvaluationError("deployed case index is outside the reviewed suite")
    case = cases[case_index - 1]
    case_id = cast(str, case["id"])
    run_identity = {
        "api_version": "valkeyrie.io/deployed-evaluation-run/1",
        "kind": "DeployedEvaluationRun",
        **asdict(identity),
        "case_ids": [item["id"] for item in cases],
    }
    run_id = _content_id("deployed-evaluation-run/1", run_identity)
    request_id = _request_id(run_id, case_id)
    _assert_unattempted(ledger, case_index, case_id, request_id)

    try:
        journal_directory.mkdir(mode=0o700)
        _fsync_directory(journal_directory.parent)
    except FileExistsError as error:
        raise DeployedEvaluationError("refusing to resume or repeat a case journal") from error
    except OSError as error:
        raise DeployedEvaluationError("cannot create deployed case journal") from error
    attempts = journal_directory / "attempts"
    results = journal_directory / "results"
    attempts.mkdir(mode=0o700)
    results.mkdir(mode=0o700)
    _fsync_directory(journal_directory)

    case_run = {
        "api_version": "valkeyrie.io/deployed-evaluation-case-run/1",
        "kind": "DeployedEvaluationCaseRun",
        "logical_run_id": run_id,
        "identity": asdict(identity),
        "index": case_index,
        "case_id": case_id,
        "request_id": request_id,
    }
    _write_exclusive(journal_directory / "case.json", _canonical_bytes(case_run))
    timestamp = _timestamp(clock())
    event = {
        "action": "answer",
        "request_id": request_id,
        "question": case["question"],
        "version_requirement": "current_state" if case["expected_external_calls"] else "none",
        "requested_version": None,
        "knowledge_base_id": identity.knowledge_base_id,
        "owner": f"deployed-evaluation-{run_id[-12:]}",
        "lease_duration_seconds": 300,
        "now": timestamp,
        "completed_at": timestamp,
    }
    payload = _canonical_bytes(event, newline=False)
    request_hash = _sha256(payload)
    stem = f"{case_index:02d}-{case_id}"
    attempt_path = attempts / f"{stem}.json"
    _write_exclusive(
        attempt_path,
        _canonical_bytes(
            {
                "api_version": "valkeyrie.io/deployed-evaluation-attempt/1",
                "kind": "DeployedEvaluationAttempt",
                "run_id": run_id,
                "index": case_index,
                "case_id": case_id,
                "request_id": request_id,
                "request_sha256": request_hash,
                "function_name": identity.function_name,
                "function_qualifier": identity.function_qualifier,
                "application_revision": identity.application_revision,
                "evaluation_suite_revision": identity.evaluation_suite_revision,
                "generation_id": identity.generation_id,
                "attempted": True,
                "request": event,
            }
        ),
    )
    invocation = DeployedInvocation(
        identity.function_name,
        identity.function_qualifier,
        identity.application_revision,
        identity.evaluation_suite_revision,
        identity.generation_id,
        request_id,
        payload,
    )
    try:
        raw_result = invoker.invoke(invocation)
    except Exception as error:
        raise DeployedEvaluationError(
            f"case {case_index} stopped after its journaled attempt; retry is forbidden"
        ) from error
    result = _bounded_result(raw_result)
    result_hash = _sha256(_canonical_bytes(result, newline=False))
    result_path = results / f"{stem}.json"
    _write_exclusive(
        result_path,
        _canonical_bytes(
            {
                "api_version": "valkeyrie.io/deployed-evaluation-result/1",
                "kind": "DeployedEvaluationResult",
                "run_id": run_id,
                "case_id": case_id,
                "request_id": request_id,
                "request_sha256": request_hash,
                "response_sha256": result_hash,
                "response": result,
            }
        ),
    )
    _validate_result(
        result,
        request_id,
        identity.generation_id,
        bool(case["expected_external_calls"]),
        cast(str, case["expected_behavior"]),
    )
    return DeployedCaseResult(case_id, request_id, journal_directory, result_path, result)


def _assert_unattempted(
    ledger: Path,
    case_index: int,
    case_id: str,
    request_id: str,
) -> None:
    matches: list[Path] = []
    for path in ledger.rglob("*.json"):
        if path.parent.name != "attempts":
            continue
        try:
            value = json.loads(path.read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DeployedEvaluationError(
                "dispatch ledger contains an unreadable attempt"
            ) from error
        if not isinstance(value, Mapping):
            raise DeployedEvaluationError("dispatch ledger contains a malformed attempt")
        if (
            value.get("index") == case_index
            or value.get("case_id") == case_id
            or value.get("request_id") == request_id
        ):
            matches.append(path)
    if matches:
        raise DeployedEvaluationError(
            f"case {case_index} already has a journaled attempt; retry is forbidden"
        )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Invoke one previously unattempted real case with a shared exact-once ledger."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--function-name", required=True)
    parser.add_argument("--function-qualifier", required=True)
    parser.add_argument("--application-revision", required=True)
    parser.add_argument("--evaluation-suite-revision", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--knowledge-base-id", required=True)
    values = parser.parse_args(arguments)
    identity = DeployedEvaluationIdentity(
        values.function_name,
        values.function_qualifier,
        values.application_revision,
        values.evaluation_suite_revision,
        values.generation_id,
        values.knowledge_base_id,
    )
    try:
        result = run_deployed_case_once(
            values.root,
            values.ledger_root,
            values.journal,
            identity,
            values.case_index,
            AwsLambdaInvoker(create_lambda_client(values.region)),
            clock=lambda: datetime.now(UTC),
        )
    except (DeployedEvaluationError, OSError, RuntimeError, ValueError) as error:
        print(f"deployed case evaluation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "case_id": result.case_id,
                "request_id": result.request_id,
                "result_path": result.result_path.as_posix(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
