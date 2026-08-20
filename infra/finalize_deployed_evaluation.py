from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from infra.continue_deployed_evaluation import _bounded_result
from valkeyrie.deployed_evaluation import (
    DeployedEvaluationError,
    DeployedEvaluationIdentity,
    _canonical_bytes,
    _content_id,
    _real_cases,
    _request_id,
    _sha256,
)

_TIMEOUT_INDICES = {3, 9}
_MAX_RECORD_BYTES = 1024 * 1024


def finalize_deployed_evaluation(
    root: Path,
    ledger_root: Path,
    output_directory: Path,
    identity: DeployedEvaluationIdentity,
) -> tuple[Path, Mapping[str, object]]:
    """Verify the immutable dispatch ledger and emit one offline evidence matrix."""
    project_root = root.resolve(strict=True)
    ledger = ledger_root.resolve(strict=True)
    if ledger.is_symlink():
        raise DeployedEvaluationError("deployed evaluation ledger cannot be a symlink")
    cases = _real_cases(project_root)
    run_identity = {
        "api_version": "valkeyrie.io/deployed-evaluation-run/1",
        "kind": "DeployedEvaluationRun",
        **asdict(identity),
        "case_ids": [case["id"] for case in cases],
    }
    run_id = _content_id("deployed-evaluation-run/1", run_identity)
    attempts = _records_by_index(ledger, "attempts", cases)
    results = _records_by_index(ledger, "results", cases)
    if set(attempts) != set(range(1, 11)) or any(len(values) != 1 for values in attempts.values()):
        raise DeployedEvaluationError("dispatch ledger must contain exactly one attempt per case")
    if set(results) != set(range(1, 11)) - _TIMEOUT_INDICES or any(
        len(values) != 1 for values in results.values()
    ):
        raise DeployedEvaluationError("dispatch ledger result set differs from reviewed evidence")

    timeout_evidence = _validate_timeout_evidence(ledger, attempts)
    observations: list[dict[str, object]] = []
    passed = 0
    for index, case in enumerate(cases, start=1):
        attempt_path, attempt = attempts[index][0]
        request_id = _request_id(run_id, cast(str, case["id"]))
        request = attempt.get("request")
        if not isinstance(request, Mapping):
            raise DeployedEvaluationError("attempt request is malformed")
        expected_request = {
            "action": "answer",
            "request_id": request_id,
            "question": case["question"],
            "version_requirement": ("current_state" if case["expected_external_calls"] else "none"),
            "requested_version": None,
            "knowledge_base_id": identity.knowledge_base_id,
            "owner": f"deployed-evaluation-{run_id[-12:]}",
            "lease_duration_seconds": 300,
            "now": request.get("now"),
            "completed_at": request.get("now"),
        }
        request_hash = _sha256(_canonical_bytes(expected_request, newline=False))
        expected_attempt = {
            "api_version": "valkeyrie.io/deployed-evaluation-attempt/1",
            "kind": "DeployedEvaluationAttempt",
            "run_id": run_id,
            "index": index,
            "case_id": case["id"],
            "request_id": request_id,
            "request_sha256": request_hash,
            "function_name": identity.function_name,
            "function_qualifier": identity.function_qualifier,
            "application_revision": identity.application_revision,
            "evaluation_suite_revision": identity.evaluation_suite_revision,
            "generation_id": identity.generation_id,
            "attempted": True,
            "request": expected_request,
        }
        if attempt != expected_attempt:
            raise DeployedEvaluationError(f"attempt {index} is inconsistent")
        base: dict[str, object] = {
            "index": index,
            "case_id": case["id"],
            "question": case["question"],
            "route": "live" if case["expected_external_calls"] else "static",
            "expected_behavior": case["expected_behavior"],
            "executed_lambda_version": identity.function_qualifier,
            "request_id": request_id,
            "request_sha256": request_hash,
            "attempt_path": attempt_path.relative_to(project_root).as_posix(),
            "attempt_sha256": _sha256(attempt_path.read_bytes()),
            "latency_ms": None,
        }
        if index in _TIMEOUT_INDICES:
            observations.append(
                {
                    **base,
                    "verdict": "fail",
                    "outcome": "timeout",
                    "generation_id": None,
                    "message": (
                        "Immutable Lambda version 7 timed out after 30000 ms "
                        "with zero response bytes."
                    ),
                    "claims": [],
                    "citations": [],
                    "result_path": None,
                    "result_sha256": None,
                    "terminal_evidence": timeout_evidence[index],
                }
            )
            continue

        result_path, result_record = results[index][0]
        response = _bounded_result(result_record.get("response"))
        response_hash = _sha256(_canonical_bytes(response, newline=False))
        expected_result = {
            "api_version": "valkeyrie.io/deployed-evaluation-result/1",
            "kind": "DeployedEvaluationResult",
            "run_id": run_id,
            "case_id": case["id"],
            "request_id": request_id,
            "request_sha256": request_hash,
            "response_sha256": response_hash,
            "response": response,
        }
        if result_record != expected_result:
            raise DeployedEvaluationError(f"result {index} is inconsistent")
        expected_outcome = (
            "abstention" if case["expected_behavior"] == "abstain" else case["expected_behavior"]
        )
        provenance_ok = (
            response["generation_id"] is None
            if case["expected_external_calls"]
            else response["generation_id"] == identity.generation_id
        )
        verdict = "pass" if response["outcome"] == expected_outcome and provenance_ok else "fail"
        if verdict == "pass":
            passed += 1
        observations.append(
            {
                **base,
                "verdict": verdict,
                "outcome": response["outcome"],
                "generation_id": response["generation_id"],
                "message": response["message"],
                "claims": response["claims"],
                "citations": response["citations"],
                "response_sha256": response_hash,
                "request_revision": response["request_revision"],
                "request_fence": response["request_fence"],
                "result_path": result_path.relative_to(project_root).as_posix(),
                "result_sha256": _sha256(result_path.read_bytes()),
            }
        )

    preimage: dict[str, object] = {
        "api_version": "valkeyrie.io/deployed-evaluation-final-evidence/1",
        "kind": "DeployedEvaluationFinalEvidence",
        "logical_run_id": run_id,
        "identity": asdict(identity),
        "evaluation_result": "pass" if passed == 10 else "fail",
        "summary": {
            "case_count": 10,
            "attempt_count": 10,
            "result_count": 8,
            "passed": passed,
            "failed": 10 - passed,
            "timeouts": 2,
            "automatic_retries": 0,
        },
        "observations": observations,
    }
    evidence_id = _content_id("deployed-evaluation-final-evidence/1", preimage)
    manifest = {**preimage, "evidence_id": evidence_id}
    output = output_directory / f"{evidence_id.removeprefix('sha256:')}.json"
    try:
        with output.open("xb") as stream:
            stream.write(_canonical_bytes(manifest))
            stream.flush()
    except FileExistsError as error:
        raise DeployedEvaluationError("refusing to overwrite final deployed evidence") from error
    return output, manifest


def _records_by_index(
    ledger: Path,
    directory_name: str,
    cases: tuple[Mapping[str, object], ...],
) -> dict[int, list[tuple[Path, dict[str, object]]]]:
    case_indices = {case["id"]: index for index, case in enumerate(cases, start=1)}
    records: dict[int, list[tuple[Path, dict[str, object]]]] = {}
    for path in ledger.rglob("*.json"):
        if path.parent.name != directory_name:
            continue
        value = _load_record(path)
        case_id = value.get("case_id")
        index = value.get("index") if directory_name == "attempts" else case_indices.get(case_id)
        if not isinstance(index, int) or isinstance(index, bool) or not 1 <= index <= 10:
            raise DeployedEvaluationError(f"{directory_name} record has an invalid case")
        records.setdefault(index, []).append((path, value))
    return records


def _validate_timeout_evidence(
    ledger: Path,
    attempts: Mapping[int, list[tuple[Path, dict[str, object]]]],
) -> dict[int, dict[str, object]]:
    evidence = ledger / "terminal-evidence"
    cloudwatch_path = evidence / "cloudwatch-version-7.json"
    cloudwatch = _load_record(cloudwatch_path)
    events = cloudwatch.get("events")
    if not isinstance(events, list):
        raise DeployedEvaluationError("CloudWatch timeout evidence is malformed")
    timeout_events: list[dict[str, object]] = []
    for raw in events:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("message"), str):
            continue
        try:
            message = json.loads(cast(str, raw["message"]))
        except json.JSONDecodeError:
            continue
        if (
            isinstance(message, Mapping)
            and message.get("type") == "platform.runtimeDone"
            and isinstance(message.get("record"), Mapping)
            and cast(Mapping[str, object], message["record"]).get("status") == "timeout"
        ):
            timeout_events.append(dict(cast(Mapping[str, object], message)))
    if len(timeout_events) != 2:
        raise DeployedEvaluationError("expected exactly two version-7 timeout events")

    result: dict[int, dict[str, object]] = {}
    available = list(timeout_events)
    for index in sorted(_TIMEOUT_INDICES):
        attempt = attempts[index][0][1]
        request = cast(Mapping[str, object], attempt["request"])
        started = datetime.fromisoformat(cast(str, request["now"]).replace("Z", "+00:00"))
        match_index = -1
        for offset, event in enumerate(available):
            ended = datetime.fromisoformat(cast(str, event["time"]).replace("Z", "+00:00"))
            if 29 <= (ended.astimezone(UTC) - started.astimezone(UTC)).total_seconds() <= 31:
                match_index = offset
                break
        if match_index < 0:
            raise DeployedEvaluationError(f"timeout event does not bind to attempt {index}")
        timeout_event = available.pop(match_index)
        state_path = evidence / f"case-{index:02d}-request-state.json"
        state = _load_record(state_path)
        item = state.get("Item")
        if not isinstance(item, Mapping):
            raise DeployedEvaluationError("timeout request state is missing")
        expected_pk = f"request#{attempt['request_id']}"
        if (
            item.get("pk") != {"S": expected_pk}
            or item.get("revision") != {"N": "1"}
            or item.get("fence") != {"N": "1"}
            or "outcome" in item
            or "completed_at" in item
        ):
            raise DeployedEvaluationError("timeout request state is not incomplete revision 1")
        result[index] = {
            "cloudwatch_path": cloudwatch_path.relative_to(ledger.parent.parent.parent).as_posix(),
            "cloudwatch_sha256": _sha256(cloudwatch_path.read_bytes()),
            "platform_event": timeout_event,
            "request_state_path": state_path.relative_to(ledger.parent.parent.parent).as_posix(),
            "request_state_sha256": _sha256(state_path.read_bytes()),
            "request_state": "revision_1_without_outcome",
        }
    return result


def _load_record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    if not 1 <= len(content) <= _MAX_RECORD_BYTES:
        raise DeployedEvaluationError("deployed evidence record exceeds its byte bound")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise DeployedEvaluationError("deployed evidence record is not strict JSON") from error
    if not isinstance(value, Mapping):
        raise DeployedEvaluationError("deployed evidence record root is malformed")
    return dict(cast(Mapping[str, object], value))


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Finalize the exact-once deployed evidence offline."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
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
        output, manifest = finalize_deployed_evaluation(
            values.root, values.ledger_root, values.output_directory, identity
        )
    except (DeployedEvaluationError, OSError, ValueError) as error:
        print(f"deployed evaluation finalization failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "evidence_id": manifest["evidence_id"],
                "evaluation_result": manifest["evaluation_result"],
                "output": output.as_posix(),
                "summary": manifest["summary"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
