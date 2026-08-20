from __future__ import annotations

import argparse
import base64
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from infra.run_comparison_case import (
    ComparisonEvaluationIdentity,
    _add_identity_arguments,
    _identity_from_arguments,
    _validate_response_shape,
    comparison_run_identity,
)
from valkeyrie.deployed_evaluation import (
    DeployedEvaluationError,
    _bounded_result,
    _canonical_bytes,
    _content_id,
    _real_cases,
    _request_id,
    _sha256,
    _write_exclusive,
)

_BASELINE_EVIDENCE_ID = "sha256:2fe6adb72c33008ab8a6b9ea07d251b3937de87dc14fcbe60762629eab83d547"
_BASELINE_FILE_SHA256 = "sha256:f715a8a324a0f9be10a7da55c8a78f81e835e374dcaf43d9ab3e5e80f0af202f"
_BASELINE_PATH = Path(
    "evals/deployed-reports/2fe6adb72c33008ab8a6b9ea07d251b3937de87dc14fcbe60762629eab83d547.json"
)
_MAX_RECORD_BYTES = 1024 * 1024


def finalize_comparison_evaluation(
    root: Path,
    ledger_root: Path,
    output_directory: Path,
    identity: ComparisonEvaluationIdentity,
) -> tuple[Path, Mapping[str, object]]:
    """Verify a complete exact-once Opus ledger and emit content-addressed evidence."""
    project_root = root.resolve(strict=True)
    if ledger_root.is_symlink() or output_directory.is_symlink():
        raise DeployedEvaluationError("comparison evidence directories cannot be symlinks")
    ledger = ledger_root.resolve(strict=True)
    output = output_directory.resolve(strict=True)
    cases = _real_cases(project_root)
    _, run_id = comparison_run_identity(identity, cases)
    expected_ledger = project_root / "evals/comparison-attempts" / run_id.removeprefix("sha256:")
    if ledger != expected_ledger.resolve(strict=True):
        raise DeployedEvaluationError("comparison ledger is not the canonical run directory")
    if output != (project_root / "evals/comparison-reports").resolve(strict=True):
        raise DeployedEvaluationError("comparison report directory is not canonical")
    attempts = _records_by_index(ledger, "attempts", cases)
    results = _records_by_index(ledger, "results", cases)
    terminals = _records_by_index(ledger, "terminal", cases)
    expected_indices = set(range(1, 11))
    if set(attempts) != expected_indices or any(len(values) != 1 for values in attempts.values()):
        raise DeployedEvaluationError("comparison ledger must contain one attempt per case")
    completed_indices = set(results) | set(terminals)
    if completed_indices != expected_indices or set(results) & set(terminals):
        raise DeployedEvaluationError(
            "comparison cases require exactly one result or terminal record"
        )
    if any(len(values) != 1 for values in (*results.values(), *terminals.values())):
        raise DeployedEvaluationError("comparison ledger contains duplicate terminal records")

    observations: list[dict[str, object]] = []
    latencies: list[float] = []
    protocol_passed = 0
    timeout_count = 0
    for index, case in enumerate(cases, start=1):
        attempt_path, attempt = attempts[index][0]
        request_id = _request_id(run_id, cast(str, case["id"]))
        request = attempt.get("request")
        if not isinstance(request, Mapping):
            raise DeployedEvaluationError(f"comparison attempt {index} request is malformed")
        timestamp = request.get("now")
        expected_request = {
            "action": "answer",
            "request_id": request_id,
            "question": case["question"],
            "version_requirement": "current_state" if case["expected_external_calls"] else "none",
            "requested_version": None,
            "knowledge_base_id": identity.knowledge_base_id,
            "owner": f"comparison-evaluation-{run_id[-12:]}",
            "lease_duration_seconds": 300,
            "now": timestamp,
            "completed_at": timestamp,
        }
        request_hash = _sha256(_canonical_bytes(expected_request, newline=False))
        preflight_path = attempt_path.parent.parent / "preflight.json"
        preflight = _load_record(preflight_path)
        _validate_preflight(preflight, identity, run_id)
        preflight_sha256 = _sha256(preflight_path.read_bytes())
        expected_attempt = {
            "api_version": "valkeyrie.io/comparison-evaluation-attempt/1",
            "kind": "ComparisonEvaluationAttempt",
            "run_id": run_id,
            "identity": asdict(identity),
            "index": index,
            "case_id": case["id"],
            "request_id": request_id,
            "request_sha256": request_hash,
            "preflight_path": "preflight.json",
            "preflight_sha256": preflight_sha256,
            "attempted": True,
            "request": expected_request,
        }
        if attempt != expected_attempt:
            raise DeployedEvaluationError(f"comparison attempt {index} is inconsistent")
        base: dict[str, object] = {
            "index": index,
            "case_id": case["id"],
            "question": case["question"],
            "route": "live" if case["expected_external_calls"] else "static",
            "expected_behavior": case["expected_behavior"],
            "executed_lambda_version": identity.function_qualifier,
            "request_id": request_id,
            "request_sha256": request_hash,
            "attempt_path": attempt_path.relative_to(ledger).as_posix(),
            "attempt_sha256": _sha256(attempt_path.read_bytes()),
            "preflight_path": preflight_path.relative_to(ledger).as_posix(),
            "preflight_sha256": preflight_sha256,
        }
        if index in terminals:
            terminal_path, terminal = terminals[index][0]
            latency = _validate_terminal(
                terminal,
                identity,
                run_id,
                index,
                cast(str, case["id"]),
                request_id,
                request_hash,
            )
            latencies.append(latency)
            failure_class = cast(str, terminal["failure_class"])
            timeout_count += int(failure_class == "timeout")
            observations.append(
                {
                    **base,
                    "protocol_verdict": "fail",
                    "semantic_verdict": "not_evaluated",
                    "outcome": failure_class,
                    "generation_id": None,
                    "message": terminal["message"],
                    "claims": [],
                    "citations": [],
                    "latency_ms": latency,
                    "result_path": None,
                    "result_sha256": None,
                    "terminal_path": terminal_path.relative_to(ledger).as_posix(),
                    "terminal_sha256": _sha256(terminal_path.read_bytes()),
                    "terminal": terminal,
                }
            )
            continue

        result_path, result_record = results[index][0]
        response = _bounded_result(result_record.get("response"))
        _validate_response_shape(response, request_id)
        response_hash = _sha256(_canonical_bytes(response, newline=False))
        latency = _latency(result_record.get("latency_ms"))
        expected_result = {
            "api_version": "valkeyrie.io/comparison-evaluation-result/1",
            "kind": "ComparisonEvaluationResult",
            "run_id": run_id,
            "identity": asdict(identity),
            "index": index,
            "case_id": case["id"],
            "request_id": request_id,
            "request_sha256": request_hash,
            "response_sha256": response_hash,
            "latency_ms": latency,
            "response": response,
        }
        if result_record != expected_result:
            raise DeployedEvaluationError(f"comparison result {index} is inconsistent")
        latencies.append(latency)
        expected_outcome = (
            "abstention" if case["expected_behavior"] == "abstain" else case["expected_behavior"]
        )
        provenance_ok = (
            response["generation_id"] is None
            if case["expected_external_calls"]
            else response["generation_id"] == identity.generation_id
        )
        protocol_verdict = (
            "pass" if _protocol_conforms(response, expected_outcome, provenance_ok) else "fail"
        )
        protocol_passed += int(protocol_verdict == "pass")
        observations.append(
            {
                **base,
                "protocol_verdict": protocol_verdict,
                "semantic_verdict": "not_evaluated",
                "outcome": response["outcome"],
                "generation_id": response["generation_id"],
                "message": response["message"],
                "claims": response["claims"],
                "citations": response["citations"],
                "request_revision": response["request_revision"],
                "request_fence": response["request_fence"],
                "response_sha256": response_hash,
                "latency_ms": latency,
                "result_path": result_path.relative_to(ledger).as_posix(),
                "result_sha256": _sha256(result_path.read_bytes()),
                "terminal_path": None,
                "terminal_sha256": None,
            }
        )

    baseline = _baseline(project_root, cases)
    preimage: dict[str, object] = {
        "api_version": "valkeyrie.io/comparison-evaluation-final-evidence/1",
        "kind": "ComparisonEvaluationFinalEvidence",
        "logical_run_id": run_id,
        "identity": asdict(identity),
        "evaluation_result": "evidence_collected_pending_semantic_review",
        "summary": {
            "case_count": 10,
            "attempt_count": 10,
            "result_count": len(results),
            "terminal_count": len(terminals),
            "protocol_passed": protocol_passed,
            "protocol_failed": 10 - protocol_passed,
            "semantic_review_status": "not_evaluated",
            "timeouts": timeout_count,
            "automatic_retries": 0,
            "client_observed_latency_ms": _latency_summary(latencies),
        },
        "baseline_fable": baseline,
        "observations": observations,
    }
    evidence_id = _content_id("comparison-evaluation-final-evidence/1", preimage)
    manifest = {**preimage, "evidence_id": evidence_id}
    path = output / f"{evidence_id.removeprefix('sha256:')}.json"
    _write_exclusive(path, _canonical_bytes(manifest))
    return path, manifest


def _records_by_index(
    ledger: Path,
    directory_name: str,
    cases: tuple[Mapping[str, object], ...],
) -> dict[int, list[tuple[Path, dict[str, object]]]]:
    indices = {case["id"]: index for index, case in enumerate(cases, start=1)}
    records: dict[int, list[tuple[Path, dict[str, object]]]] = {}
    for path in ledger.rglob("*.json"):
        if path.parent.name != directory_name:
            continue
        if path.is_symlink():
            raise DeployedEvaluationError("comparison evidence record cannot be a symlink")
        value = _load_record(path)
        index = value.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            index = indices.get(value.get("case_id"))
        if not isinstance(index, int) or not 1 <= index <= 10:
            raise DeployedEvaluationError(f"comparison {directory_name} record has invalid case")
        records.setdefault(index, []).append((path, value))
    return records


def _validate_preflight(
    value: Mapping[str, object], identity: ComparisonEvaluationIdentity, run_id: str
) -> None:
    if set(value) != {"api_version", "kind", "run_id", "identity", "proof"}:
        raise DeployedEvaluationError("comparison preflight fields are inconsistent")
    if (
        value.get("api_version") != "valkeyrie.io/comparison-evaluation-preflight/1"
        or value.get("kind") != "ComparisonEvaluationPreflight"
        or value.get("run_id") != run_id
        or value.get("identity") != asdict(identity)
    ):
        raise DeployedEvaluationError("comparison preflight identity is inconsistent")
    proof = value.get("proof")
    if not isinstance(proof, Mapping) or set(proof) != {"configuration", "health"}:
        raise DeployedEvaluationError("comparison preflight proof is malformed")
    configuration = proof.get("configuration")
    health = proof.get("health")
    if not isinstance(configuration, Mapping) or not isinstance(health, Mapping):
        raise DeployedEvaluationError("comparison preflight proof values are malformed")
    expected_configuration = {
        "FunctionName": identity.function_name,
        "Version": identity.function_qualifier,
        "CodeSha256": base64.b64encode(bytes.fromhex(identity.artifact_sha256[7:])).decode(),
        "Timeout": 240,
        "APPLICATION_REVISION": identity.application_revision,
        "SELECTED_INFERENCE_PROFILE_ARN": identity.selected_inference_profile_arn,
    }
    if dict(configuration) != expected_configuration:
        raise DeployedEvaluationError("comparison preflight configuration is inconsistent")
    expected_health = {
        "status": "healthy",
        "application_revision": identity.application_revision,
        "selection_id": identity.selection_id,
        "selected_model_revision": identity.selected_model_revision,
        "selected_inference_profile_arn": identity.selected_inference_profile_arn,
        "execution_authorization": identity.execution_authorization,
        "qualification_status": identity.qualification_status,
    }
    if any(health.get(field) != item for field, item in expected_health.items()):
        raise DeployedEvaluationError("comparison preflight health identity is inconsistent")


def _validate_terminal(
    value: Mapping[str, object],
    identity: ComparisonEvaluationIdentity,
    run_id: str,
    index: int,
    case_id: str,
    request_id: str,
    request_hash: str,
) -> float:
    expected_fields = {
        "api_version",
        "kind",
        "run_id",
        "identity",
        "index",
        "case_id",
        "request_id",
        "request_sha256",
        "latency_ms",
        "failure_class",
        "error_type",
        "message",
        "retry_permitted",
    }
    if set(value) != expected_fields:
        raise DeployedEvaluationError("comparison terminal record fields are inconsistent")
    expected = {
        "api_version": "valkeyrie.io/comparison-evaluation-terminal/1",
        "kind": "ComparisonEvaluationTerminal",
        "run_id": run_id,
        "identity": asdict(identity),
        "index": index,
        "case_id": case_id,
        "request_id": request_id,
        "request_sha256": request_hash,
        "retry_permitted": False,
    }
    if any(value.get(field) != item for field, item in expected.items()):
        raise DeployedEvaluationError("comparison terminal record identity is inconsistent")
    if value.get("failure_class") not in {"timeout", "invocation_error", "invalid_response"}:
        raise DeployedEvaluationError("comparison terminal failure class is invalid")
    for field, maximum in (("error_type", 128), ("message", 512)):
        item = value.get(field)
        if not isinstance(item, str) or not item or len(item.encode("utf-8")) > maximum:
            raise DeployedEvaluationError(f"comparison terminal {field} is invalid")
    return _latency(value.get("latency_ms"))


def _protocol_conforms(
    response: Mapping[str, object], expected_outcome: object, provenance_ok: bool
) -> bool:
    if response.get("outcome") != expected_outcome or not provenance_ok:
        return False
    if expected_outcome == "answer":
        claims = response.get("claims")
        citations = response.get("citations")
        if (
            not isinstance(claims, list)
            or not claims
            or not isinstance(citations, list)
            or not citations
            or not all(isinstance(item, str) and item.strip() for item in citations)
        ):
            return False
        for claim in claims:
            if not isinstance(claim, Mapping):
                return False
            if not isinstance(claim.get("claim_id"), str) or not claim["claim_id"]:
                return False
            if not isinstance(claim.get("text"), str) or not claim["text"]:
                return False
            evidence_ids = claim.get("evidence_ids")
            if (
                not isinstance(evidence_ids, list)
                or not evidence_ids
                or not all(isinstance(item, str) and item for item in evidence_ids)
            ):
                return False
        return True
    message = response.get("message")
    return isinstance(message, str) and bool(message.strip())


def _latency(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DeployedEvaluationError("comparison latency is not numeric")
    latency = float(value)
    if not math.isfinite(latency) or latency < 0:
        raise DeployedEvaluationError("comparison latency is invalid")
    return latency


def _latency_summary(values: list[float]) -> dict[str, object]:
    if len(values) != 10:
        raise DeployedEvaluationError("comparison latency set is incomplete")
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean": round(statistics.fmean(ordered), 6),
        "median": round(statistics.median(ordered), 6),
        "p95_nearest_rank": ordered[9],
        "measurement": "client_monotonic_single_lambda_invoke",
    }


def _baseline(root: Path, cases: tuple[Mapping[str, object], ...]) -> dict[str, object]:
    path = root / _BASELINE_PATH
    value = _load_record(path)
    if _sha256(path.read_bytes()) != _BASELINE_FILE_SHA256:
        raise DeployedEvaluationError("Fable baseline file hash is inconsistent")
    evidence_id = value.get("evidence_id")
    preimage = dict(value)
    preimage.pop("evidence_id", None)
    if (
        evidence_id != _BASELINE_EVIDENCE_ID
        or _content_id("deployed-evaluation-final-evidence/1", preimage) != _BASELINE_EVIDENCE_ID
    ):
        raise DeployedEvaluationError("Fable baseline evidence identity is inconsistent")
    summary = value.get("summary")
    observations = value.get("observations")
    expected_summary = {
        "case_count": 10,
        "attempt_count": 10,
        "result_count": 8,
        "passed": 6,
        "failed": 4,
        "timeouts": 2,
        "automatic_retries": 0,
    }
    if summary != expected_summary or not isinstance(observations, list):
        raise DeployedEvaluationError("Fable baseline evidence is malformed")
    expected_case_ids = [case["id"] for case in cases]
    if (
        len(observations) != 10
        or [item.get("case_id") if isinstance(item, Mapping) else None for item in observations]
        != expected_case_ids
    ):
        raise DeployedEvaluationError("Fable baseline case matrix is inconsistent")
    matrix: list[dict[str, object]] = []
    for index, item in enumerate(observations, start=1):
        if (
            not isinstance(item, Mapping)
            or item.get("index") != index
            or item.get("verdict") not in {"pass", "fail"}
            or not isinstance(item.get("outcome"), str)
        ):
            raise DeployedEvaluationError("Fable baseline observation is malformed")
        matrix.append(
            {
                "index": item["index"],
                "case_id": item["case_id"],
                "verdict": item["verdict"],
                "outcome": item["outcome"],
                "latency_ms": item.get("latency_ms"),
            }
        )
    return {
        "evidence_id": _BASELINE_EVIDENCE_ID,
        "path": _BASELINE_PATH.as_posix(),
        "file_sha256": _BASELINE_FILE_SHA256,
        "summary": expected_summary,
        "latency_measurement_available": any(item["latency_ms"] is not None for item in matrix),
        "observations": matrix,
    }


def _load_record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    if not 1 <= len(content) <= _MAX_RECORD_BYTES:
        raise DeployedEvaluationError("comparison record exceeds its byte bound")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = item
        return result

    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise DeployedEvaluationError("comparison record is not strict JSON") from error
    if not isinstance(value, Mapping):
        raise DeployedEvaluationError("comparison record root is malformed")
    return dict(cast(Mapping[str, object], value))


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Finalize a complete exact-once Opus comparison ledger offline."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    _add_identity_arguments(parser)
    values = parser.parse_args(arguments)
    try:
        path, manifest = finalize_comparison_evaluation(
            values.root,
            values.ledger_root,
            values.output_directory,
            _identity_from_arguments(values),
        )
    except (DeployedEvaluationError, OSError, ValueError) as error:
        print(f"comparison finalization failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "evidence_id": manifest["evidence_id"],
                "path": path.as_posix(),
                "summary": manifest["summary"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
