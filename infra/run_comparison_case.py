from __future__ import annotations

import argparse
import base64
import json
import math
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from valkeyrie.aws_adapters import AwsLambdaInvoker, create_lambda_client
from valkeyrie.deployed_evaluation import (
    DeployedEvaluationError,
    DeployedEvaluationIdentity,
    DeployedInvocation,
    DeployedInvoker,
    _bounded_result,
    _canonical_bytes,
    _content_id,
    _fsync_directory,
    _real_cases,
    _request_id,
    _sha256,
    _timestamp,
    _validate_inputs,
    _write_exclusive,
)

_AUTHORIZATION = "owner_directed_comparison"
_MODEL_REVISION = "us.anthropic.claude-opus-5"
_PROFILE_ARN = "arn:aws:bedrock:us-east-1:968533178160:inference-profile/us.anthropic.claude-opus-5"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_FAILURE_MESSAGE = 512
FailureClass = Literal["timeout", "invocation_error", "invalid_response"]


@dataclass(frozen=True)
class ComparisonEvaluationIdentity:
    function_name: str
    function_qualifier: str
    application_revision: str
    artifact_sha256: str
    evaluation_suite_revision: str
    generation_id: str
    knowledge_base_id: str
    selected_model_revision: str
    selected_inference_profile_arn: str
    selection_id: str
    execution_authorization: str
    qualification_status: str

    def deployment(self) -> DeployedEvaluationIdentity:
        return DeployedEvaluationIdentity(
            self.function_name,
            self.function_qualifier,
            self.application_revision,
            self.evaluation_suite_revision,
            self.generation_id,
            self.knowledge_base_id,
        )


@dataclass(frozen=True)
class ComparisonCaseResult:
    case_id: str
    request_id: str
    journal_directory: Path
    latency_ms: float
    response: Mapping[str, object] | None
    terminal: Mapping[str, object] | None


class ComparisonLambdaClient(Protocol):
    def get_function(self, **kwargs: object) -> Mapping[str, object]: ...

    def invoke(self, **kwargs: object) -> Mapping[str, object]: ...


class ComparisonPreflight(Protocol):
    def verify(self, identity: ComparisonEvaluationIdentity) -> Mapping[str, object]: ...


class AwsComparisonPreflight:
    def __init__(self, client: ComparisonLambdaClient) -> None:
        self._client = client

    def verify(self, identity: ComparisonEvaluationIdentity) -> Mapping[str, object]:
        response = self._client.get_function(
            FunctionName=identity.function_name,
            Qualifier=identity.function_qualifier,
        )
        configuration = response.get("Configuration")
        if not isinstance(configuration, Mapping):
            raise DeployedEvaluationError("Lambda preflight configuration is malformed")
        environment = configuration.get("Environment")
        variables = environment.get("Variables") if isinstance(environment, Mapping) else None
        if not isinstance(variables, Mapping):
            raise DeployedEvaluationError("Lambda preflight environment is malformed")
        expected_code = base64.b64encode(bytes.fromhex(identity.artifact_sha256[7:])).decode()
        expected_configuration = {
            "FunctionName": identity.function_name,
            "Version": identity.function_qualifier,
            "CodeSha256": expected_code,
            "Timeout": 240,
            "APPLICATION_REVISION": identity.application_revision,
            "SELECTED_INFERENCE_PROFILE_ARN": identity.selected_inference_profile_arn,
        }
        observed_configuration = {
            "FunctionName": configuration.get("FunctionName"),
            "Version": configuration.get("Version"),
            "CodeSha256": configuration.get("CodeSha256"),
            "Timeout": configuration.get("Timeout"),
            "APPLICATION_REVISION": variables.get("APPLICATION_REVISION"),
            "SELECTED_INFERENCE_PROFILE_ARN": variables.get("SELECTED_INFERENCE_PROFILE_ARN"),
        }
        if observed_configuration != expected_configuration:
            raise DeployedEvaluationError("Lambda preflight configuration differs from identity")
        health_invocation = DeployedInvocation(
            identity.function_name,
            identity.function_qualifier,
            identity.application_revision,
            identity.evaluation_suite_revision,
            identity.generation_id,
            "req_comparison-preflight",
            _canonical_bytes(
                {
                    "action": "health",
                    "application_revision": identity.application_revision,
                },
                newline=False,
            ),
        )
        health = dict(AwsLambdaInvoker(self._client).invoke(health_invocation))
        expected_health = {
            "status": "healthy",
            "application_revision": identity.application_revision,
            "selection_id": identity.selection_id,
            "selected_model_revision": identity.selected_model_revision,
            "selected_inference_profile_arn": identity.selected_inference_profile_arn,
            "execution_authorization": identity.execution_authorization,
            "qualification_status": identity.qualification_status,
        }
        if any(health.get(field) != value for field, value in expected_health.items()):
            raise DeployedEvaluationError("Lambda health identity differs from comparison identity")
        return {
            "configuration": observed_configuration,
            "health": health,
        }


def comparison_run_identity(
    identity: ComparisonEvaluationIdentity, cases: tuple[Mapping[str, object], ...]
) -> tuple[dict[str, object], str]:
    value: dict[str, object] = {
        "api_version": "valkeyrie.io/comparison-evaluation-run/1",
        "kind": "ComparisonEvaluationRun",
        "identity": asdict(identity),
        "case_ids": [case["id"] for case in cases],
    }
    return value, _content_id("comparison-evaluation-run/1", value)


def run_comparison_case_once(
    root: Path,
    ledger_root: Path,
    journal_directory: Path,
    identity: ComparisonEvaluationIdentity,
    case_index: int,
    invoker: DeployedInvoker,
    preflight: ComparisonPreflight,
    *,
    clock: Callable[[], datetime],
    monotonic_ns: Callable[[], int],
) -> ComparisonCaseResult:
    """Journal and invoke one Opus comparison case exactly once."""
    _validate_comparison_inputs(root, journal_directory, identity, invoker, preflight)
    project_root = root.resolve(strict=True)
    cases = _real_cases(project_root)
    if not isinstance(case_index, int) or isinstance(case_index, bool) or not 1 <= case_index <= 10:
        raise DeployedEvaluationError("comparison case index is outside the reviewed suite")
    case = cases[case_index - 1]
    case_id = cast(str, case["id"])
    stem = f"{case_index:02d}-{case_id}"
    run_identity, run_id = comparison_run_identity(identity, cases)
    expected_ledger = project_root / "evals/comparison-attempts" / run_id.removeprefix("sha256:")
    ledger = ledger_root.resolve(strict=True)
    if ledger != expected_ledger.resolve(strict=True):
        raise DeployedEvaluationError("comparison ledger is not the canonical run directory")
    expected_journal = ledger / "cases" / stem
    if journal_directory.parent.resolve(strict=True) / journal_directory.name != expected_journal:
        raise DeployedEvaluationError("comparison journal is not the canonical case claim")
    request_id = _request_id(run_id, case_id)
    _assert_unattempted(ledger, case_index, case_id, request_id)
    proof = dict(preflight.verify(identity))
    proof_bytes = _canonical_bytes(proof, newline=False)
    if not proof or len(proof_bytes) > 1024 * 1024:
        raise DeployedEvaluationError("comparison deployment preflight proof is malformed")

    try:
        journal_directory.mkdir(mode=0o700)
        _fsync_directory(journal_directory.parent)
    except FileExistsError as error:
        raise DeployedEvaluationError("refusing to resume or repeat a comparison case") from error
    except OSError as error:
        raise DeployedEvaluationError("cannot create comparison case journal") from error
    attempts = journal_directory / "attempts"
    results = journal_directory / "results"
    terminal = journal_directory / "terminal"
    attempts.mkdir(mode=0o700)
    results.mkdir(mode=0o700)
    terminal.mkdir(mode=0o700)
    _fsync_directory(journal_directory)
    preflight_record = {
        "api_version": "valkeyrie.io/comparison-evaluation-preflight/1",
        "kind": "ComparisonEvaluationPreflight",
        "run_id": run_id,
        "identity": asdict(identity),
        "proof": proof,
    }
    preflight_path = journal_directory / "preflight.json"
    _write_exclusive(preflight_path, _canonical_bytes(preflight_record))
    preflight_sha256 = _sha256(preflight_path.read_bytes())

    _write_exclusive(
        journal_directory / "case.json",
        _canonical_bytes(
            {
                **run_identity,
                "logical_run_id": run_id,
                "index": case_index,
                "case_id": case_id,
                "request_id": request_id,
            }
        ),
    )
    timestamp = _timestamp(clock())
    event = {
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
    payload = _canonical_bytes(event, newline=False)
    request_hash = _sha256(payload)
    attempt_path = attempts / f"{stem}.json"
    _write_exclusive(
        attempt_path,
        _canonical_bytes(
            {
                "api_version": "valkeyrie.io/comparison-evaluation-attempt/1",
                "kind": "ComparisonEvaluationAttempt",
                "run_id": run_id,
                "identity": asdict(identity),
                "index": case_index,
                "case_id": case_id,
                "request_id": request_id,
                "request_sha256": request_hash,
                "preflight_path": "preflight.json",
                "preflight_sha256": preflight_sha256,
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
    started: int | None = None
    try:
        started = _monotonic_value(monotonic_ns())
        raw_result = invoker.invoke(invocation)
        finished = _monotonic_value(monotonic_ns())
        latency_ms = _latency_ms(started, finished)
    except Exception as error:
        latency_ms = 0.0
        if started is not None:
            try:
                finished = _monotonic_value(monotonic_ns())
                latency_ms = _latency_ms(started, finished)
            except Exception:
                latency_ms = 0.0
        record = _terminal_record(
            run_id,
            identity,
            case_index,
            case_id,
            request_id,
            request_hash,
            latency_ms,
            _failure_class(error),
            error,
        )
        _write_exclusive(terminal / f"{stem}.json", _canonical_bytes(record))
        return ComparisonCaseResult(
            case_id, request_id, journal_directory, latency_ms, None, record
        )

    try:
        response = _bounded_result(raw_result)
        _validate_response_shape(response, request_id)
    except Exception as error:
        record = _terminal_record(
            run_id,
            identity,
            case_index,
            case_id,
            request_id,
            request_hash,
            latency_ms,
            "invalid_response",
            error,
        )
        _write_exclusive(terminal / f"{stem}.json", _canonical_bytes(record))
        return ComparisonCaseResult(
            case_id, request_id, journal_directory, latency_ms, None, record
        )

    response_hash = _sha256(_canonical_bytes(response, newline=False))
    record = {
        "api_version": "valkeyrie.io/comparison-evaluation-result/1",
        "kind": "ComparisonEvaluationResult",
        "run_id": run_id,
        "identity": asdict(identity),
        "index": case_index,
        "case_id": case_id,
        "request_id": request_id,
        "request_sha256": request_hash,
        "response_sha256": response_hash,
        "latency_ms": latency_ms,
        "response": response,
    }
    _write_exclusive(results / f"{stem}.json", _canonical_bytes(record))
    return ComparisonCaseResult(case_id, request_id, journal_directory, latency_ms, response, None)


def _validate_comparison_inputs(
    root: Path,
    journal_directory: Path,
    identity: ComparisonEvaluationIdentity,
    invoker: DeployedInvoker,
    preflight: ComparisonPreflight,
) -> None:
    if not isinstance(identity, ComparisonEvaluationIdentity):
        raise DeployedEvaluationError("comparison identity is malformed")
    _validate_inputs(root, journal_directory, identity.deployment(), invoker)
    for value, label in (
        (identity.artifact_sha256, "comparison artifact"),
        (identity.selection_id, "comparison selection"),
    ):
        if _DIGEST.fullmatch(value) is None:
            raise DeployedEvaluationError(f"{label} identity is malformed")
    if not callable(getattr(preflight, "verify", None)):
        raise DeployedEvaluationError("comparison preflight is malformed")
    if identity.selected_model_revision != _MODEL_REVISION:
        raise DeployedEvaluationError("comparison model must be exact Claude Opus 5")
    if identity.selected_inference_profile_arn != _PROFILE_ARN:
        raise DeployedEvaluationError("comparison inference profile is incompatible")
    if identity.execution_authorization != _AUTHORIZATION:
        raise DeployedEvaluationError("comparison execution authorization is incompatible")
    if identity.qualification_status != "not_run_not_qualified":
        raise DeployedEvaluationError("comparison qualification status is incompatible")


def _monotonic_value(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DeployedEvaluationError("monotonic clock returned an invalid value")
    return value


def _latency_ms(started: int, finished: int) -> float:
    if finished < started:
        raise DeployedEvaluationError("monotonic clock moved backwards")
    latency = round((finished - started) / 1_000_000, 6)
    if not math.isfinite(latency) or latency < 0:
        raise DeployedEvaluationError("client-observed latency is invalid")
    return latency


def _failure_class(error: Exception) -> FailureClass:
    timeout_types = {"ReadTimeoutError", "ConnectTimeoutError"}
    return (
        "timeout"
        if isinstance(error, TimeoutError) or type(error).__name__ in timeout_types
        else "invocation_error"
    )


def _terminal_record(
    run_id: str,
    identity: ComparisonEvaluationIdentity,
    index: int,
    case_id: str,
    request_id: str,
    request_hash: str,
    latency_ms: float,
    failure_class: FailureClass,
    error: Exception,
) -> dict[str, object]:
    message = " ".join(str(error).split())
    if not message:
        message = "no exception message"
    message = message.encode("utf-8", errors="replace")[:_MAX_FAILURE_MESSAGE].decode(
        "utf-8", errors="ignore"
    )
    return {
        "api_version": "valkeyrie.io/comparison-evaluation-terminal/1",
        "kind": "ComparisonEvaluationTerminal",
        "run_id": run_id,
        "identity": asdict(identity),
        "index": index,
        "case_id": case_id,
        "request_id": request_id,
        "request_sha256": request_hash,
        "latency_ms": latency_ms,
        "failure_class": failure_class,
        "error_type": type(error).__name__[:128],
        "message": message,
        "retry_permitted": False,
    }


def _validate_response_shape(response: Mapping[str, object], request_id: str) -> None:
    if response.get("request_id") != request_id:
        raise DeployedEvaluationError("comparison response request identity is inconsistent")
    if response.get("outcome") not in {
        "answer",
        "clarification",
        "abstention",
        "partial",
        "error",
    }:
        raise DeployedEvaluationError("comparison response outcome is unsupported")
    message = response.get("message")
    if message is not None and not isinstance(message, str):
        raise DeployedEvaluationError("comparison response message is malformed")
    for field in ("claims", "citations"):
        if not isinstance(response.get(field), list):
            raise DeployedEvaluationError(f"comparison response {field} is malformed")
    generation = response.get("generation_id")
    if generation is not None and (
        not isinstance(generation, str) or _DIGEST.fullmatch(generation) is None
    ):
        raise DeployedEvaluationError("comparison response generation identity is malformed")
    for field in ("request_revision", "request_fence"):
        value = response.get(field)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            raise DeployedEvaluationError(f"comparison response {field} is malformed")


def _identity_from_arguments(values: argparse.Namespace) -> ComparisonEvaluationIdentity:
    return ComparisonEvaluationIdentity(
        values.function_name,
        values.function_qualifier,
        values.application_revision,
        values.artifact_sha256,
        values.evaluation_suite_revision,
        values.generation_id,
        values.knowledge_base_id,
        values.selected_model_revision,
        values.selected_inference_profile_arn,
        values.selection_id,
        values.execution_authorization,
        values.qualification_status,
    )


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--function-name", required=True)
    parser.add_argument("--function-qualifier", required=True)
    parser.add_argument("--application-revision", required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--evaluation-suite-revision", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--selected-model-revision", required=True)
    parser.add_argument("--selected-inference-profile-arn", required=True)
    parser.add_argument("--selection-id", required=True)
    parser.add_argument("--execution-authorization", required=True)
    parser.add_argument("--qualification-status", required=True)


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Invoke one never-attempted Opus comparison case exactly once."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--region", required=True)
    _add_identity_arguments(parser)
    values = parser.parse_args(arguments)
    try:
        client = cast(ComparisonLambdaClient, create_lambda_client(values.region))
        result = run_comparison_case_once(
            values.root,
            values.ledger_root,
            values.journal,
            _identity_from_arguments(values),
            values.case_index,
            AwsLambdaInvoker(client),
            AwsComparisonPreflight(client),
            clock=lambda: datetime.now(UTC),
            monotonic_ns=time.monotonic_ns,
        )
    except (DeployedEvaluationError, OSError, RuntimeError, ValueError) as error:
        print(f"comparison case preflight failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "case_id": result.case_id,
                "request_id": result.request_id,
                "latency_ms": result.latency_ms,
                "record_type": "result" if result.response is not None else "terminal",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Moved here from infra/run_deployed_case.py when the deployed-evaluation tooling was
# retired. This comparison runner is its only remaining caller.
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
