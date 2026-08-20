"""Crash-safe one-shot evaluation of the ten reviewed real-world questions."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol, cast

from valkeyrie.evaluations import load_evaluation_suite
from valkeyrie.sources import load_yaml_mapping


class DeployedEvaluationError(ValueError):
    """A deployed evaluation input, journal, invocation, or result is unsafe."""


@dataclass(frozen=True)
class DeployedEvaluationIdentity:
    """Exact immutable deployment and corpus identities under evaluation."""

    function_name: str
    function_qualifier: str
    application_revision: str
    evaluation_suite_revision: str
    generation_id: str
    knowledge_base_id: str


@dataclass(frozen=True)
class DeployedInvocation:
    """One exact payload and immutable Lambda target supplied to an injected invoker."""

    function_name: str
    function_qualifier: str
    application_revision: str
    evaluation_suite_revision: str
    generation_id: str
    request_id: str
    payload: bytes


@dataclass(frozen=True)
class DeployedEvaluationResult:
    """The immutable content-addressed evidence manifest produced by a complete run."""

    run_id: str
    evidence_id: str
    journal_directory: Path
    manifest_path: Path
    manifest: Mapping[str, object]


class DeployedInvoker(Protocol):
    """One-call boundary implemented by an operator-owned exact Lambda adapter."""

    def invoke(self, invocation: DeployedInvocation) -> Mapping[str, object]: ...


_REAL_CASE_IDS: Final = (
    "real-governance",
    "real-getting-started",
    "real-tsc-process",
    "real-upcoming-events",
    "real-contributor-onboarding",
    "real-workstream-status",
    "real-named-person",
    "real-recent-community-meeting",
    "real-replication-failover",
    "real-leaderboard",
)
_DIGEST: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_FUNCTION: Final = re.compile(r"^[A-Za-z0-9-_]{1,64}$")
_QUALIFIER: Final = re.compile(r"^[1-9][0-9]*$")
_KNOWLEDGE_BASE: Final = re.compile(r"^[A-Z0-9]{10}$")
_REQUEST_ID: Final = re.compile(r"^req_[a-z0-9-]+$")
_MAX_RECORD_BYTES: Final = 1024 * 1024
_RESULT_FIELDS: Final = {
    "outcome",
    "request_id",
    "message",
    "claims",
    "citations",
    "generation_id",
    "request_revision",
    "request_fence",
}


def run_deployed_evaluation(
    root: Path,
    journal_directory: Path,
    identity: DeployedEvaluationIdentity,
    invoker: DeployedInvoker,
    *,
    clock: Callable[[], datetime],
) -> DeployedEvaluationResult:
    """Invoke each exact real case once after durably journaling its request."""
    _validate_inputs(root, journal_directory, identity, invoker)
    cases = _real_cases(root)
    run_identity = {
        "api_version": "valkeyrie.io/deployed-evaluation-run/1",
        "kind": "DeployedEvaluationRun",
        **asdict(identity),
        "case_ids": list(_REAL_CASE_IDS),
    }
    run_id = _content_id("deployed-evaluation-run/1", run_identity)
    try:
        journal_directory.mkdir(mode=0o700)
        _fsync_directory(journal_directory.parent)
    except FileExistsError as error:
        raise DeployedEvaluationError(
            "refusing to resume or repeat a deployed evaluation journal"
        ) from error
    except OSError as error:
        raise DeployedEvaluationError("cannot create deployed evaluation journal") from error
    attempts = journal_directory / "attempts"
    results = journal_directory / "results"
    attempts.mkdir(mode=0o700)
    results.mkdir(mode=0o700)
    _fsync_directory(journal_directory)
    _write_exclusive(
        journal_directory / "run.json",
        _canonical_bytes({**run_identity, "run_id": run_id}),
    )

    observations: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        request_id = _request_id(run_id, cast(str, case["id"]))
        timestamp = _timestamp(clock())
        event = {
            "action": "answer",
            "request_id": request_id,
            "question": case["question"],
            "version_requirement": ("current_state" if case["expected_external_calls"] else "none"),
            "requested_version": None,
            "knowledge_base_id": identity.knowledge_base_id,
            "owner": f"deployed-evaluation-{run_id[-12:]}",
            "now": timestamp,
            "completed_at": timestamp,
            "lease_duration_seconds": 300,
        }
        payload = _canonical_bytes(event, newline=False)
        request_hash = _sha256(payload)
        stem = f"{index:02d}-{case['id']}"
        attempt_path = attempts / f"{stem}.json"
        attempt = {
            "api_version": "valkeyrie.io/deployed-evaluation-attempt/1",
            "kind": "DeployedEvaluationAttempt",
            "run_id": run_id,
            "index": index,
            "case_id": case["id"],
            "function_name": identity.function_name,
            "function_qualifier": identity.function_qualifier,
            "application_revision": identity.application_revision,
            "evaluation_suite_revision": identity.evaluation_suite_revision,
            "generation_id": identity.generation_id,
            "request_id": request_id,
            "request_sha256": request_hash,
            "request": event,
            "attempted": True,
        }
        _write_exclusive(attempt_path, _canonical_bytes(attempt))
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
                f"deployed evaluation stopped after journaled attempt {index}; retry is forbidden"
            ) from error
        result = _bounded_result(raw_result)
        result_bytes = _canonical_bytes(result, newline=False)
        result_hash = _sha256(result_bytes)
        result_path = results / f"{stem}.json"
        _write_exclusive(
            result_path,
            _canonical_bytes(
                {
                    "api_version": "valkeyrie.io/deployed-evaluation-result/1",
                    "kind": "DeployedEvaluationResult",
                    "run_id": run_id,
                    "case_id": case["id"],
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
        observations.append(
            {
                "index": index,
                "case_id": case["id"],
                "expected_behavior": case["expected_behavior"],
                "expected_external_calls": case["expected_external_calls"],
                "request_id": request_id,
                "request_sha256": request_hash,
                "response_sha256": result_hash,
                "outcome": result["outcome"],
                "generation_id": result["generation_id"],
                "attempt_path": attempt_path.relative_to(journal_directory).as_posix(),
                "attempt_sha256": _sha256(attempt_path.read_bytes()),
                "result_path": result_path.relative_to(journal_directory).as_posix(),
                "result_sha256": _sha256(result_path.read_bytes()),
            }
        )

    preimage: dict[str, object] = {
        "api_version": "valkeyrie.io/deployed-evaluation-evidence/1",
        "kind": "DeployedEvaluationEvidence",
        "run_id": run_id,
        "identity": asdict(identity),
        "case_count": len(observations),
        "observations": observations,
        "automatic_retries": 0,
    }
    evidence_id = _content_id("deployed-evaluation-evidence/1", preimage)
    manifest = {**preimage, "evidence_id": evidence_id}
    manifest_path = journal_directory / f"evidence-{evidence_id.removeprefix('sha256:')}.json"
    _write_exclusive(manifest_path, _canonical_bytes(manifest))
    return DeployedEvaluationResult(run_id, evidence_id, journal_directory, manifest_path, manifest)


def _validate_inputs(
    root: Path,
    journal_directory: Path,
    identity: DeployedEvaluationIdentity,
    invoker: DeployedInvoker,
) -> None:
    if not isinstance(root, Path) or not root.is_dir() or root.is_symlink():
        raise DeployedEvaluationError("evaluation root must be a real local directory")
    if not isinstance(journal_directory, Path) or journal_directory.exists():
        raise DeployedEvaluationError("refusing to resume or repeat a deployed evaluation journal")
    parent = journal_directory.parent
    if not parent.is_dir() or parent.is_symlink():
        raise DeployedEvaluationError("evaluation journal parent must be a real directory")
    if not isinstance(identity, DeployedEvaluationIdentity):
        raise DeployedEvaluationError("deployed evaluation identity is malformed")
    if _FUNCTION.fullmatch(identity.function_name) is None:
        raise DeployedEvaluationError("Lambda function name is malformed")
    if _QUALIFIER.fullmatch(identity.function_qualifier) is None:
        raise DeployedEvaluationError("Lambda qualifier must be an immutable numeric version")
    for value, label in (
        (identity.application_revision, "application revision"),
        (identity.evaluation_suite_revision, "evaluation suite revision"),
        (identity.generation_id, "generation ID"),
    ):
        if _DIGEST.fullmatch(value) is None:
            raise DeployedEvaluationError(f"{label} is malformed")
    if _KNOWLEDGE_BASE.fullmatch(identity.knowledge_base_id) is None:
        raise DeployedEvaluationError("knowledge base ID is malformed")
    if not callable(getattr(invoker, "invoke", None)):
        raise DeployedEvaluationError("deployed invoker is malformed")
    suite = load_evaluation_suite(root)
    if suite.revision != identity.evaluation_suite_revision:
        raise DeployedEvaluationError("evaluation suite revision differs from the pinned identity")


def _real_cases(root: Path) -> tuple[Mapping[str, object], ...]:
    try:
        document = load_yaml_mapping(root / "evals/public.yaml")
    except (OSError, UnicodeError, ValueError) as error:
        raise DeployedEvaluationError("cannot load deployed evaluation cases") from error
    values = document.get("cases")
    if not isinstance(values, list):
        raise DeployedEvaluationError("public evaluation cases are malformed")
    cases: list[Mapping[str, object]] = []
    for raw in values:
        if not isinstance(raw, Mapping):
            raise DeployedEvaluationError("public evaluation case is malformed")
        case = cast(Mapping[str, object], raw)
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.startswith("real-"):
            continue
        question = case.get("question")
        behavior = case.get("expected_behavior")
        external = case.get("expected_external_calls")
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(behavior, str)
            or not isinstance(external, list)
            or not all(isinstance(item, str) for item in external)
        ):
            raise DeployedEvaluationError(f"deployed evaluation case {case_id} is malformed")
        cases.append(case)
    if tuple(case["id"] for case in cases) != _REAL_CASE_IDS:
        raise DeployedEvaluationError("deployed evaluation must contain exactly the ten real cases")
    return tuple(cases)


def _bounded_result(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _RESULT_FIELDS:
        raise DeployedEvaluationError("deployed result has an unknown or missing field")
    result = dict(value)
    content = _canonical_bytes(result, newline=False)
    if not content or len(content) > _MAX_RECORD_BYTES:
        raise DeployedEvaluationError("deployed result exceeds its byte bound")
    return result


def _validate_result(
    result: Mapping[str, object],
    request_id: str,
    generation_id: str,
    live: bool,
    expected_behavior: str,
) -> None:
    if result.get("request_id") != request_id:
        raise DeployedEvaluationError("deployed result request identity is inconsistent")
    outcome = result.get("outcome")
    if outcome not in {"answer", "clarification", "abstention", "partial", "error"}:
        raise DeployedEvaluationError("deployed result outcome is unsupported")
    expected_outcome = "abstention" if expected_behavior == "abstain" else expected_behavior
    if outcome != expected_outcome:
        raise DeployedEvaluationError("deployed result does not satisfy the reviewed case behavior")
    observed_generation = result.get("generation_id")
    if live:
        if observed_generation is not None:
            raise DeployedEvaluationError(
                "live deployed result claimed static generation provenance"
            )
    elif observed_generation != generation_id:
        raise DeployedEvaluationError("static deployed result differs from the pinned generation")
    message = result.get("message")
    if message is not None and not isinstance(message, str):
        raise DeployedEvaluationError("deployed result message is malformed")
    if outcome != "answer" and not isinstance(message, str):
        raise DeployedEvaluationError("non-answer deployed result requires a message")
    for field in ("claims", "citations"):
        if not isinstance(result.get(field), list):
            raise DeployedEvaluationError(f"deployed result {field} is malformed")


def _request_id(run_id: str, case_id: str) -> str:
    digest = hashlib.sha256(f"{run_id}\0{case_id}".encode()).hexdigest()
    value = f"req_deployed-eval-{digest}"
    if _REQUEST_ID.fullmatch(value) is None:  # pragma: no cover - construction is fixed
        raise DeployedEvaluationError("deterministic request ID is malformed")
    return value


def _timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DeployedEvaluationError("evaluation clock must return an aware datetime")
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_exclusive(path: Path, content: bytes) -> None:
    if not 1 <= len(content) <= _MAX_RECORD_BYTES:
        raise DeployedEvaluationError("evaluation journal record exceeds its byte bound")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except FileExistsError as error:
        raise DeployedEvaluationError(
            "refusing to overwrite an evaluation journal record"
        ) from error
    except OSError as error:
        raise DeployedEvaluationError("cannot persist an evaluation journal record") from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        content = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    except (TypeError, ValueError, UnicodeError) as error:
        raise DeployedEvaluationError("evaluation value is not canonical JSON") from error
    return content + (b"\n" if newline else b"")


def _content_id(domain: str, value: object) -> str:
    digest = hashlib.sha256()
    for part in (domain.encode(), _canonical_bytes(value, newline=False)):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return "sha256:" + digest.hexdigest()


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()
