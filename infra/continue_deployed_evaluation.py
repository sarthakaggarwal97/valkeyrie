from __future__ import annotations

import argparse
import json
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from valkeyrie.aws_adapters import AwsLambdaInvoker, create_lambda_client
from valkeyrie.deployed_evaluation import (
    DeployedEvaluationError,
    DeployedEvaluationIdentity,
    DeployedEvaluationResult,
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

_MAX_RECORD_BYTES = 1024 * 1024
_DIGEST_PREFIX = "sha256:"


@dataclass(frozen=True)
class AdoptedJournalHashes:
    run_sha256: str
    attempt_sha256: str
    result_sha256: str


def continue_after_nullable_message_rejection(
    root: Path,
    adopted_journal: Path,
    journal_directory: Path,
    identity: DeployedEvaluationIdentity,
    hashes: AdoptedJournalHashes,
    invoker: DeployedInvoker,
    *,
    clock: Callable[[], datetime],
) -> DeployedEvaluationResult:
    """Adopt one verified result and invoke only the remaining nine reviewed cases."""
    _validate_inputs(root, journal_directory, identity, invoker)
    project_root = root.resolve(strict=True)
    source = adopted_journal.resolve(strict=True)
    try:
        source_locator = source.relative_to(project_root).as_posix()
    except ValueError as error:
        raise DeployedEvaluationError(
            "adopted journal must be preserved under the project root"
        ) from error
    cases = _real_cases(project_root)
    run_id, adopted_observation, source_record = _validate_adopted_journal(
        source, identity, hashes, cases[0], source_locator
    )

    try:
        journal_directory.mkdir(mode=0o700)
        _fsync_directory(journal_directory.parent)
    except FileExistsError as error:
        raise DeployedEvaluationError(
            "refusing to resume or repeat a continuation journal"
        ) from error
    except OSError as error:
        raise DeployedEvaluationError("cannot create continuation journal") from error
    attempts = journal_directory / "attempts"
    results = journal_directory / "results"
    attempts.mkdir(mode=0o700)
    results.mkdir(mode=0o700)
    _fsync_directory(journal_directory)

    continuation_preimage: dict[str, object] = {
        "api_version": "valkeyrie.io/deployed-evaluation-continuation/1",
        "kind": "DeployedEvaluationContinuation",
        "logical_run_id": run_id,
        "identity": asdict(identity),
        "adopted_source": source_record,
        "reason": "post-result nullable-message validator mismatch",
        "remaining_case_ids": [case["id"] for case in cases[1:]],
    }
    continuation_id = _content_id("deployed-evaluation-continuation/1", continuation_preimage)
    continuation_run = {**continuation_preimage, "continuation_id": continuation_id}
    continuation_path = journal_directory / "continuation.json"
    _write_exclusive(continuation_path, _canonical_bytes(continuation_run))

    observations: list[dict[str, object]] = [adopted_observation]
    for index, case in enumerate(cases[1:], start=2):
        case_id = cast(str, case["id"])
        request_id = _request_id(run_id, case_id)
        timestamp = _timestamp(clock())
        event = {
            "action": "answer",
            "request_id": request_id,
            "question": case["question"],
            "version_requirement": ("current_state" if case["expected_external_calls"] else "none"),
            "requested_version": None,
            "knowledge_base_id": identity.knowledge_base_id,
            "owner": f"deployed-evaluation-{run_id[-12:]}",
            "lease_duration_seconds": 300,
            "now": timestamp,
            "completed_at": timestamp,
        }
        payload = _canonical_bytes(event, newline=False)
        request_hash = _sha256(payload)
        stem = f"{index:02d}-{case_id}"
        attempt_path = attempts / f"{stem}.json"
        _write_exclusive(
            attempt_path,
            _canonical_bytes(
                {
                    "api_version": "valkeyrie.io/deployed-evaluation-attempt/1",
                    "kind": "DeployedEvaluationAttempt",
                    "run_id": run_id,
                    "index": index,
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
            function_name=identity.function_name,
            function_qualifier=identity.function_qualifier,
            application_revision=identity.application_revision,
            evaluation_suite_revision=identity.evaluation_suite_revision,
            generation_id=identity.generation_id,
            request_id=request_id,
            payload=payload,
        )
        try:
            raw_result = invoker.invoke(invocation)
        except Exception as error:
            raise DeployedEvaluationError(
                f"continuation stopped after journaled attempt {index}; retry is forbidden"
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
        observations.append(
            _observation(
                index,
                case,
                request_id,
                request_hash,
                result_hash,
                result,
                "continuation",
                attempt_path.relative_to(journal_directory).as_posix(),
                result_path.relative_to(journal_directory).as_posix(),
                _sha256(attempt_path.read_bytes()),
                _sha256(result_path.read_bytes()),
            )
        )

    preimage: dict[str, object] = {
        "api_version": "valkeyrie.io/deployed-evaluation-recovery-evidence/1",
        "kind": "DeployedEvaluationRecoveryEvidence",
        "logical_run_id": run_id,
        "continuation_id": continuation_id,
        "identity": asdict(identity),
        "case_count": len(observations),
        "invocation_count": len(observations),
        "adopted_journaled_results": 1,
        "new_invocations": 9,
        "automatic_retries": 0,
        "source_journals": [
            source_record,
            {
                "role": "continuation",
                "continuation_sha256": _sha256(continuation_path.read_bytes()),
                "adapter_envelope": "validated_in_memory_not_persisted",
            },
        ],
        "observations": observations,
    }
    evidence_id = _content_id("deployed-evaluation-recovery-evidence/1", preimage)
    manifest = {**preimage, "evidence_id": evidence_id}
    manifest_path = journal_directory / f"evidence-{evidence_id.removeprefix('sha256:')}.json"
    _write_exclusive(manifest_path, _canonical_bytes(manifest))
    return DeployedEvaluationResult(run_id, evidence_id, journal_directory, manifest_path, manifest)


def _validate_adopted_journal(
    source: Path,
    identity: DeployedEvaluationIdentity,
    hashes: AdoptedJournalHashes,
    first_case: Mapping[str, object],
    source_locator: str,
) -> tuple[str, dict[str, object], dict[str, object]]:
    for value in asdict(hashes).values():
        if not isinstance(value, str) or not value.startswith(_DIGEST_PREFIX) or len(value) != 71:
            raise DeployedEvaluationError("adopted journal hash is malformed")
    expected_entries = {"run.json", "attempts", "results"}
    if source.is_symlink() or {path.name for path in source.iterdir()} != expected_entries:
        raise DeployedEvaluationError("adopted journal layout is not exact")
    attempts = source / "attempts"
    results = source / "results"
    if attempts.is_symlink() or results.is_symlink():
        raise DeployedEvaluationError("adopted journal contains a symlink")
    attempt_path = attempts / "01-real-governance.json"
    result_path = results / "01-real-governance.json"
    if {path.name for path in attempts.iterdir()} != {attempt_path.name} or {
        path.name for path in results.iterdir()
    } != {result_path.name}:
        raise DeployedEvaluationError("adopted journal must contain exactly one complete result")
    paths_and_hashes = (
        (source / "run.json", hashes.run_sha256),
        (attempt_path, hashes.attempt_sha256),
        (result_path, hashes.result_sha256),
    )
    for path, expected_hash in paths_and_hashes:
        if path.is_symlink():
            raise DeployedEvaluationError("adopted journal file mode or type is unsafe")
        mode = path.lstat().st_mode
        # The protection that matters is that nobody but the owner can rewrite an adopted
        # journal before its hash is checked. Requiring exactly 0o600 also rejected 0o400
        # and 0o644, and journals committed to git necessarily check out as 0o644 because
        # git records only the executable bit, so a fresh clone could never satisfy it.
        if not stat.S_ISREG(mode) or stat.S_IMODE(mode) & (stat.S_IWGRP | stat.S_IWOTH):
            raise DeployedEvaluationError("adopted journal file mode or type is unsafe")
        if _sha256(path.read_bytes()) != expected_hash:
            raise DeployedEvaluationError("adopted journal hash differs from its explicit pin")
    run = _load_record(source / "run.json")
    run_identity = {
        "api_version": "valkeyrie.io/deployed-evaluation-run/1",
        "kind": "DeployedEvaluationRun",
        **asdict(identity),
        "case_ids": [case["id"] for case in _real_cases(source.parents[3])],
    }
    run_id = _content_id("deployed-evaluation-run/1", run_identity)
    if run != {**run_identity, "run_id": run_id}:
        raise DeployedEvaluationError("adopted run identity is inconsistent")

    request_id = _request_id(run_id, cast(str, first_case["id"]))
    attempt = _load_record(attempt_path)
    request = attempt.get("request")
    if not isinstance(request, Mapping):
        raise DeployedEvaluationError("adopted request is malformed")
    timestamp = request.get("now")
    expected_request = {
        "action": "answer",
        "request_id": request_id,
        "question": first_case["question"],
        "version_requirement": "none",
        "requested_version": None,
        "knowledge_base_id": identity.knowledge_base_id,
        "owner": f"deployed-evaluation-{run_id[-12:]}",
        "lease_duration_seconds": 300,
        "now": timestamp,
        "completed_at": timestamp,
    }
    request_hash = _sha256(_canonical_bytes(expected_request, newline=False))
    expected_attempt = {
        "api_version": "valkeyrie.io/deployed-evaluation-attempt/1",
        "kind": "DeployedEvaluationAttempt",
        "run_id": run_id,
        "index": 1,
        "case_id": first_case["id"],
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
        raise DeployedEvaluationError("adopted attempt is inconsistent")

    result_record = _load_record(result_path)
    response = result_record.get("response")
    result = _bounded_result(response)
    response_hash = _sha256(_canonical_bytes(result, newline=False))
    expected_result = {
        "api_version": "valkeyrie.io/deployed-evaluation-result/1",
        "kind": "DeployedEvaluationResult",
        "run_id": run_id,
        "case_id": first_case["id"],
        "request_id": request_id,
        "request_sha256": request_hash,
        "response_sha256": response_hash,
        "response": result,
    }
    if result_record != expected_result:
        raise DeployedEvaluationError("adopted result is inconsistent")
    _validate_result(
        result,
        request_id,
        identity.generation_id,
        False,
        cast(str, first_case["expected_behavior"]),
    )
    observation = _observation(
        1,
        first_case,
        request_id,
        request_hash,
        response_hash,
        result,
        "adopted_initial",
        "attempts/01-real-governance.json",
        "results/01-real-governance.json",
        hashes.attempt_sha256,
        hashes.result_sha256,
    )
    source_record: dict[str, object] = {
        "role": "adopted_initial",
        "locator": source_locator,
        "run_sha256": hashes.run_sha256,
        "attempt_sha256": hashes.attempt_sha256,
        "result_sha256": hashes.result_sha256,
        "adapter_envelope": "validated_in_memory_not_persisted",
    }
    return run_id, observation, source_record


def _observation(
    index: int,
    case: Mapping[str, object],
    request_id: str,
    request_hash: str,
    response_hash: str,
    result: Mapping[str, object],
    journal_role: str,
    attempt_path: str,
    result_path: str,
    attempt_hash: str,
    result_hash: str,
) -> dict[str, object]:
    return {
        "index": index,
        "case_id": case["id"],
        "expected_behavior": case["expected_behavior"],
        "expected_external_calls": case["expected_external_calls"],
        "request_id": request_id,
        "request_sha256": request_hash,
        "response_sha256": response_hash,
        "outcome": result["outcome"],
        "generation_id": result["generation_id"],
        "journal_role": journal_role,
        "attempt_path": attempt_path,
        "attempt_sha256": attempt_hash,
        "result_path": result_path,
        "result_sha256": result_hash,
    }


def _bounded_result(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DeployedEvaluationError("deployed result root is malformed")
    result = dict(value)
    expected = {
        "outcome",
        "request_id",
        "message",
        "claims",
        "citations",
        "generation_id",
        "request_revision",
        "request_fence",
    }
    if set(result) != expected:
        raise DeployedEvaluationError("deployed result has an unknown or missing field")
    if len(_canonical_bytes(result, newline=False)) > _MAX_RECORD_BYTES:
        raise DeployedEvaluationError("deployed result exceeds its byte bound")
    return result


def _load_record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    if not 1 <= len(content) <= _MAX_RECORD_BYTES:
        raise DeployedEvaluationError("adopted journal record exceeds its byte bound")

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
        raise DeployedEvaluationError("adopted journal record is not strict JSON") from error
    if not isinstance(value, Mapping):
        raise DeployedEvaluationError("adopted journal record root is malformed")
    return dict(cast(Mapping[str, object], value))


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Continue cases 2-10 after the verified nullable-message validator mismatch."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--adopted-journal", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--function-name", required=True)
    parser.add_argument("--function-qualifier", required=True)
    parser.add_argument("--application-revision", required=True)
    parser.add_argument("--evaluation-suite-revision", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--run-sha256", required=True)
    parser.add_argument("--attempt-sha256", required=True)
    parser.add_argument("--result-sha256", required=True)
    values = parser.parse_args(arguments)
    identity = DeployedEvaluationIdentity(
        values.function_name,
        values.function_qualifier,
        values.application_revision,
        values.evaluation_suite_revision,
        values.generation_id,
        values.knowledge_base_id,
    )
    hashes = AdoptedJournalHashes(values.run_sha256, values.attempt_sha256, values.result_sha256)
    try:
        result = continue_after_nullable_message_rejection(
            values.root,
            values.adopted_journal,
            values.journal,
            identity,
            hashes,
            AwsLambdaInvoker(create_lambda_client(values.region)),
            clock=lambda: datetime.now(UTC),
        )
    except (DeployedEvaluationError, OSError, RuntimeError, ValueError) as error:
        print(f"deployed evaluation continuation failed: {error}", file=sys.stderr)
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
