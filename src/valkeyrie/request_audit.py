"""Immutable executable request pins with lease/fence-bound terminal audit.

The first successful conditional claim stores the already verified A-04
``ModelInvocation`` itself, not merely identifiers describing it. Retries and
expired-lease recovery resolve that exact frozen model-visible input, model
target, inference settings, and application revision. The compact terminal
audit export is derived from the stored executable plan.

Claims carry an owner and bounded lease. Recovery requires the caller's exact
nonterminal record, an expired lease, and a different owner; one atomic
nonterminal compare-and-swap updates owner, lease, revision, and fence without
altering the pin. Completion uses the same store invariant, so a terminal
record can never be replaced.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal, Protocol, cast
from urllib.parse import urlsplit

from valkeyrie.answer_models import (
    AnswerModelError,
    AnswerModelProfile,
    AnswerModelSelection,
    _validate_profile,
    _validated_model_revision,
)
from valkeyrie.drafting import (
    DraftingError,
    ModelInput,
    ModelInvocation,
    prepare_model_invocation,
)
from valkeyrie.evaluations import EvaluationSuite
from valkeyrie.evidence import EvidencePackage
from valkeyrie.generation import GenerationBundle
from valkeyrie.prompts import PromptTemplate


class RequestAuditError(ValueError):
    """A request pin, claim, live observation, or completion input is invalid."""


RequestOutcome = Literal["answer", "clarification", "abstention", "error"]


@dataclass(frozen=True)
class LiveObservation:
    """One content-addressed exact snapshot of a single live GitHub read."""

    observation_id: str
    observed_at: str
    source_url: str
    object_type: str
    canonical_payload: bytes
    payload_digest: str
    complete: bool
    truncated: bool


@dataclass(frozen=True)
class PinnedExecution:
    """The exact executable A-04 plan and reviewed application revision."""

    invocation: ModelInvocation
    application_revision: str


@dataclass(frozen=True)
class RequestPin:
    """The immutable executable plan recorded by the first successful claim."""

    request_id: str
    request_content_digest: str
    execution: PinnedExecution
    live_observations: tuple[LiveObservation, ...]
    started_at: str

    @property
    def generation_id(self) -> str:
        return self.execution.invocation.input.evidence.generation_id

    @property
    def prompt_revision(self) -> str:
        return self.execution.invocation.prompt_revision

    @property
    def application_revision(self) -> str:
        return self.execution.application_revision

    @property
    def model_revision(self) -> str:
        return self.execution.invocation.profile.model_revision

    @property
    def inference_config_revision(self) -> str:
        return self.execution.invocation.profile.inference_config_revision

    @property
    def static_evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            record.evidence_id for record in self.execution.invocation.input.evidence.records
        )


@dataclass(frozen=True)
class RequestAuditRecord:
    """One conditional request item: immutable pin plus fenced leased lifecycle."""

    pin: RequestPin
    revision: int
    fence: int
    owner: str
    lease_expires_at: str
    outcome: RequestOutcome | None
    completed_at: str | None


@dataclass(frozen=True)
class RequestClaim:
    """The authoritative record plus whether this call created the pin."""

    record: RequestAuditRecord
    pinned: bool


class RequestAuditStore(Protocol):
    """Strongly consistent request store with terminal immutability as an invariant."""

    def get_request(self, request_id: str) -> RequestAuditRecord | None: ...

    def put_request_if_absent(self, record: RequestAuditRecord) -> RequestAuditRecord | None:
        """Atomically insert one record; return None on success or the existing record."""
        ...

    def compare_and_swap_nonterminal(
        self,
        expected: RequestAuditRecord,
        replacement: RequestAuditRecord,
    ) -> bool:
        """Replace an exact nonterminal expected record atomically.

        The store must return false when the stored or supplied expected record
        is terminal. A nonterminal expected record may advance once to another
        nonterminal lease or to its first terminal value; a terminal record can
        never be replaced.
        """
        ...


_AUDIT_API_VERSION: Final = "valkeyrie.io/request-audit/1"
_OBSERVATION_API_VERSION: Final = "valkeyrie.io/live-observation/1"
_OBSERVATION_IDENTITY_VERSION: Final = "valkeyrie.io/live-observation-identity/1"
_CONTENT_API_VERSION: Final = "valkeyrie.io/request-content/1"
_REQUEST_ID: Final = re.compile(r"^req_[a-z0-9-]+$")
_OBSERVATION_ID: Final = re.compile(r"^obs_[0-9a-f]{64}$")
_EVIDENCE_ID: Final = re.compile(r"^ev_[a-z0-9-]+$")
_DIGEST: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_TIMESTAMP: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)
_OBSERVATION_TYPES: Final = frozenset(
    {
        "repository",
        "issue",
        "pull_request",
        "review",
        "check",
        "workflow_run",
        "tag",
        "release",
        "controller_status",
    }
)
_OUTCOMES: Final = ("answer", "clarification", "abstention", "error")
_MAX_QUESTION_BYTES: Final = 8 * 1024
_MAX_URL_CHARACTERS: Final = 2_048
_MAX_OBSERVATIONS: Final = 50
_MAX_STATIC_EVIDENCE_IDS: Final = 50
_MAX_PAYLOAD_BYTES: Final = 256 * 1024
_MAX_OWNER_BYTES: Final = 128
_MAX_LEASE_SECONDS: Final = 60 * 60


def claim_request(
    store: RequestAuditStore,
    root: Path,
    suite: EvaluationSuite,
    reports: tuple[Mapping[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    evidence: EvidencePackage,
    *,
    request_id: str,
    question: str,
    application_revision: str,
    owner: str,
    now: str,
    lease_duration_seconds: int,
    live_observations: tuple[LiveObservation, ...] = (),
) -> RequestClaim:
    """Idempotently claim a request and pin its complete executable plan."""
    _validate_store(store)
    _validated_request_id(request_id)
    content_digest = request_content_digest(question)
    _validated_owner(owner)
    now_value = _validated_timestamp(now, "claim timestamp")
    lease_seconds = _validated_lease_duration(lease_duration_seconds)
    lease_expires_at = _lease_expiration(now_value, lease_seconds)

    existing = store.get_request(request_id)
    if existing is not None:
        return RequestClaim(_authoritative(existing, request_id, content_digest), False)

    pin = _build_pin(
        root,
        suite,
        reports,
        selection,
        bundle,
        evidence,
        request_id=request_id,
        question=question,
        content_digest=content_digest,
        application_revision=application_revision,
        started_at=now,
        live_observations=live_observations,
    )
    record = RequestAuditRecord(
        pin=pin,
        revision=1,
        fence=1,
        owner=owner,
        lease_expires_at=lease_expires_at,
        outcome=None,
        completed_at=None,
    )
    loser = store.put_request_if_absent(record)
    if loser is None:
        return RequestClaim(record, True)
    return RequestClaim(_authoritative(loser, request_id, content_digest), False)


def resolve_pinned_execution(
    store: RequestAuditStore,
    *,
    request_id: str,
    question: str,
) -> PinnedExecution:
    """Return the exact stored executable plan for a retry, or fail closed."""
    _validate_store(store)
    _validated_request_id(request_id)
    record = _authoritative(
        _stored_record(store, request_id), request_id, request_content_digest(question)
    )
    return record.pin.execution


def recover_request(
    store: RequestAuditStore,
    expected: RequestAuditRecord,
    *,
    new_owner: str,
    now: str,
    lease_duration_seconds: int,
) -> RequestAuditRecord:
    """Atomically take over an expired exact nonterminal claim without changing its pin."""
    _validate_store(store)
    _validate_record(expected)
    if expected.outcome is not None:
        raise RequestAuditError("request audit is already terminal")
    owner = _validated_owner(new_owner)
    if owner == expected.owner:
        raise RequestAuditError("recovery requires a new request owner")
    now_value = _validated_timestamp(now, "recovery timestamp")
    expiration = _validated_timestamp(expected.lease_expires_at, "lease expiration")
    if now_value < expiration:
        raise RequestAuditError("request lease is still active")
    lease_seconds = _validated_lease_duration(lease_duration_seconds)
    replacement = replace(
        expected,
        revision=expected.revision + 1,
        fence=expected.fence + 1,
        owner=owner,
        lease_expires_at=_lease_expiration(now_value, lease_seconds),
    )
    if not store.compare_and_swap_nonterminal(expected, replacement):
        raise RequestAuditError("request claim state changed during recovery")
    return replacement


def complete_request(
    store: RequestAuditStore,
    claimed: RequestAuditRecord,
    *,
    outcome: RequestOutcome,
    completed_at: str,
) -> RequestAuditRecord:
    """Complete an exact nonterminal claim once, preserving terminal immutability."""
    _validate_store(store)
    _validate_record(claimed)
    if claimed.outcome is not None:
        raise RequestAuditError("completion requires a non-terminal claimed record")
    if outcome not in _OUTCOMES:
        raise RequestAuditError("request outcome is malformed")
    _validated_timestamp(completed_at, "completion timestamp")
    replacement = replace(
        claimed,
        revision=claimed.revision + 1,
        outcome=outcome,
        completed_at=completed_at,
    )
    if not store.compare_and_swap_nonterminal(claimed, replacement):
        raise RequestAuditError("request state changed during completion")
    return replacement


def create_live_observation(
    *,
    observed_at: str,
    source_url: str,
    object_type: str,
    payload: object,
) -> LiveObservation:
    """Create a bounded immutable exact JSON snapshot with internally derived identity."""
    _validated_timestamp(observed_at, "observation timestamp")
    _validated_https_url(source_url)
    if object_type not in _OBSERVATION_TYPES:
        raise RequestAuditError("live observation object type is unsupported")
    canonical_payload = _canonical_payload(payload)
    payload_digest = _sha256(canonical_payload)
    observation_id = _observation_id(
        observed_at=observed_at,
        source_url=source_url,
        object_type=object_type,
        payload_digest=payload_digest,
    )
    return LiveObservation(
        observation_id=observation_id,
        observed_at=observed_at,
        source_url=source_url,
        object_type=object_type,
        canonical_payload=canonical_payload,
        payload_digest=payload_digest,
        complete=True,
        truncated=False,
    )


def request_content_digest(question: str) -> str:
    """Return the canonical sha256 digest of one bounded request's content."""
    _validated_question(question)
    return _sha256(
        _canonical_json(
            {
                "api_version": _CONTENT_API_VERSION,
                "kind": "RequestContent",
                "question": question,
            }
        )
    )


def live_observation_value(observation: LiveObservation) -> dict[str, object]:
    """Return a fresh full payload value matching the live-observation schema."""
    _validate_observation(observation)
    return {
        "api_version": _OBSERVATION_API_VERSION,
        "kind": "LiveObservation",
        "observation_id": observation.observation_id,
        "observed_at": observation.observed_at,
        "source_url": observation.source_url,
        "object_type": observation.object_type,
        "payload": _decoded_payload(observation.canonical_payload),
        "payload_digest": observation.payload_digest,
        "complete": True,
        "truncated": False,
    }


def request_audit_value(record: RequestAuditRecord) -> dict[str, object]:
    """Derive one compact terminal audit export from the stored executable plan."""
    _validate_record(record)
    if record.outcome is None:
        raise RequestAuditError("request audit value requires a terminal record")
    pin = record.pin
    return {
        "api_version": _AUDIT_API_VERSION,
        "kind": "RequestAudit",
        "request_id": pin.request_id,
        "request_content_digest": pin.request_content_digest,
        "generation_id": pin.generation_id,
        "prompt_revision": pin.prompt_revision,
        "application_revision": pin.application_revision,
        "model_revision": pin.model_revision,
        "inference_config_revision": pin.inference_config_revision,
        "static_evidence_ids": list(pin.static_evidence_ids),
        "live_observations": [
            live_observation_value(observation) for observation in pin.live_observations
        ],
        "started_at": pin.started_at,
        "outcome": record.outcome,
    }


def _build_pin(
    root: Path,
    suite: EvaluationSuite,
    reports: tuple[Mapping[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    evidence: EvidencePackage,
    *,
    request_id: str,
    question: str,
    content_digest: str,
    application_revision: str,
    started_at: str,
    live_observations: tuple[LiveObservation, ...],
) -> RequestPin:
    _validated_digest(application_revision, "application revision")
    observations = _canonical_observations(live_observations)
    try:
        invocation = prepare_model_invocation(
            root, question, suite, reports, selection, bundle, evidence
        )
    except DraftingError as error:
        raise RequestAuditError(f"request pin inputs are invalid: {error}") from error
    pin = RequestPin(
        request_id=request_id,
        request_content_digest=content_digest,
        execution=PinnedExecution(invocation, application_revision),
        live_observations=observations,
        started_at=started_at,
    )
    _validate_pin(pin)
    return pin


def _authoritative(
    existing: object,
    request_id: str,
    content_digest: str,
) -> RequestAuditRecord:
    record = _validate_record(existing)
    if record.pin.request_id != request_id:
        raise RequestAuditError("request audit store returned a mismatched identity")
    if record.pin.request_content_digest != content_digest:
        raise RequestAuditError("request identity is already pinned to different request content")
    return record


def _stored_record(store: RequestAuditStore, request_id: str) -> RequestAuditRecord:
    current = store.get_request(request_id)
    if current is None:
        raise RequestAuditError("request is unknown")
    record = _validate_record(current)
    if record.pin.request_id != request_id:
        raise RequestAuditError("request audit store returned a mismatched identity")
    return record


def _validate_record(record: object) -> RequestAuditRecord:
    if not isinstance(record, RequestAuditRecord):
        raise RequestAuditError("request audit record has the wrong runtime type")
    _validate_pin(record.pin)
    for value, label in ((record.revision, "revision"), (record.fence, "fence")):
        if type(value) is not int or value < 1:
            raise RequestAuditError(f"request audit {label} is malformed")
    if record.revision < record.fence:
        raise RequestAuditError("request audit revision cannot trail its fence")
    _validated_owner(record.owner)
    _validated_timestamp(record.lease_expires_at, "lease expiration")
    if record.outcome is None:
        if record.completed_at is not None:
            raise RequestAuditError("non-terminal request audit cannot carry a completion time")
    else:
        if record.outcome not in _OUTCOMES:
            raise RequestAuditError("request audit outcome is malformed")
        _validated_timestamp(record.completed_at, "completion timestamp")
    return record


def _validate_pin(pin: object) -> None:
    if not isinstance(pin, RequestPin):
        raise RequestAuditError("request pin has the wrong runtime type")
    _validated_request_id(pin.request_id)
    _validated_digest(pin.request_content_digest, "request content digest")
    _validated_timestamp(pin.started_at, "request start timestamp")
    _validate_execution(pin.execution)
    if (
        request_content_digest(pin.execution.invocation.input.question)
        != pin.request_content_digest
    ):
        raise RequestAuditError("pinned question does not match the request content digest")
    ids = pin.static_evidence_ids
    if not 1 <= len(ids) <= _MAX_STATIC_EVIDENCE_IDS:
        raise RequestAuditError("pinned static evidence IDs are outside their bound")
    if any(_EVIDENCE_ID.fullmatch(item) is None for item in ids):
        raise RequestAuditError("pinned static evidence ID is malformed")
    if list(ids) != sorted(set(ids)):
        raise RequestAuditError("pinned static evidence IDs must be unique and lexically ordered")
    canonical = _canonical_observations(pin.live_observations)
    if pin.live_observations != canonical:
        raise RequestAuditError("pinned live observations must be canonically ordered")


def _validate_execution(execution: object) -> None:
    if not isinstance(execution, PinnedExecution):
        raise RequestAuditError("pinned execution has the wrong runtime type")
    _validated_digest(execution.application_revision, "pinned application revision")
    invocation = execution.invocation
    if not isinstance(invocation, ModelInvocation) or not isinstance(invocation.input, ModelInput):
        raise RequestAuditError("pinned model invocation has the wrong runtime type")
    if not isinstance(invocation.profile, AnswerModelProfile):
        raise RequestAuditError("pinned answer-model profile has the wrong runtime type")
    try:
        _validate_profile(invocation.profile)
        _validated_model_revision(invocation.profile.model_revision)
    except AnswerModelError as error:
        raise RequestAuditError("pinned answer-model profile is malformed") from error
    if not isinstance(invocation.input.prompts, tuple) or not invocation.input.prompts:
        raise RequestAuditError("pinned prompts must be a non-empty immutable tuple")
    if any(not isinstance(prompt, PromptTemplate) for prompt in invocation.input.prompts):
        raise RequestAuditError("pinned prompt has the wrong runtime type")
    if not isinstance(invocation.input.evidence, EvidencePackage):
        raise RequestAuditError("pinned evidence package has the wrong runtime type")
    _validated_question(invocation.input.question)
    if invocation.prompt_revision != invocation.profile.prompt_revision:
        raise RequestAuditError("pinned prompt revision does not match the model profile")
    if invocation.input.evidence.generation_id != invocation.profile.corpus_generation:
        raise RequestAuditError("pinned evidence generation does not match the model profile")
    for value, label in (
        (invocation.prompt_revision, "pinned prompt revision"),
        (invocation.input.evidence.generation_id, "pinned generation ID"),
        (invocation.input.evidence.digest, "pinned evidence package digest"),
    ):
        _validated_digest(value, label)


def _canonical_observations(values: object) -> tuple[LiveObservation, ...]:
    if not isinstance(values, tuple):
        raise RequestAuditError("live observations must be an immutable tuple")
    if len(values) > _MAX_OBSERVATIONS:
        raise RequestAuditError("live observation count is outside its bound")
    for observation in values:
        _validate_observation(observation)
    ids = [observation.observation_id for observation in values]
    if len(ids) != len(set(ids)):
        raise RequestAuditError("duplicate live observation")
    return tuple(sorted(values, key=lambda observation: observation.observation_id))


def _validate_observation(observation: object) -> None:
    if not isinstance(observation, LiveObservation):
        raise RequestAuditError("live observation has the wrong runtime type")
    if (
        not isinstance(observation.observation_id, str)
        or _OBSERVATION_ID.fullmatch(observation.observation_id) is None
    ):
        raise RequestAuditError("live observation ID is malformed")
    _validated_timestamp(observation.observed_at, "observation timestamp")
    _validated_https_url(observation.source_url)
    if observation.object_type not in _OBSERVATION_TYPES:
        raise RequestAuditError("live observation object type is unsupported")
    payload = _decoded_payload(observation.canonical_payload)
    canonical_payload = _canonical_payload(payload)
    if canonical_payload != observation.canonical_payload:
        raise RequestAuditError("live observation payload is not canonical JSON")
    payload_digest = _sha256(canonical_payload)
    if observation.payload_digest != payload_digest:
        raise RequestAuditError("live observation payload digest does not match its payload")
    expected_id = _observation_id(
        observed_at=observation.observed_at,
        source_url=observation.source_url,
        object_type=observation.object_type,
        payload_digest=payload_digest,
    )
    if observation.observation_id != expected_id:
        raise RequestAuditError("live observation identity does not match its exact content")
    if observation.complete is not True or observation.truncated is not False:
        raise RequestAuditError("live observation must be a complete untruncated exact snapshot")


def _observation_id(
    *,
    observed_at: str,
    source_url: str,
    object_type: str,
    payload_digest: str,
) -> str:
    identity = {
        "api_version": _OBSERVATION_IDENTITY_VERSION,
        "kind": "LiveObservationIdentity",
        "observed_at": observed_at,
        "source_url": source_url,
        "object_type": object_type,
        "payload_digest": payload_digest,
        "complete": True,
        "truncated": False,
    }
    return f"obs_{hashlib.sha256(_canonical_json(identity)).hexdigest()}"


def _canonical_payload(value: object) -> bytes:
    _validated_json_value(value)
    try:
        encoded = _canonical_json(value)
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise RequestAuditError("live observation payload is not canonical JSON") from error
    if not 1 <= len(encoded) <= _MAX_PAYLOAD_BYTES:
        raise RequestAuditError(
            f"live observation payload exceeds its {_MAX_PAYLOAD_BYTES}-byte bound"
        )
    return encoded


def _decoded_payload(value: object) -> object:
    if not isinstance(value, bytes) or not 1 <= len(value) <= _MAX_PAYLOAD_BYTES:
        raise RequestAuditError("live observation canonical payload is outside its byte bound")
    try:
        text = value.decode("utf-8")
        decoded = cast(object, json.loads(text))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise RequestAuditError("live observation canonical payload is not valid JSON") from error
    _validated_json_value(decoded)
    return decoded


def _validated_json_value(value: object, ancestors: set[int] | None = None) -> None:
    if ancestors is None:
        ancestors = set()
    if value is None or type(value) is bool or type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise RequestAuditError("live observation payload contains a non-finite number")
        return
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise RequestAuditError("live observation payload contains invalid UTF-8") from error
        return
    if isinstance(value, (list, dict)):
        identity = id(value)
        if identity in ancestors:
            raise RequestAuditError("live observation payload contains a cyclic JSON value")
        ancestors.add(identity)
        try:
            if isinstance(value, list):
                for item in value:
                    _validated_json_value(item, ancestors)
            else:
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise RequestAuditError("live observation JSON object key is not text")
                    _validated_json_value(key, ancestors)
                    _validated_json_value(item, ancestors)
        except RecursionError as error:
            raise RequestAuditError("live observation payload nesting is too deep") from error
        finally:
            ancestors.remove(identity)
        return
    raise RequestAuditError("live observation payload contains a non-JSON value")


def _validated_question(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestAuditError("user question must be non-blank text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise RequestAuditError("user question must be valid UTF-8") from error
    if len(encoded) > _MAX_QUESTION_BYTES:
        raise RequestAuditError(f"user question exceeds its {_MAX_QUESTION_BYTES}-byte bound")
    if any(
        (ord(character) < 32 and character not in "\t\n") or 127 <= ord(character) <= 159
        for character in value
    ):
        raise RequestAuditError("user question contains a control character")
    return value


def _validated_https_url(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= _MAX_URL_CHARACTERS:
        raise RequestAuditError("live observation URL is outside its bound")
    if any(ord(character) <= 32 or ord(character) == 127 for character in value):
        raise RequestAuditError("live observation URL contains whitespace or control characters")
    try:
        parts = urlsplit(value)
    except ValueError as error:
        raise RequestAuditError("live observation URL is malformed") from error
    if (
        parts.scheme != "https"
        or parts.username is not None
        or parts.password is not None
        or parts.port is not None
        or parts.fragment
        or "//" in parts.path
        or any(part in {".", ".."} for part in parts.path.split("/"))
    ):
        raise RequestAuditError("live observation URL must be canonical HTTPS")
    if parts.netloc == "api.github.com":
        allowed = (
            parts.path.startswith("/repos/valkey-io/")
            or parts.path == "/search/issues"
            or parts.path == "/graphql"
        )
    elif parts.netloc == "github.com":
        allowed = parts.path.startswith("/valkey-io/") or parts.path.startswith(
            "/orgs/valkey-io/projects/"
        )
    else:
        allowed = False
    if not allowed:
        raise RequestAuditError("live observation URL is outside the Valkey GitHub allowlist")
    return value


def _validated_request_id(value: object) -> str:
    if not isinstance(value, str) or _REQUEST_ID.fullmatch(value) is None:
        raise RequestAuditError("request ID is malformed")
    return value


def _validated_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise RequestAuditError(f"{label} must be a canonical sha256 digest")
    return value


def _validated_owner(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestAuditError("request owner must be non-blank text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise RequestAuditError("request owner must be valid UTF-8") from error
    if len(encoded) > _MAX_OWNER_BYTES:
        raise RequestAuditError(f"request owner exceeds its {_MAX_OWNER_BYTES}-byte bound")
    if any(ord(character) < 33 or ord(character) == 127 for character in value):
        raise RequestAuditError("request owner contains whitespace or a control character")
    return value


def _validated_lease_duration(value: object) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_LEASE_SECONDS:
        raise RequestAuditError(
            f"lease duration must be from 1 through {_MAX_LEASE_SECONDS} seconds"
        )
    return value


def _validated_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or _TIMESTAMP.fullmatch(value) is None:
        raise RequestAuditError(f"{label} is malformed")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise RequestAuditError(f"{label} is malformed") from error
    if parsed.tzinfo != UTC:
        raise RequestAuditError(f"{label} is malformed")
    return parsed


def _lease_expiration(now: datetime, lease_duration_seconds: int) -> str:
    return (now + timedelta(seconds=lease_duration_seconds)).isoformat().replace("+00:00", "Z")


def _validate_store(store: object) -> None:
    required = ("get_request", "put_request_if_absent", "compare_and_swap_nonterminal")
    if store is None or any(not callable(getattr(store, method, None)) for method in required):
        raise RequestAuditError("request audit store does not implement conditional state")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"
