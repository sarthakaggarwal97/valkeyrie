"""Local-only orchestration for one pinned, cited Valkeyrie answer.

This injected harness is not a public entrypoint. Existing modules remain the
sole owners of routing, retrieval, evidence, model selection, request pinning,
output acceptance, and citation rendering.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TextIO, TypeAlias, cast

from valkeyrie.answer_models import AnswerModelSelection
from valkeyrie.bedrock_response import BedrockTextResponse
from valkeyrie.drafting import (
    DraftedAnswer,
    DraftedClarification,
    DraftingError,
    ModelInvocation,
    accept_bedrock_response,
)
from valkeyrie.evaluations import EvaluationSuite
from valkeyrie.evidence import (
    DocumentExcerpt,
    EvidenceError,
    EvidencePackage,
    StructuredRecordEvidence,
    create_evidence_package,
)
from valkeyrie.generation import GenerationBundle, GenerationError, verify_generation_bundle
from valkeyrie.request_audit import (
    PinnedExecution,
    RequestAuditError,
    RequestAuditRecord,
    RequestAuditStore,
    claim_request,
    complete_request,
    recover_request,
    resolve_pinned_execution,
)
from valkeyrie.retrieval import (
    BedrockRetrievalClient,
    GenerationRegistry,
    RetrievalError,
    RetrievedChunk,
    pin_generation,
    retrieve_generation,
)
from valkeyrie.retrieval_config import FrozenRetrievalConfiguration
from valkeyrie.routing import (
    EvidenceStatus,
    QuestionRequest,
    Route,
    RouteDecision,
    RoutingError,
    VersionRequirement,
    resolve_evidence,
    route_question,
)
from valkeyrie.structured import (
    ExactIdentifier,
    ExactLookup,
    MissingIdentifierError,
    ReleaseArtifactIdentifier,
    StructuredRecord,
    StructuredRecordError,
)

LocalOutcome: TypeAlias = Literal["answer", "clarification", "abstention", "partial", "error"]


class GenerationBundleStore(Protocol):
    """Local immutable bundles plus the explicit version-to-generation pointer."""

    def available_versions(self) -> tuple[str, ...]: ...

    def generation_for_version(self, version_scope: str | None) -> GenerationBundle | None: ...

    def get_generation(self, generation_id: str) -> GenerationBundle | None: ...


class ModelBackend(Protocol):
    """Injected Bedrock boundary receiving one verified A-04 invocation."""

    def generate(self, invocation: ModelInvocation) -> BedrockTextResponse: ...


@dataclass(frozen=True)
class LocalDependencies:
    root: Path
    suite: EvaluationSuite
    reports: tuple[Mapping[str, object], ...]
    selection: AnswerModelSelection
    generation_store: GenerationBundleStore
    generation_registry: GenerationRegistry
    retrieval_client: BedrockRetrievalClient
    request_store: RequestAuditStore
    model: ModelBackend
    retrieval_config: FrozenRetrievalConfiguration
    knowledge_base_id: str
    application_revision: str


@dataclass(frozen=True)
class LocalRequest:
    """Deterministic request input; time and version choices are explicit."""

    request_id: str
    question: str
    version_requirement: VersionRequirement
    requested_version: str | None
    exact_identifier: ExactIdentifier | None
    owner: str
    now: str
    completed_at: str
    lease_duration_seconds: int


@dataclass(frozen=True)
class LocalClaim:
    claim_id: str
    text: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class LocalResult:
    """Bounded machine-readable result for the local harness."""

    outcome: LocalOutcome
    request_id: str
    route: Route | None = None
    version_scope: str | None = None
    message: str | None = None
    claims: tuple[LocalClaim, ...] = ()
    citations: tuple[str, ...] = ()
    generation_id: str | None = None
    prompt_revision: str | None = None
    application_revision: str | None = None
    model_revision: str | None = None
    inference_config_revision: str | None = None
    evidence_package_digest: str | None = None
    request_revision: int | None = None
    request_fence: int | None = None


class _Stop(Exception):
    def __init__(
        self,
        outcome: Literal["abstention", "partial"],
        message: str,
        *,
        route: Route | None = None,
        version_scope: str | None = None,
        generation_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.message = message
        self.route = route
        self.version_scope = version_scope
        self.generation_id = generation_id


_MAX_CLI_INPUT_BYTES = 64 * 1024
_REQUIRED_METADATA = frozenset(
    {
        "generation_id",
        "document_id",
        "repository",
        "path",
        "commit",
        "authority",
        "version_scope",
        "content_digest",
    }
)


def run_local_request(request: LocalRequest, dependencies: LocalDependencies) -> LocalResult:
    """Route and execute one local request without network or implicit current state."""
    request_id = request.request_id if isinstance(request, LocalRequest) else "req_invalid"
    if not isinstance(request, LocalRequest):
        return _error(request_id, "local request has the wrong runtime type")
    if not isinstance(dependencies, LocalDependencies):
        return _error(request_id, "local dependencies have the wrong runtime type")
    try:
        try:
            existing = dependencies.request_store.get_request(request_id)
        except Exception as error:
            raise _Stop("partial", "request audit dependency is unavailable") from error
        if existing is not None:
            return _execute_existing(request, dependencies, existing)
        decision = _route(request, dependencies)
        terminal = _routing_terminal(request_id, decision)
        if terminal is not None:
            return terminal
        if decision.routes[0] == "live_read":
            disposition = resolve_evidence(
                decision,
                EvidenceStatus(True, True, False),
            )
            if disposition.outcome == "answer":
                raise RoutingError("live route answered without an observation")
            raise _Stop(
                disposition.outcome,
                disposition.reason or "live evidence does not permit an answer",
                route="live_read",
            )
        bundle = _resolved_bundle(request_id, decision, dependencies)
        package, exact_lookup, exact_record = _prepare_evidence(
            request, decision, bundle, dependencies
        )
        disposition = resolve_evidence(
            decision,
            EvidenceStatus(
                True,
                True,
                False,
                generation=bundle,
                package=package,
                exact_lookup=exact_lookup,
                exact_record=exact_record,
            ),
        )
        if disposition.outcome != "answer":
            raise _Stop(
                disposition.outcome,
                disposition.reason or "evidence does not permit an answer",
                route=decision.routes[0],
                version_scope=decision.version_scope,
                generation_id=bundle.generation_id,
            )
        try:
            claim = claim_request(
                dependencies.request_store,
                dependencies.root,
                dependencies.suite,
                dependencies.reports,
                dependencies.selection,
                bundle,
                package,
                request_id=request_id,
                question=request.question,
                application_revision=dependencies.application_revision,
                owner=request.owner,
                now=request.now,
                lease_duration_seconds=request.lease_duration_seconds,
            )
        except RequestAuditError:
            raise
        except Exception as error:
            raise _Stop("partial", "request claim dependency is unavailable") from error
        return _execute(
            request,
            dependencies,
            claim.record,
            decision.routes[0],
            decision.version_scope,
        )
    except _Stop as stop:
        return LocalResult(
            stop.outcome,
            request_id,
            route=stop.route,
            version_scope=stop.version_scope,
            message=stop.message,
            generation_id=stop.generation_id,
        )
    except (RoutingError, RequestAuditError, EvidenceError, StructuredRecordError) as error:
        return _error(request_id, str(error))


def local_result_value(result: LocalResult) -> dict[str, object]:
    """Return a fresh JSON-compatible representation."""
    if not isinstance(result, LocalResult):
        raise ValueError("local result has the wrong runtime type")
    return {
        "outcome": result.outcome,
        "request_id": result.request_id,
        "route": result.route,
        "version_scope": result.version_scope,
        "message": result.message,
        "claims": [
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "evidence_ids": list(claim.evidence_ids),
            }
            for claim in result.claims
        ],
        "citations": list(result.citations),
        "generation_id": result.generation_id,
        "prompt_revision": result.prompt_revision,
        "application_revision": result.application_revision,
        "model_revision": result.model_revision,
        "inference_config_revision": result.inference_config_revision,
        "evidence_package_digest": result.evidence_package_digest,
        "request_revision": result.request_revision,
        "request_fence": result.request_fence,
    }


def render_local_result(result: LocalResult) -> str:
    """Render without constructing or accepting any URL."""
    if result.outcome == "answer":
        lines = [claim.text for claim in result.claims]
        if result.citations:
            lines.extend(("", "Sources:", *result.citations))
        return "\n".join(lines)
    return f"{result.outcome.capitalize()}: {result.message or 'No additional detail.'}"


def main(
    argv: Sequence[str],
    stdin: TextIO,
    stdout: TextIO,
    dependencies: LocalDependencies,
) -> int:
    """Execute one local JSON request; no console entrypoint is installed."""
    if argv:
        result = _error("req_invalid", "the local harness accepts no command-line arguments")
    else:
        raw = stdin.read(_MAX_CLI_INPUT_BYTES + 1)
        if len(raw.encode("utf-8", errors="replace")) > _MAX_CLI_INPUT_BYTES:
            result = _error("req_invalid", "local CLI input exceeds its 65536-byte bound")
        else:
            try:
                request = _parse_request(json.loads(raw, object_pairs_hook=_unique_object))
            except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
                result = _error("req_invalid", f"invalid local CLI request: {error}")
            else:
                result = run_local_request(request, dependencies)
    stdout.write(
        json.dumps(
            local_result_value(result),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return int(result.outcome == "error")


def _route(request: LocalRequest, dependencies: LocalDependencies) -> RouteDecision:
    versions: tuple[str, ...] = ()
    if request.version_requirement != "current_state":
        try:
            versions = dependencies.generation_store.available_versions()
        except Exception as error:
            raise _Stop("partial", "generation version resolution is unavailable") from error
    return route_question(
        QuestionRequest(
            request.question,
            request.version_requirement,
            request.requested_version,
            versions,
            request.exact_identifier,
        )
    )


def _resolved_bundle(
    request_id: str,
    decision: RouteDecision,
    dependencies: LocalDependencies,
) -> GenerationBundle:
    del request_id
    try:
        value = dependencies.generation_store.generation_for_version(decision.version_scope)
    except Exception as error:
        raise _Stop(
            "partial",
            "generation bundle dependency is unavailable",
            route=decision.routes[0],
            version_scope=decision.version_scope,
        ) from error
    if value is None:
        raise _Stop(
            "partial",
            "the resolved generation bundle is unavailable",
            route=decision.routes[0],
            version_scope=decision.version_scope,
        )
    try:
        return verify_generation_bundle(value)
    except (GenerationError, TypeError, ValueError) as error:
        raise RoutingError(f"resolved generation is invalid: {error}") from error


def _prepare_evidence(
    request: LocalRequest,
    decision: RouteDecision,
    bundle: GenerationBundle,
    dependencies: LocalDependencies,
) -> tuple[EvidencePackage, ExactLookup | None, StructuredRecord | None]:
    route = decision.routes[0]
    try:
        pin = pin_generation(dependencies.generation_registry, bundle.generation_id)
    except RetrievalError as error:
        raise _Stop(
            "abstention",
            str(error),
            route=route,
            version_scope=decision.version_scope,
            generation_id=bundle.generation_id,
        ) from error
    except Exception as error:
        raise _Stop(
            "partial",
            "generation registry dependency is unavailable",
            route=route,
            version_scope=decision.version_scope,
            generation_id=bundle.generation_id,
        ) from error

    if route == "exact_lookup":
        try:
            return _exact_evidence(bundle, cast(ExactIdentifier, decision.exact_identifier))
        except MissingIdentifierError as error:
            raise _Stop(
                "abstention",
                "validated evidence does not support an answer",
                route=route,
                version_scope=decision.version_scope,
                generation_id=bundle.generation_id,
            ) from error
        except (EvidenceError, StructuredRecordError, ValueError) as error:
            raise _Stop(
                "abstention",
                f"exact evidence is unavailable: {error}",
                route=route,
                version_scope=decision.version_scope,
                generation_id=bundle.generation_id,
            ) from error

    try:
        chunks = retrieve_generation(
            dependencies.generation_registry,
            dependencies.retrieval_client,
            dependencies.retrieval_config,
            knowledge_base_id=dependencies.knowledge_base_id,
            generation=pin,
            query=request.question,
        )
    except RetrievalError as error:
        raise _Stop(
            "abstention",
            f"retrieval result was rejected: {error}",
            route=route,
            version_scope=decision.version_scope,
            generation_id=bundle.generation_id,
        ) from error
    except Exception as error:
        raise _Stop(
            "partial",
            "retrieval dependency is unavailable",
            route=route,
            version_scope=decision.version_scope,
            generation_id=bundle.generation_id,
        ) from error
    if not chunks:
        raise _Stop(
            "abstention",
            "validated evidence does not support an answer",
            route=route,
            version_scope=decision.version_scope,
            generation_id=bundle.generation_id,
        )
    try:
        return _retrieved_evidence(bundle, chunks), None, None
    except EvidenceError as error:
        raise _Stop(
            "abstention",
            f"retrieved evidence is unverifiable: {error}",
            route=route,
            version_scope=decision.version_scope,
            generation_id=bundle.generation_id,
        ) from error


def _execute_existing(
    request: LocalRequest,
    dependencies: LocalDependencies,
    existing: RequestAuditRecord,
) -> LocalResult:
    if existing.outcome is not None:
        return _error(request.request_id, "request audit is already terminal")
    claimed = existing
    if existing.owner != request.owner:
        try:
            claimed = recover_request(
                dependencies.request_store,
                existing,
                new_owner=request.owner,
                now=request.now,
                lease_duration_seconds=request.lease_duration_seconds,
            )
        except RequestAuditError as error:
            return LocalResult(
                "partial",
                request.request_id,
                message=f"request recovery is unavailable: {error}",
            )
        except Exception:
            return LocalResult(
                "partial",
                request.request_id,
                message="request recovery dependency is unavailable",
            )
    return _execute(request, dependencies, claimed, None, None)


def _execute(
    request: LocalRequest,
    dependencies: LocalDependencies,
    record: RequestAuditRecord,
    route: Route | None,
    version_scope: str | None,
) -> LocalResult:
    try:
        execution = resolve_pinned_execution(
            dependencies.request_store,
            request_id=request.request_id,
            question=request.question,
        )
    except RequestAuditError as error:
        return _error(request.request_id, f"pinned execution is invalid: {error}")
    except Exception:
        return LocalResult(
            "partial",
            request.request_id,
            route=route,
            version_scope=version_scope,
            message="pinned execution dependency is unavailable",
        )
    bundle = _bundle_for_execution(request, dependencies, execution, record, route, version_scope)
    if isinstance(bundle, LocalResult):
        return bundle
    try:
        value = dependencies.model.generate(execution.invocation)
    except Exception:
        return _execution_result(
            "partial",
            execution,
            record,
            route,
            version_scope,
            message="model dependency is unavailable",
        )
    try:
        accepted = accept_bedrock_response(value, bundle, execution.invocation.input.evidence)
    except DraftingError as error:
        message = f"model output was rejected: {error}"
        terminal = record
        try:
            terminal = complete_request(
                dependencies.request_store,
                record,
                outcome="error",
                completed_at=request.completed_at,
            )
        except RequestAuditError as completion_error:
            message += f"; request error completion failed: {completion_error}"
        except Exception:
            message += "; request error completion dependency is unavailable"
        return _execution_result(
            "error", execution, terminal, route, version_scope, message=message
        )
    outcome: Literal["answer", "clarification", "abstention"]
    outcome = (
        "answer"
        if isinstance(accepted, DraftedAnswer)
        else "clarification"
        if isinstance(accepted, DraftedClarification)
        else "abstention"
    )
    try:
        terminal = complete_request(
            dependencies.request_store,
            record,
            outcome=outcome,
            completed_at=request.completed_at,
        )
    except RequestAuditError as error:
        return _execution_result(
            "error",
            execution,
            record,
            route,
            version_scope,
            message=f"request completion failed: {error}",
        )
    except Exception:
        return _execution_result(
            "partial",
            execution,
            record,
            route,
            version_scope,
            message="request completion dependency is unavailable",
        )
    if isinstance(accepted, DraftedAnswer):
        return _execution_result(
            "answer",
            execution,
            terminal,
            route,
            version_scope,
            claims=tuple(
                LocalClaim(claim.claim_id, claim.text, claim.evidence_ids)
                for claim in accepted.claims
            ),
            citations=accepted.citations,
        )
    message = accepted.question if isinstance(accepted, DraftedClarification) else accepted.reason
    return _execution_result(outcome, execution, terminal, route, version_scope, message=message)


def _bundle_for_execution(
    request: LocalRequest,
    dependencies: LocalDependencies,
    execution: PinnedExecution,
    record: RequestAuditRecord,
    route: Route | None,
    version_scope: str | None,
) -> GenerationBundle | LocalResult:
    generation_id = execution.invocation.input.evidence.generation_id
    try:
        value = dependencies.generation_store.get_generation(generation_id)
    except Exception:
        return _execution_result(
            "partial",
            execution,
            record,
            route,
            version_scope,
            message="pinned generation dependency is unavailable",
        )
    if value is None:
        return _execution_result(
            "partial",
            execution,
            record,
            route,
            version_scope,
            message="pinned generation is unavailable",
        )
    try:
        bundle = verify_generation_bundle(value)
    except (GenerationError, TypeError, ValueError) as error:
        return _error(request.request_id, f"pinned generation is invalid: {error}")
    if bundle.generation_id != generation_id:
        return _error(request.request_id, "pinned generation store returned a substitution")
    return bundle


def _execution_result(
    outcome: LocalOutcome,
    execution: PinnedExecution,
    record: RequestAuditRecord,
    route: Route | None,
    version_scope: str | None,
    *,
    message: str | None = None,
    claims: tuple[LocalClaim, ...] = (),
    citations: tuple[str, ...] = (),
) -> LocalResult:
    invocation = execution.invocation
    return LocalResult(
        outcome,
        record.pin.request_id,
        route,
        version_scope,
        message,
        claims,
        citations,
        invocation.input.evidence.generation_id,
        invocation.prompt_revision,
        execution.application_revision,
        invocation.profile.model_revision,
        invocation.profile.inference_config_revision,
        invocation.input.evidence.digest,
        record.revision,
        record.fence,
    )


def _retrieved_evidence(
    bundle: GenerationBundle,
    chunks: tuple[RetrievedChunk, ...],
) -> EvidencePackage:
    documents = {document.document_id: document for document in bundle.document_templates}
    selections: list[DocumentExcerpt] = []
    seen: set[tuple[str, int, int]] = set()
    for chunk in chunks:
        if set(chunk.metadata) != _REQUIRED_METADATA:
            raise EvidenceError("retrieval metadata is not the exact reviewed field set")
        document_id = chunk.metadata.get("document_id")
        document = documents.get(document_id) if isinstance(document_id, str) else None
        if document is None:
            raise EvidenceError("retrieval metadata identifies an unknown document")
        source = document.source
        expected = {
            "generation_id": bundle.generation_id,
            "document_id": document.document_id,
            "repository": source.repository,
            "path": document.path,
            "commit": source.commit,
            "authority": source.authority,
            "version_scope": source.version_scope,
            "content_digest": document.content_digest,
        }
        if dict(chunk.metadata) != expected:
            raise EvidenceError("retrieval metadata does not match the canonical document")
        start = document.content.find(chunk.text)
        selection = (document.document_id, start, start + len(chunk.text))
        if start < 0 or document.content.find(chunk.text, start + 1) >= 0:
            raise EvidenceError("retrieved text is not one unique canonical document span")
        if selection in seen:
            raise EvidenceError("retrieval repeated one canonical document span")
        seen.add(selection)
        selections.append(DocumentExcerpt(*selection))
    return create_evidence_package(bundle, tuple(selections))


def _exact_evidence(
    bundle: GenerationBundle,
    identifier: ExactIdentifier,
) -> tuple[EvidencePackage, ExactLookup, StructuredRecord]:
    lookup = ExactLookup(bundle.structured_record_templates)
    record = lookup.lookup(identifier)
    record_id = next(
        canonical.object_id
        for template, canonical in zip(
            bundle.structured_record_templates,
            bundle.structured_records,
            strict=True,
        )
        if template == record
    )
    package = create_evidence_package(bundle, (StructuredRecordEvidence(record_id),))
    return package, lookup, record


def _routing_terminal(request_id: str, decision: RouteDecision) -> LocalResult | None:
    if decision.outcome == "clarification":
        return LocalResult("clarification", request_id, message=decision.question)
    if decision.outcome == "abstention":
        return LocalResult("abstention", request_id, message=decision.reason)
    return None


def _error(request_id: str, message: str) -> LocalResult:
    return LocalResult("error", request_id, message=message)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_request(value: object) -> LocalRequest:
    fields = {
        "request_id",
        "question",
        "version_requirement",
        "requested_version",
        "exact_identifier",
        "owner",
        "now",
        "completed_at",
        "lease_duration_seconds",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("request must have the exact local CLI field set")
    requirement = value["version_requirement"]
    if requirement not in {"none", "required", "current_state"}:
        raise ValueError("version_requirement is unsupported")
    requested = value["requested_version"]
    text_fields = ("request_id", "question", "owner", "now", "completed_at")
    if any(not isinstance(value[field], str) for field in text_fields):
        raise ValueError("request text fields must be strings")
    if requested is not None and not isinstance(requested, str):
        raise ValueError("requested_version must be text or null")
    lease = value["lease_duration_seconds"]
    if type(lease) is not int:
        raise ValueError("lease_duration_seconds must be an integer")
    return LocalRequest(
        cast(str, value["request_id"]),
        cast(str, value["question"]),
        cast(VersionRequirement, requirement),
        requested,
        _parse_exact_identifier(value["exact_identifier"]),
        cast(str, value["owner"]),
        cast(str, value["now"]),
        cast(str, value["completed_at"]),
        lease,
    )


def _parse_exact_identifier(value: object) -> ExactIdentifier | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"kind", "release", "artifact"}:
        raise ValueError("exact_identifier must be null or one release-artifact object")
    if value["kind"] != "release_artifact":
        raise ValueError("exact_identifier kind is unsupported")
    release, artifact = value["release"], value["artifact"]
    if not isinstance(release, str) or not isinstance(artifact, str):
        raise ValueError("release-artifact identifier fields must be strings")
    return ReleaseArtifactIdentifier(release, artifact)
