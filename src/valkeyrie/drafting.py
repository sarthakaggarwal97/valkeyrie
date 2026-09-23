"""Fail-closed model-invocation boundary and strict per-claim output acceptance.

The model-visible payload is exactly three inputs: exact reviewed prompts
loaded fail-closed from the prompt package, one bounded user question, and one
fully verified evidence package. The model target and inference metadata stay
outside that payload. The qualification candidate set is the approved
answer-model inventory, never a caller argument: the selection is recomputed
here from the complete exact evaluation suite, the inventory-constructed
candidate profiles, and exactly one live report per candidate, so a caller
cannot forge a selection or narrow the candidate set.

Accepted model output is a strict per-claim structure; every claim must pass
``validate_claim_support`` and every model-authored text must pass bounded
lexical prohibition screens. Those screens are lexical only: semantic
entailment between a claim and its cited evidence remains evaluation-owned and
is never established deterministically here. Citations are rendered by the
application alone from evidence IDs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Final, cast

from jsonschema import Draft202012Validator, FormatChecker

from valkeyrie.answer_models import (
    AnswerModelError,
    AnswerModelProfile,
    AnswerModelSelection,
    create_candidate_profiles,
    load_answer_model_inventory,
    select_answer_model,
)
from valkeyrie.bedrock_response import (
    BedrockResponseError,
    BedrockTextResponse,
    normalize_bedrock_response,
)
from valkeyrie.evaluations import EvaluationSuite
from valkeyrie.evidence import (
    ClaimSupport,
    EvidenceError,
    EvidenceLimits,
    EvidencePackage,
    render_citations,
    validate_claim_support,
    verify_evidence_package,
)
from valkeyrie.generation import GenerationBundle
from valkeyrie.prompts import PromptPackageError, PromptTemplate, load_prompt_package
from valkeyrie.sources import load_yaml_mapping


class DraftingError(ValueError):
    """A drafting input, selection binding, or model output is invalid."""


@dataclass(frozen=True)
class ModelInput:
    """The complete model-visible payload: exactly reviewed prompts, one bounded
    question, and one verified evidence package. Model target and inference
    metadata must never enter this payload."""

    prompts: tuple[PromptTemplate, ...]
    question: str
    evidence: EvidencePackage


@dataclass(frozen=True)
class ModelInvocation:
    """The model-visible input plus non-visible target and provenance metadata."""

    input: ModelInput
    profile: AnswerModelProfile
    prompt_revision: str


@dataclass(frozen=True)
class DraftClaim:
    """One accepted claim with its canonical supporting evidence IDs."""

    claim_id: str
    text: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class DraftedAnswer:
    """An accepted per-claim answer plus application-rendered citations."""

    claims: tuple[DraftClaim, ...]
    citations: tuple[str, ...]


@dataclass(frozen=True)
class DraftedClarification:
    """An accepted strict clarification containing one bounded question."""

    question: str


@dataclass(frozen=True)
class DraftedAbstention:
    """An accepted strict abstention containing one bounded reason."""

    reason: str


DraftedOutput = DraftedAnswer | DraftedClarification | DraftedAbstention

_DEFAULT_LIMITS = EvidenceLimits()
_INVENTORY_FILE: Final = "answer-models.yaml"
_MAX_QUESTION_BYTES: Final = 8 * 1024
_MAX_OUTPUT_BYTES: Final = 256 * 1024
_MAX_CLAIM_TEXT_BYTES: Final = 4 * 1024
_MAX_CLARIFICATION_BYTES: Final = 1024
_MAX_ABSTENTION_BYTES: Final = 2048

_READINESS_LABEL: Final = "a release-readiness decision"
# The decision named AND attributed to whoever owns it, in either order.
_READINESS_DEFERRAL: Final = re.compile(
    # "project" alone is not an authority: it appears in almost any sentence about Valkey.
    r"\b(?:decision|judgement|judgment|call|matter)\b[^.;:!?]*"
    r"\b(?:maintainers?|tsc|technical\s+steering\s+committee|release\s+owner)\b"
    r"|\b(?:maintainers?|tsc|technical\s+steering\s+committee|release\s+owner)\b[^.;:!?]*"
    r"\b(?:decide|decides|determined?|determines|decision|judgement|judgment|call)\b"
    # "... must be determined by the TSC": the verb precedes the authority.
    r"|\b(?:decided?|determined?|judged)\s+by\s+(?:the\s+)?"
    r"(?:maintainers?|tsc|technical\s+steering\s+committee|release\s+owner)\b",
    re.IGNORECASE,
)


_PROHIBITED_MODEL_TEXT: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("an evidence ID", re.compile(r"\bev_[a-z0-9-]+")),
    (
        "a link",
        re.compile(
            # A scheme, www., or any host with a path. Slack auto-links a bare domain, so a
            # model-authored one is as clickable as a real link and qualification counts it as a
            # fabricated citation. The last label must be a plausible public suffix, so
            # valkey.conf/foo stays a file path.
            r"\b(?:https?|ftp)://|\bwww\.|"
            r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*"
            r"\.(?:com|org|net|io|dev|sh|app|co|ai|me|info|edu|gov|cloud|xyz)/[a-z0-9]",
            re.IGNORECASE,
        ),
    ),
    (
        "a citation label",
        re.compile(
            # "[MATCH pattern] (with optional COUNT)" is command syntax and "c->argv[1]" is C
            # indexing; both were refused as citations. A Markdown link has no space before its
            # target, and a citation number stands alone rather than following an identifier.
            r"\[[^\]\n]*\]\(|(?<![A-Za-z0-9_\])])\[[0-9]+\]"
            r"|\b[a-z0-9.][a-z0-9._-]*/[^\s@]+@[0-9a-f]{7,40}\b"
        ),
    ),
    (
        "a source-authority declaration",
        re.compile(
            # "source of truth" is deliberately NOT here. The screen exists to stop Valkeyrie
            # asserting that its own evidence is authoritative; "src/commands/<cmd>.json is the
            # single source of truth for command metadata" is a fact about Valkey, taken from
            # Valkey's own README, and refusing it turned "how do I add a new command" into an
            # error every time, which is the most common onboarding question there is.
            # The subject must be THIS answer's evidence. "valkey-glide is the official client
            # library" and "the command JSON is the authoritative reference for arity" are facts
            # about Valkey, and "according to valkey.conf, the default is no" is ordinary
            # attribution: all three were refused, and the user lost the answer entirely.
            r"\b(?:this|the|my|our)\s+(?:supplied\s+|retrieved\s+|provided\s+|cited\s+)?"
            r"(?:evidence|context|sources?|documents?|records?)\b[^.;:!?]*"
            r"\b(?:canonical|authoritative|official|source\s+of\s+truth)\b"
            r"|\b(?:canonical|authoritative|official)\b[^.;:!?]*"
            r"\b(?:supplied|retrieved|provided|cited)\s+(?:evidence|context|sources?)\b"
            # "According to X" stays prohibited in every form. It is attribution prose rather than
            # a fact, and the same sentence reads better without it: "valkey.conf sets the default
            # to no" says more than "according to valkey.conf, the default is no".
            r"|\baccording\s+to\b"
            # The classic bare forms, where the authority IS the subject and no artifact is named.
            # "the authoritative reference FOR command arity" names one and is allowed.
            # Refused when the authority phrase ends the clause or attests something ("the
            # canonical source confirms this"), allowed when it goes on to say what the artifact
            # is ("the official documentation repository for Valkey commands").
            r"|\b(?:this|the)\s+(?:canonical|authoritative|official)\s+"
            r"(?:source|reference|documentation|authority)"
            r"\s*(?:[.,;:!?]|$"
            r"|\b(?:confirms?|states?|says?|shows?|indicates?|proves?)\b"
            # "The canonical source IS authoritative" asserts the authority outright.
            r"|\s*\b(?:is|are|was|were|remains?)\s+(?:the\s+)?"
            r"(?:canonical|authoritative|official)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        _READINESS_LABEL,
        re.compile(
            # A verdict needs a RELEASE subject. "The build is ready to accept connections",
            # "the lazy-free worker is ready to release memory", "valkey-go is a Go client" and
            # "Debian does not ship it by default" are not release decisions, and every one of
            # them was refused.
            r"\b(?:release|version|candidate)\s+is\s+(?:ready|approved)\b"
            r"|\b(?:is|are)\s+ready\s+(?:for\s+release\b|to\s+(?:release|ship|tag)\b(?!\s+\w))"
            r"|\bapprove(?:s|d)?\s+the\s+release\b"
            r"|\bgo\s*/?\s*no[- ]?go\b"
            # A bare go verdict, which the go/no-go alternative never matched: "this is a go for
            # the release" and "given the go-ahead" are decisions in the words a release owner uses.
            # Case-sensitive "a go": the Go language is capitalised, and "valkey-go is a Go
            # client library" was refused as a release verdict.
            r"|\bis\s+a\s+go\b(?![- ]client|[- ]library)"
            r"|\bgo[- ]ahead\b"
            # "Ship it" is the verdict; "does not ship it by default" is a packaging fact.
            r"|(?<!not\s)\bship\s+it\b(?!\s+by\s+default)"
            r"|\b(?:can|could|may)\s+(?:now\s+)?ship\b(?!\s+(?:its|their|the)\b)"
            # "safe to merge" about an iterator or a config is not a release decision, so the
            # sentence must be about a release for this to be one.
            r"|\bsafe\s+to\s+(?:release|ship|tag|deploy)\b"
            r"|\bsafe\s+to\s+merge\b(?=[^.;:!?]*\b(?:release|candidate|version\s+[0-9])\b)",
            re.IGNORECASE,
        ),
    ),
    (
        # This guards against the assistant claiming to have ACTED: "I merged it", "we deployed".
        # It deliberately does not match the passive voice. "The pull request was merged on
        # 2026-09-15" is a fact about GitHub reported from a live observation, and reporting that
        # fact is the live route's whole purpose. The earlier pattern forbade every passive form,
        # so a correct, evidence-backed answer about a merged pull request or a published release
        # was rejected as though the assistant had performed the merge itself, and the asker saw
        # "I couldn't produce a reliable answer" for a question with a known answer.
        "a completed project-state write",
        re.compile(
            # Subject, then any run of auxiliaries and adverbs (have, 've, did, just, already,
            # successfully, now...), then the action. Review showed "I've merged it", "I
            # successfully deployed it" and "We did publish the release" all slipping past a
            # fixed list of intervening words, so the gap is a bounded run of any short words.
            # The intervening words may not include a negation, a modal, or a recommending verb:
            # "I cannot merge", "we could merge", "I think it was merged", "we recommend you
            # merge" are not claims of having acted, and the assistant must be free to say them.
            r"\b(?:i|we)(?:'ve|'d)?"
            r"(?:\s+(?!(?:can|cannot|can't|could|couldn't|would|wouldn't|should|shouldn't|may|"
            r"might|must|will|won't|do|don't|not|never|think|believe|recommend|suggest|hope|"
            r"cannot)\b)[a-z]{2,12}){0,3}\s+"
            r"(?:merged|pushed|committed|deployed|released|tagged|published|closed|created)\b"
            # Emphatic past: "we did publish the release". "did" followed directly by the bare
            # verb is a claim of having acted; "did not" is excluded by the negation above being
            # required to sit between them.
            r"|\b(?:i|we)\s+did\s+(?:just\s+|already\s+|successfully\s+)?"
            r"(?:merge|push|commit|deploy|release|tag|publish|close|create)\b"
            r"|\b(?:i|we)(?:'ve)?(?:\s+(?!(?:can|cannot|could|would|should|not|never)\b)"
            r"[a-z]{2,12}){0,3}\s+(?:completed|finished)\s+the\s+"
            r"(?:deployment|merge|release|rollout|commit|push|publication|tag)\b",
            re.IGNORECASE,
        ),
    ),
)

_SCHEMA = cast(
    dict[str, Any],
    load_yaml_mapping(Path(str(files("valkeyrie").joinpath("schemas", "contracts.schema.json")))),
)
_OUTPUT_SCHEMA = {
    "$schema": _SCHEMA["$schema"],
    "$defs": _SCHEMA["$defs"],
    "$ref": "#/$defs/model_output",
}
Draft202012Validator.check_schema(_OUTPUT_SCHEMA)
_OUTPUT_VALIDATOR = Draft202012Validator(_OUTPUT_SCHEMA, format_checker=FormatChecker())


def prepare_model_invocation(
    root: Path,
    question: str,
    suite: EvaluationSuite,
    reports: tuple[Mapping[str, object], ...],
    selection: AnswerModelSelection,
    bundle: GenerationBundle,
    evidence: EvidencePackage,
    *,
    limits: EvidenceLimits = _DEFAULT_LIMITS,
) -> ModelInvocation:
    """Assemble the only model-visible payload plus target metadata, or fail closed.

    The candidate set is the approved answer-model inventory, never a caller
    argument. Candidate profiles are constructed from the inventory using the
    exact reviewed prompt revision, verified corpus generation, and evaluation
    suite revision; exactly one live evaluation report is required per
    inventory candidate, and omitted or extra candidates or reports fail. A
    caller-supplied selection that differs from the recomputed one is rejected.
    """
    _bounded_text(question, "user question", _MAX_QUESTION_BYTES)
    try:
        package = load_prompt_package(root)
    except PromptPackageError as error:
        raise DraftingError(f"reviewed prompt package is invalid: {error}") from error
    try:
        candidates = load_answer_model_inventory(root / _INVENTORY_FILE)
    except AnswerModelError as error:
        raise DraftingError(f"approved answer-model inventory is invalid: {error}") from error
    try:
        verified = verify_evidence_package(bundle, evidence, limits=limits)
    except EvidenceError as error:
        raise DraftingError(f"evidence package is not verified: {error}") from error
    try:
        profiles = create_candidate_profiles(
            candidates,
            prompt_revision=package.prompt_revision,
            corpus_generation=verified.generation_id,
            evaluation_suite_revision=suite.revision,
        )
        recomputed = select_answer_model(suite, profiles, reports)
    except AnswerModelError as error:
        raise DraftingError(f"answer-model qualification is invalid: {error}") from error
    if not isinstance(selection, AnswerModelSelection) or selection != recomputed:
        raise DraftingError("selection does not match the recomputed qualification")
    return ModelInvocation(
        input=ModelInput(prompts=package.templates, question=question, evidence=verified),
        profile=recomputed.profile,
        prompt_revision=package.prompt_revision,
    )


def accept_model_output(
    value: object,
    bundle: GenerationBundle,
    evidence: EvidencePackage,
    *,
    limits: EvidenceLimits = _DEFAULT_LIMITS,
) -> DraftedOutput:
    """Accept one strict model output or fail closed.

    An answer must use the per-claim structure ``[{claim_id, text,
    evidence_ids}]``; every claim must pass ``validate_claim_support`` and
    every model-authored text must pass the bounded lexical prohibition
    screens. The screens are lexical only and do not establish semantic
    entailment, which remains evaluation-owned. Citations are rendered by the
    application alone from validated evidence IDs.
    """
    document = _bounded_document(value)
    errors = sorted(
        _OUTPUT_VALIDATOR.iter_errors(document),
        key=lambda error: "/".join(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        where = "/".join(str(part) for part in error.absolute_path) or "$"
        raise DraftingError(f"model output schema validation failed at {where}: {error.message}")

    outcome = document["outcome"]
    if outcome == "clarification":
        question = cast(str, document["question"])
        _screened_model_text(question, "clarification question", _MAX_CLARIFICATION_BYTES)
        return DraftedClarification(question)
    if outcome == "abstention":
        reason = cast(str, document["reason"])
        _screened_model_text(reason, "abstention reason", _MAX_ABSTENTION_BYTES)
        return DraftedAbstention(reason)

    claims = cast(list[Mapping[str, object]], document["claims"])
    texts: dict[str, str] = {}
    order: list[str] = []
    supports: list[ClaimSupport] = []
    for claim in claims:
        claim_id = cast(str, claim["claim_id"])
        text = cast(str, claim["text"])
        if claim_id in texts:
            raise DraftingError(f"duplicate claim ID: {claim_id}")
        _screened_model_text(text, f"claim {claim_id!r} text", _MAX_CLAIM_TEXT_BYTES)
        evidence_ids = cast(list[str], claim["evidence_ids"])
        texts[claim_id] = text
        order.append(claim_id)
        supports.append(ClaimSupport(claim_id, tuple(evidence_ids)))
    try:
        canonical = validate_claim_support(bundle, evidence, tuple(supports), limits=limits)
        cited = tuple(
            sorted({evidence_id for support in canonical for evidence_id in support.evidence_ids})
        )
        citations = render_citations(bundle, evidence, cited, limits=limits)
    except EvidenceError as error:
        raise DraftingError(f"claim support is invalid: {error}") from error
    by_claim = {support.claim_id: support.evidence_ids for support in canonical}
    return DraftedAnswer(
        claims=tuple(
            DraftClaim(claim_id, texts[claim_id], by_claim[claim_id]) for claim_id in order
        ),
        citations=citations,
    )


def _bounded_document(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DraftingError("model output must be a mapping with string keys")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise DraftingError("model output is not canonically encodable JSON") from error
    if len(encoded) > _MAX_OUTPUT_BYTES:
        raise DraftingError(f"model output exceeds its {_MAX_OUTPUT_BYTES}-byte bound")
    return cast(Mapping[str, object], value)


def accept_bedrock_response(
    value: object,
    bundle: GenerationBundle,
    evidence: EvidencePackage,
    *,
    limits: EvidenceLimits = _DEFAULT_LIMITS,
) -> DraftedOutput:
    """Normalize a typed Bedrock final-text response before parsing any model text."""
    if not isinstance(value, BedrockTextResponse):
        raise DraftingError("Bedrock response has an unknown or missing field")
    try:
        normalized = normalize_bedrock_response(value.response_text, value.stop_reason)
    except BedrockResponseError as error:
        raise DraftingError(f"Bedrock response failed closed: {error.code}") from error
    try:
        document = json.loads(
            normalized.response_text,
            object_pairs_hook=_reject_json_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError, DraftingError) as error:
        raise DraftingError("normalized Bedrock response is invalid JSON") from error
    return accept_model_output(document, bundle, evidence, limits=limits)


def _reject_json_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise DraftingError(f"normalized Bedrock response contains duplicate key: {key}")
        value[key] = item
    return value


def _is_readiness_deferral(value: str) -> bool:
    """True only when EVERY readiness mention in the text is a deferral, clause by clause.

    Matching the two patterns across the whole claim let one sentence exempt another: "The
    maintainers make the call. Ship it." carried a deferral and a verdict, and the verdict rode
    in on the exemption. Each clause is now judged on its own.
    """
    label_pattern = dict((label, pattern) for label, pattern in _PROHIBITED_MODEL_TEXT)[
        _READINESS_LABEL
    ]
    for clause in re.split(r"[.;:!?\n]+", value):
        if label_pattern.search(clause) is None:
            continue
        if _READINESS_DEFERRAL.search(clause) is None:
            return False
    return True


def _screened_model_text(value: str, field: str, maximum: int) -> None:
    _bounded_text(value, field, maximum)
    for label, pattern in _PROHIBITED_MODEL_TEXT:
        if pattern.search(value) is None:
            continue
        # One exemption, for one rule. A readiness question may be answered with facts plus who
        # decides, and "whether X is ready to release is the maintainers' decision" is the
        # opposite of a verdict: it names readiness only to hand it to the people whose call it
        # is. Refusing that sentence left the answer with board totals and no statement of who
        # decides, which is the part a reader needs. Nothing else is exempt, and a sentence that
        # ASSERTS readiness cannot match this shape because the decision must be attributed.
        if label == _READINESS_LABEL and _is_readiness_deferral(value):
            continue
        raise DraftingError(f"{field} contains {label}")


def _bounded_text(value: object, field: str, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DraftingError(f"{field} must be non-blank text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise DraftingError(f"{field} must be valid UTF-8") from error
    if len(encoded) > maximum:
        raise DraftingError(f"{field} exceeds its {maximum}-byte bound")
    if any(
        (ord(character) < 32 and character not in "\t\n") or 127 <= ord(character) <= 159
        for character in value
    ):
        raise DraftingError(f"{field} contains a control character")
