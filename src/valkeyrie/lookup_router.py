"""Let the model choose which lookups a question needs, from a closed catalog.

The keyword router decided both whether a question was about live state and which live object it
named, using hand-written term lists. That fails on phrasing: "release" routed live while
"released" did not, so "is 9.2 rc1 released?" abstained although the release existed. Every
phrasing outside the lists degrades, and the space of phrasings is unbounded, so adding terms as
examples arrive can never finish.

A model has no such gap. This module asks it one narrow question, which lookups to perform, and
accepts the answer only from a fixed catalog with every argument bounded exactly as the existing
query types bound theirs. The model widens phrasing coverage; it cannot invent a capability, reach
a repository outside the organization, or shape a request the transport would not already accept.

Everything downstream is unchanged: the chosen lookups are executed, the evidence is assembled and
content-addressed into the plan, and the answer turn runs under the same grounding contract. The
router decides what to look up, never what to claim.

Any failure here (a timeout, malformed output, an unknown lookup) returns None, and the caller
falls back to the keyword path. The router can therefore only add coverage, never remove it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from valkeyrie.live_github import (
    IssueQuery,
    LiveGitHubQuery,
    PullRequestQuery,
    ReleaseListQuery,
)

ROUTER_PROMPT_REVISION: Final = "lookup-router/1"

# The closed catalog. Adding an entry here is the ONLY way the model gains a capability.
LookupKind = Literal["corpus_search", "pull_request", "issue", "releases"]
# project_board is deliberately absent: the runtime holds no GitHub Projects credential, so that
# lookup can only fail closed, and offering it would turn roadmap questions into confident
# abstentions instead of a corpus answer. Add it when a credential exists.
_KINDS: Final[frozenset[str]] = frozenset({"corpus_search", "pull_request", "issue", "releases"})

# Same bounds the live query types enforce; a mismatch here would let the model shape a request
# the transport then refuses, which would fail closed but waste the call.
_REPOSITORY: Final = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_MAX_NUMBER: Final = 10_000_000
_MAX_LOOKUPS: Final = 4
_MAX_RESPONSE_BYTES: Final = 4096
_DEFAULT_REPOSITORY: Final = "valkey"

ROUTER_SYSTEM: Final = (
    "You decide which lookups answer a question about the Valkey project. You do not answer the "
    "question. Reply with one JSON object and nothing else.\n"
    "\n"
    "Available lookups:\n"
    '- {"kind":"corpus_search"}: the indexed corpus of Valkey repositories: source code, '
    "documentation, governance, contribution guides, client libraries, modules. Use for how "
    "something works, how to use a command, who maintains what, how to contribute.\n"
    '- {"kind":"pull_request","repository":"valkey","number":N}: one pull request by number. '
    "Use when a number is given and the asker wants its status, design, or whether it merged.\n"
    '- {"kind":"issue","repository":"valkey","number":N}: one issue by number.\n'
    '- {"kind":"releases","repository":"valkey"}: the most recent releases INCLUDING release '
    "candidates and other prereleases. Use for anything about what has shipped, what the latest "
    "or newest version is, whether an rc or a version exists.\n"
    "\n"
    "Rules:\n"
    "- Choose every lookup that would help; a question about a feature that may be unreleased "
    "wants both corpus_search and the relevant live lookup.\n"
    "- A bare number like #3853 could be an issue or a pull request; GitHub shares one number "
    'space, so choose "issue" for it unless the asker says pull request or PR.\n'
    '- Repository names are within the valkey-io organization; default to "valkey". Use another '
    "only when the question names it (valkey-glide, valkey-doc, valkey-py, and so on).\n"
    "- If the question is not about Valkey at all, or is a greeting, reply "
    '{"lookups":[]}.\n'
    "\n"
    'Reply format exactly: {"lookups":[...]}'
)


class LookupRouterError(ValueError):
    """The model's lookup choice could not be accepted."""


@dataclass(frozen=True)
class LookupPlan:
    """The accepted lookups for one question."""

    corpus_search: bool
    live: tuple[LiveGitHubQuery, ...]


Converse = Callable[[str, str], str]
"""(system, question) -> raw model text. Injected so the router owns no transport."""


def route_lookups(question: str, converse: Converse) -> LookupPlan | None:
    """Ask the model which lookups the question needs. None means fall back to keywords.

    Returns None rather than raising for every failure, because a routing failure must never
    remove a capability the keyword path already has.
    """
    if not isinstance(question, str) or not question.strip():
        return None
    try:
        raw = converse(ROUTER_SYSTEM, question)
    except Exception:
        return None
    try:
        return parse_lookup_plan(raw)
    except LookupRouterError:
        return None


def parse_lookup_plan(raw: object) -> LookupPlan:
    """Accept the model's reply only if every lookup is in the catalog and every argument bounded.

    Strictness here is the security boundary. The model's output is untrusted text: a lookup kind
    outside the catalog, a repository outside the organization, or an out-of-range number is
    refused rather than coerced, because coercion would let the output shape a request the author
    never intended.
    """
    if not isinstance(raw, str):
        raise LookupRouterError("router reply must be text")
    if len(raw.encode("utf-8")) > _MAX_RESPONSE_BYTES:
        raise LookupRouterError("router reply exceeds its byte bound")
    text = _strip_fence(raw.strip())
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise LookupRouterError("router reply is not JSON") from error
    if not isinstance(value, Mapping) or set(value) != {"lookups"}:
        raise LookupRouterError("router reply must be an object with exactly one key, lookups")
    lookups = value["lookups"]
    if not isinstance(lookups, Sequence) or isinstance(lookups, str):
        raise LookupRouterError("lookups must be an array")
    if len(lookups) > _MAX_LOOKUPS:
        raise LookupRouterError("too many lookups requested")

    corpus_search = False
    live: list[LiveGitHubQuery] = []
    seen: set[tuple[object, ...]] = set()
    for item in lookups:
        if not isinstance(item, Mapping):
            raise LookupRouterError("each lookup must be an object")
        kind = item.get("kind")
        if kind not in _KINDS:
            raise LookupRouterError("lookup kind is not in the catalog")
        key = _identity(kind, item)
        if key in seen:
            continue
        seen.add(key)
        if kind == "corpus_search":
            _only_keys(item, {"kind"})
            corpus_search = True
            continue
        repository = _repository(item)
        if kind == "pull_request":
            _only_keys(item, {"kind", "repository", "number"})
            live.append(PullRequestQuery(repository, _number(item)))
        elif kind == "issue":
            _only_keys(item, {"kind", "repository", "number"})
            live.append(IssueQuery(repository, _number(item)))
        elif kind == "releases":
            _only_keys(item, {"kind", "repository"})
            live.append(ReleaseListQuery(repository))
    return LookupPlan(corpus_search=corpus_search, live=tuple(live))


def _strip_fence(text: str) -> str:
    # Models sometimes wrap JSON in a markdown fence despite instructions. Only a fence that
    # encloses the whole reply is removed; anything else is left to fail JSON parsing honestly.
    if text.startswith("```") and text.endswith("```"):
        inner = text[3:-3]
        if inner.startswith("json"):
            inner = inner[4:]
        return inner.strip()
    return text


def _identity(kind: str, item: Mapping[str, object]) -> tuple[object, ...]:
    return (kind, item.get("repository"), item.get("number"))


def _only_keys(item: Mapping[str, object], allowed: set[str]) -> None:
    extra = set(item) - allowed
    if extra:
        raise LookupRouterError(f"lookup carries unsupported keys: {sorted(extra)}")


def _repository(item: Mapping[str, object]) -> str:
    repository = item.get("repository", _DEFAULT_REPOSITORY)
    if not isinstance(repository, str) or _REPOSITORY.fullmatch(repository) is None:
        raise LookupRouterError("lookup repository is malformed")
    if repository.startswith(".") or ".." in repository:
        raise LookupRouterError("lookup repository is malformed")
    return repository


def _number(item: Mapping[str, object]) -> int:
    number = item.get("number")
    if type(number) is not int or not 1 <= number <= _MAX_NUMBER:
        raise LookupRouterError("lookup number is outside its bound")
    return number
