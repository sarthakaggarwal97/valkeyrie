"""Slack-triggered GitHub actions: a closed catalog, an operator allowlist, and an audit trail.

THE BOUNDARY. Valkeyrie's answer path is read-only and qualified, and this module is why it can
stay that way: commands are a separate path that runs in the local bot process, never in the
Lambda. The write token lives only in this process's environment, the Lambda's IAM and code are
untouched, and the read-side GitHub fetcher physically cannot make this request (it rejects any
POST that is not a GraphQL query). Nothing in a Slack message can name a repository, a workflow
file or an input key: a command selects a catalog entry, and the catalog is a reviewed file in
the repository.

A command is a mention whose text is the bare word "run", or "run" followed by the name of a
catalogued action; "run valkey-benchmark how?" names no action and is an ordinary question:

    @Valkeyrie run valkey-daily test_args="--single unit/type/hash --loops 3"
    @Valkeyrie run backport-sweep branch=8.1
    @Valkeyrie run                       <- lists what the asker may run
    @Valkeyrie run backport-sweep --dry  <- validates and shows the dispatch without sending it

"run" was chosen because a second token that is not a catalogued action falls through to ordinary
question answering, so "run down the list of eviction policies" still gets an answer.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

CATALOG_PATH = Path(__file__).with_name("actions.yaml")
AUDIT_PATH = Path(os.environ.get("VALKEYRIE_ACTIONS_AUDIT", "/tmp/valkeyrie-actions.jsonl"))
# The write credential. Deliberately a DIFFERENT variable from the read token, so granting one can
# never accidentally grant the other, and unset means every command answers "commands are off".
TOKEN_ENV = "VALKEYRIE_ACTIONS_TOKEN"

_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")


class CommandError(Exception):
    """Refused before anything was sent. The message is safe to show in Slack."""


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str
    repo: str
    workflow: str
    ref: str
    inputs: dict[str, dict[str, Any]]
    allowed_slack_users: tuple[str, ...]


@dataclass(frozen=True)
class Command:
    """A parsed, validated, authorized dispatch that has not happened yet."""

    action: ActionSpec
    inputs: dict[str, str]
    dry_run: bool
    requested_by: str


def load_catalog(path: Path = CATALOG_PATH) -> tuple[tuple[str, ...], dict[str, ActionSpec]]:
    """The operator list and the action catalog, fail-closed on any malformed entry."""
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise CommandError("the action catalog is malformed")
    operators = tuple(str(user) for user in document.get("operators") or ())
    actions: dict[str, ActionSpec] = {}
    for raw in document.get("actions") or ():
        name = str(raw.get("name", ""))
        repo = str(raw.get("repo", ""))
        if _NAME.fullmatch(name) is None:
            raise CommandError(f"catalog action name {name!r} is malformed")
        # The org boundary, enforced in code as well as stated in the catalog comment: a command
        # must not be able to touch valkey-io even if the catalog is edited carelessly.
        if repo.startswith("valkey-io/"):
            raise CommandError(f"catalog action {name} targets valkey-io, which commands may not")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise CommandError(f"catalog action {name} has a malformed repository")
        workflow = str(raw.get("workflow", ""))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.ya?ml", workflow):
            raise CommandError(f"catalog action {name} has a malformed workflow file")
        inputs = raw.get("inputs") or {}
        if not isinstance(inputs, dict):
            raise CommandError(f"catalog action {name} has malformed inputs")
        actions[name] = ActionSpec(
            name=name,
            description=str(raw.get("description", "")),
            repo=repo,
            workflow=workflow,
            ref=str(raw.get("ref", "main")),
            inputs={str(k): dict(v or {}) for k, v in inputs.items()},
            allowed_slack_users=tuple(str(u) for u in raw.get("allowed_slack_users") or ()),
        )
    return operators, actions


def match_command(question: str, path: Path = CATALOG_PATH) -> str | None:
    """The command text when the message is a command, else None (an ordinary question).

    Bare `run` asks what exists. `run <word> ...` is a command only when the word names a
    catalogued action; "run valkey-benchmark how?" is a question about running a tool and falls
    through to answering. This is what the module docstring promised and the code did not do.
    """
    stripped = question.strip()
    if stripped.lower() == "run":
        return ""
    if not stripped.lower().startswith("run "):
        return None
    rest = stripped[3:].strip()
    head, _, tail = rest.partition(" ")
    try:
        _, catalogued = load_catalog(path)
    except CommandError:
        return rest
    for name in catalogued:
        if name.lower() == head.lower():
            # The catalogue's spelling, so "RUN CI" reaches the case-sensitive parser as "ci".
            return f"{name} {tail}".strip()
    return None


def parse_command(text: str, requested_by: str, path: Path = CATALOG_PATH) -> Command | str:
    """A validated Command, or the help text when the command only asks what exists.

    Authorization happens HERE, before validation details leak: a non-operator learns that
    commands exist and who to ask, not which inputs an action takes.
    """
    operators, actions = load_catalog(path)
    if requested_by not in operators:
        raise CommandError(
            "Commands are limited to the configured operators. "
            "Ask in this thread if you need one run."
        )
    tokens = shlex.split(text) if text else []
    if not tokens:
        lines = [f"{spec.name}: {spec.description}" for spec in actions.values()]
        return "Available actions:\n" + "\n".join(f"  {line}" for line in lines)
    name, *rest = tokens
    spec = actions.get(name)
    if spec is None:
        raise CommandError(f"{name!r} is not a catalogued action. Bare `run` lists what exists.")
    if spec.allowed_slack_users and requested_by not in spec.allowed_slack_users:
        raise CommandError(f"{name} is limited to specific operators.")
    dry_run = False
    inputs: dict[str, str] = {}
    for token in rest:
        if token == "--dry":
            dry_run = True
            continue
        key, separator, value = token.partition("=")
        if not separator:
            raise CommandError(f"arguments are key=value; {token!r} is neither that nor --dry.")
        rule = spec.inputs.get(key)
        # A key the catalog does not spell out does not pass through: workflow inputs reach a
        # workflow that may interpolate them, so the catalog is the boundary, not the workflow.
        if rule is None:
            allowed = ", ".join(spec.inputs) or "none"
            raise CommandError(f"{name} takes no input {key!r} (allowed: {allowed}).")
        pattern = str(rule.get("pattern", r"^[A-Za-z0-9 ._/=-]{0,200}$"))
        if re.fullmatch(pattern, value) is None:
            raise CommandError(f"the value for {key!r} does not match its allowed pattern.")
        inputs[key] = value
    missing = [k for k, rule in spec.inputs.items() if rule.get("required") and k not in inputs]
    if missing:
        raise CommandError(f"{name} requires: {', '.join(missing)}")
    return Command(action=spec, inputs=inputs, dry_run=dry_run, requested_by=requested_by)


def execute(command: Command, token: str | None = None) -> str:
    """Dispatch the workflow, or describe what would be dispatched. Returns the Slack reply.

    The dispatch is written to the audit file BEFORE the request, so a crash mid-send leaves a
    record of intent rather than a mystery run.
    """
    spec = command.action
    described = f"{spec.repo} :: {spec.workflow} @ {spec.ref}" + (
        f" with {json.dumps(command.inputs)}" if command.inputs else ""
    )
    if command.dry_run:
        return f"Dry run only. This would dispatch {described}"
    token = token if token is not None else os.environ.get(TOKEN_ENV, "")
    if not token:
        return (
            f"Commands are configured but the dispatch credential ({TOKEN_ENV}) is not set in "
            "this bot process, so nothing was triggered."
        )
    _audit(command, described)
    request = urllib.request.Request(
        f"https://api.github.com/repos/{spec.repo}/actions/workflows/{spec.workflow}/dispatches",
        data=json.dumps({"ref": spec.ref, "inputs": command.inputs}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "valkeyrie-actions/0.1",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        detail = {
            401: "the dispatch credential was rejected",
            403: "the dispatch credential lacks Actions write on that repository",
            404: "the workflow or repository was not found (or the credential cannot see it)",
            422: "GitHub rejected the inputs or the ref",
        }.get(error.code, f"GitHub returned HTTP {error.code}")
        return f"Dispatch failed: {detail}. Nothing may have run; check the Actions page."
    if status != 204:
        return f"GitHub answered HTTP {status} instead of accepting the dispatch."
    return (
        f"Dispatched {described}\n"
        f"Runs: https://github.com/{spec.repo}/actions/workflows/{spec.workflow}"
    )


def _audit(command: Command, described: str) -> None:
    row = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "by": command.requested_by,
        "action": command.action.name,
        "dispatch": described,
    }
    try:
        with AUDIT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    except OSError:
        # The command still runs: the audit file is evidence, not authorization.
        pass
