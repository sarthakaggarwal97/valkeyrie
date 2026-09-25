"""The command path is a write path, so these tests are about what it CANNOT do."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

# The same pattern test_slack_bot.py uses for the same reason: the module lives beside the bot
# script rather than in the package, so mypy has no stub for it and pytest resolves it by path.
actions = pytest.importorskip("actions")

OPERATOR = "U08UZUQ790R"


def test_only_messages_naming_the_run_word_are_commands() -> None:
    assert actions.match_command("run backport-sweep branch=8.1") == "backport-sweep branch=8.1"
    assert actions.match_command("  RUN ci") == "ci"
    assert actions.match_command("run") == ""
    # Everything else is a question, including questions that contain the word.
    assert actions.match_command("how do I run valkey in a container?") is None
    assert actions.match_command("rundown of eviction policies") is None


def test_a_non_operator_is_refused_before_learning_anything() -> None:
    """The refusal names who may run commands, not which inputs an action takes."""
    with pytest.raises(actions.CommandError) as refusal:
        actions.parse_command("backport-sweep branch=8.1", "U_SOMEONE_ELSE")
    assert "operators" in str(refusal.value)
    assert "branch" not in str(refusal.value)


def test_nothing_outside_the_catalog_can_be_named() -> None:
    """A Slack message selects a catalog entry; it cannot supply a repo, workflow or input key."""
    with pytest.raises(actions.CommandError, match="not a catalogued action"):
        actions.parse_command("delete-everything", OPERATOR)
    # An input key the catalog does not spell out does not pass through, because workflow inputs
    # reach a workflow that may interpolate them.
    with pytest.raises(actions.CommandError, match="takes no input"):
        actions.parse_command("ci evil=payload", OPERATOR)
    # An input VALUE is bounded by its catalogued pattern.
    with pytest.raises(actions.CommandError, match="allowed pattern"):
        actions.parse_command("backport-sweep branch='8.1; rm -rf /'", OPERATOR)
    assert isinstance(actions.parse_command("backport-sweep branch=8.1", OPERATOR), actions.Command)


def test_the_catalog_itself_cannot_target_the_org(tmp_path: Path) -> None:
    """Even a careless catalog edit cannot point a command at valkey-io."""
    bad = tmp_path / "actions.yaml"
    bad.write_text(
        "operators: [U1]\n"
        "actions:\n"
        "  - name: sneaky\n"
        "    description: no\n"
        "    repo: valkey-io/valkey\n"
        "    workflow: ci.yml\n"
    )
    with pytest.raises(actions.CommandError, match="valkey-io"):
        actions.load_catalog(bad)


def test_quoted_values_and_dry_run_parse_the_way_an_operator_types_them() -> None:
    parsed = actions.parse_command(
        'valkey-daily test_args="--single unit/type/hash --loops 3" --dry', OPERATOR
    )
    assert isinstance(parsed, actions.Command)
    assert parsed.inputs == {"test_args": "--single unit/type/hash --loops 3"}
    assert parsed.dry_run is True
    reply = actions.execute(parsed)
    assert reply.startswith("Dry run only.")
    assert "daily.yml" in reply


def test_bare_run_lists_the_catalog_for_an_operator() -> None:
    listing = actions.parse_command("", OPERATOR)
    assert isinstance(listing, str)
    assert "backport-sweep" in listing and "valkey-daily" in listing


def test_without_the_credential_nothing_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset token means commands are visibly off, not silently attempted."""
    monkeypatch.delenv(actions.TOKEN_ENV, raising=False)
    parsed = actions.parse_command("ci", OPERATOR)
    assert isinstance(parsed, actions.Command)
    reply = actions.execute(parsed)
    assert actions.TOKEN_ENV in reply and "nothing was triggered" in reply


def test_a_dispatch_is_audited_before_it_is_sent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A crash mid-send must leave a record of intent rather than a mystery run."""
    monkeypatch.setattr(actions, "AUDIT_PATH", tmp_path / "audit.jsonl")

    def boom(request: object, timeout: float) -> None:
        raise AssertionError("sent")

    monkeypatch.setattr(actions.urllib.request, "urlopen", boom)
    parsed = actions.parse_command("ci", OPERATOR)
    assert isinstance(parsed, actions.Command)
    with pytest.raises(AssertionError):
        actions.execute(parsed, token="test-token")
    rows = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert rows and rows[0]["action"] == "ci" and rows[0]["by"] == OPERATOR
