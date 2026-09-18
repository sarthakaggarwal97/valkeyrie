from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import cast

from tests.helpers import load_yaml

ROOT = Path(__file__).resolve().parents[1]
FULL_ACTION_PIN = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
EXPECTED_ACTIONS = {
    "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
    "actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020",
    "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
    "astral-sh/setup-uv@37802adc94f370d6bfd71619e3f0bf239e1f3b78",
}


def test_standard_project_and_governance_files_exist() -> None:
    required = (
        "LICENSE",
        "README.md",
        "CONTRIBUTING.md",
        "GOVERNANCE.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        ".github/CODEOWNERS",
        ".github/dependabot.yml",
        "docs/repository-governance.md",
        "docs/rfcs/valkeyrie-maintainer-rfc.md",
    )
    for relative in required:
        path = ROOT / relative
        assert path.is_file(), relative
        assert path.read_text(encoding="utf-8").strip(), relative


def test_codeowners_covers_sensitive_phase0_inputs() -> None:
    codeowners = (ROOT / ".github/CODEOWNERS").read_text(encoding="utf-8")
    for path in (
        "/PROTOTYPE_AUTHORIZATION.md",
        "/docs/rfcs/",
        "/sources.yaml",
        "/prompts/",
        "/evals/",
        "/infra/",
        "/.github/",
        "/GOVERNANCE.md",
        "/SECURITY.md",
    ):
        assert path in codeowners
    assert "@sarthakaggarwal97" in codeowners


def test_ci_has_explicit_read_only_permissions_and_immutable_actions() -> None:
    workflow = load_yaml(ROOT / ".github/workflows/ci.yml")
    assert workflow["permissions"] == {"contents": "read"}
    jobs = cast(dict[str, dict[str, object]], workflow["jobs"])
    assert set(jobs) == {"phase0"}
    assert jobs["phase0"]["name"] == "phase0"
    steps = cast(list[dict[str, object]], jobs["phase0"]["steps"])
    action_pins = [cast(str, step["uses"]) for step in steps if "uses" in step]
    assert set(action_pins) == EXPECTED_ACTIONS
    assert all(FULL_ACTION_PIN.fullmatch(pin) for pin in action_pins)
    checkout = next(
        step for step in steps if cast(str, step.get("uses", "")).startswith("actions/checkout@")
    )
    assert cast(dict[str, object], checkout["with"])["persist-credentials"] is False
    assert all("permissions" not in job for job in jobs.values())

    def contains_write(value: object) -> bool:
        if isinstance(value, dict):
            return any(contains_write(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_write(item) for item in value)
        return value == "write"

    assert not contains_write(workflow)
    workflow_text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for forbidden in (
        "id-token: write",
        "contents: write",
        "actions: write",
        "checks: write",
        "issues: write",
        "pull-requests: write",
    ):
        assert forbidden not in workflow_text


def test_dependabot_is_bounded_to_reviewed_ecosystems() -> None:
    document = load_yaml(ROOT / ".github/dependabot.yml")
    updates = cast(list[dict[str, object]], document["updates"])
    assert {entry["package-ecosystem"] for entry in updates} == {"pip", "github-actions"}
    assert all(cast(dict[str, str], entry["schedule"])["interval"] == "weekly" for entry in updates)
    assert all(cast(int, entry["open-pull-requests-limit"]) <= 5 for entry in updates)


def test_all_direct_dependencies_and_build_tools_are_exactly_pinned() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    runtime = cast(list[str], project["project"]["dependencies"])
    development = cast(list[str], project["dependency-groups"]["dev"])
    build = cast(list[str], project["build-system"]["requires"])
    assert runtime and development and build
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^=<>~!]+", item) for item in runtime)
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^=<>~!]+", item) for item in development)
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^=<>~!]+", item) for item in build)

    def normalized(requirement: str) -> str:
        return requirement.split("==", 1)[0].lower().replace("_", "-")

    assert {normalized(item) for item in build} <= {normalized(item) for item in development}
    assert runtime == [
        "aws-cdk-lib==2.265.0",
        "jsonschema==4.26.0",
        "pathspec==1.1.1",
        "PyYAML==6.0.3",
    ]


def test_phase0_executable_paths_have_no_remote_mutation_commands() -> None:
    paths = [
        ROOT / "Makefile",
        ROOT / "infra/app.py",
        ROOT / "src/valkeyrie/__init__.py",
        ROOT / ".github/workflows/ci.yml",
    ]
    executable_text = "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()
    forbidden_patterns = (
        r"cdk\s+deploy",
        r"aws\s+cloudformation\s+(create|update|deploy|delete)",
        r"gh\s+api.*-(x|f)\s+(post|put|patch|delete)",
        r"chat\.postmessage",
        r"workflow_dispatch",
        r"terraform\s+apply",
    )
    for pattern in forbidden_patterns:
        assert re.search(pattern, executable_text) is None, pattern
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "uv export --locked --all-groups --no-emit-project" in makefile
    assert "--no-dev" not in makefile


def test_remote_governance_is_explicitly_proposed_not_applied() -> None:
    governance = (ROOT / "docs/repository-governance.md").read_text(encoding="utf-8")
    normalized = " ".join(governance.split())
    assert "Reviewed local proposal; no remote setting is created or changed" in normalized
    assert "`application`" in governance
    assert "`corpus`" in governance
    # The corpus environment is live and the document must say so; the application environment
    # is still reserved. The separation invariant is unchanged by either.
    assert "`corpus`" in governance and "Configured and in use" in normalized
    assert "`application`" in governance and "not configured or used" in normalized
    assert "repo:sarthakaggarwal97/valkeyrie:environment:corpus" in normalized
    assert "one workflow or role must never combine both responsibilities" in normalized
