from __future__ import annotations

import io
import json
import shutil
import stat
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from infra.application_artifact import (
    _NON_RUNTIME_MODULES,
    _RUNTIME_MODULES,
    ApplicationArtifactError,
    _assert_target_abi,
    _revision,
    _validate_source_inventory,
    build_application_artifact,
    verify_application_artifact,
)
from tests.test_live_qualification import EXPECTED_IDENTITY, FakeConverse, Monotonic
from valkeyrie.evaluations import load_evaluation_suite
from valkeyrie.live_qualification import _load_cases, run_live_qualification

ROOT = Path(__file__).resolve().parents[1]


def _copy(destination: Path) -> Path:
    shutil.copytree(
        ROOT,
        destination,
        ignore=shutil.ignore_patterns(
            ".venv",
            ".git",
            "cdk.out",
            "cdk.application.out",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        ),
    )
    return destination


def _qualify_current(root: Path) -> dict[str, object]:
    shutil.rmtree(root / "evals/reports", ignore_errors=True)
    (root / "evals/model-selection.json").unlink(missing_ok=True)
    cases = _load_cases(root, load_evaluation_suite(root))
    result = run_live_qualification(
        root,
        FakeConverse(cases),
        EXPECTED_IDENTITY,
        monotonic=Monotonic(),
        clock=lambda: datetime(2026, 8, 20, 5, 0, tzinfo=UTC),
    )
    return cast(dict[str, object], result.selection_record)


@pytest.fixture(scope="module")
def qualified_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = _copy(tmp_path_factory.mktemp("artifact") / "qualified")
    _qualify_current(root)
    return root


def _rewrite(content: bytes, values: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(content)) as source, zipfile.ZipFile(output, "w") as target:
        for entry in source.infolist():
            target.writestr(entry, values.get(entry.filename, source.read(entry.filename)))
        for name, value in values.items():
            if name not in source.namelist():
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                target.writestr(info, value)
    return output.getvalue()


def test_runtime_and_non_runtime_source_inventory_is_exact() -> None:
    _validate_source_inventory(ROOT)
    assert "live_github.py" in _RUNTIME_MODULES
    assert "aws_adapters.py" not in _RUNTIME_MODULES
    assert "aws_adapters.py" in _NON_RUNTIME_MODULES
    assert "corpus.py" not in _RUNTIME_MODULES
    assert "corpus.py" in _NON_RUNTIME_MODULES
    assert "deployed_evaluation.py" not in _RUNTIME_MODULES
    assert "deployed_evaluation.py" in _NON_RUNTIME_MODULES
    assert "live_github.py" not in _NON_RUNTIME_MODULES


def test_artifact_is_deterministic_deployable_and_exactly_qualified(
    tmp_path: Path, qualified_root: Path
) -> None:
    selection = json.loads((qualified_root / "evals/model-selection.json").read_text())
    candidate_profiles = [
        candidate["profile_revision"].removeprefix("sha256:")
        for candidate in selection["candidate_reports"]
    ]
    expected_reports = [f"evals/reports/{profile}.json" for profile in candidate_profiles]
    expected_evidence = [f"evals/reports/evidence/{profile}.json" for profile in candidate_profiles]
    first = build_application_artifact(qualified_root)
    second = build_application_artifact(qualified_root)
    assert first == second
    manifest = verify_application_artifact(
        first.content, expected_application_revision=first.application_revision
    )
    assert manifest == first.manifest
    assert manifest["architecture"] == "x86_64"
    assert manifest["handler"] == "infra.application_handler.handler"
    assert manifest["selected_model_revision"] == "us.anthropic.claude-fable-5"
    for field in (
        "selected_profile_revision",
        "selected_inference_config_revision",
        "selected_report_id",
        "selection_id",
    ):
        assert manifest[field] == selection[field]
    assert manifest["evaluation_report_paths"] == expected_reports
    assert manifest["qualification_evidence_paths"] == expected_evidence
    assert manifest["model_selection_path"] == "evals/model-selection.json"
    assert set(cast(dict[str, str], manifest["raw_evidence_sha256"])) == set(expected_evidence)
    assert manifest["corpus_state_included"] is False

    with zipfile.ZipFile(io.BytesIO(first.content)) as archive:
        names = archive.namelist()
        assert "valkeyrie/application_runtime.py" in names
        assert "valkeyrie/live_github.py" in names
        assert "valkeyrie/corpus.py" not in names
        assert "valkeyrie/drafting.py" in names
        assert "infra/application_handler.py" in names
        assert "attr/__init__.py" in names
        assert "attrs/__init__.py" in names
        assert "yaml/_yaml.cpython-311-x86_64-linux-gnu.so" in names
        assert "rpds/rpds.cpython-311-x86_64-linux-gnu.so" in names
        assert not any(name.startswith("src/") for name in names)
        assert not any(
            "/__pycache__/" in name or "/tests/" in name or ".dist-info/" in name for name in names
        )
        assert not any(name.startswith("evals/qualification-attempts/") for name in names)
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.external_attr >> 16 == stat.S_IFREG | 0o644

        extract = tmp_path / "extract"
        archive.extractall(extract)
    command = (
        "import sys;"
        f"sys.path.insert(0,{str(extract)!r});"
        "import attr,attrs,yaml,jsonschema,jsonschema_specifications,pathspec;"
        "import referencing,rpds,typing_extensions;"
        "import valkeyrie.application_runtime,infra.application_handler;"
        "from pathlib import Path;"
        f"root=Path({str(extract)!r}).resolve();"
        "modules=(attr,attrs,yaml,jsonschema,jsonschema_specifications,pathspec,referencing,rpds,typing_extensions);"
        "assert all(Path(module.__file__).resolve().is_relative_to(root) for module in modules);"
        "print('isolated-import-ok')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", command],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated-import-ok"


@pytest.mark.parametrize(
    ("kind", "index"),
    [
        ("report", 0),
        ("report", 1),
        ("evidence", 0),
        ("evidence", 1),
        ("selection", 0),
    ],
)
def test_artifact_creation_rejects_any_missing_qualification_input(
    qualified_root: Path, tmp_path: Path, kind: str, index: int
) -> None:
    root = shutil.copytree(qualified_root, tmp_path / "copy")
    selection = json.loads((root / "evals/model-selection.json").read_text())
    profiles = [
        candidate["profile_revision"].removeprefix("sha256:")
        for candidate in selection["candidate_reports"]
    ]
    path = (
        root / "evals/model-selection.json"
        if kind == "selection"
        else root
        / "evals/reports"
        / ("evidence" if kind == "evidence" else "")
        / f"{profiles[index]}.json"
    )
    path.unlink()
    with pytest.raises(ApplicationArtifactError, match="qualification|input|read"):
        build_application_artifact(root)


def test_artifact_rejects_extra_and_stale_qualification_inputs(
    qualified_root: Path, tmp_path: Path
) -> None:
    root = shutil.copytree(qualified_root, tmp_path / "extra")
    (root / "evals/reports/extra.json").write_text("{}")
    with pytest.raises(ApplicationArtifactError, match="qualification"):
        build_application_artifact(root)

    root = shutil.copytree(qualified_root, tmp_path / "stale")
    selection = json.loads((root / "evals/model-selection.json").read_text())
    selection["selected_report_id"] = "eval_" + "0" * 64
    (root / "evals/model-selection.json").write_text(json.dumps(selection, separators=(",", ":")))
    with pytest.raises(ApplicationArtifactError, match="qualification|selected_report_id"):
        build_application_artifact(root)


def test_verifier_rejects_tamper_and_self_consistent_evil_path(
    qualified_root: Path,
) -> None:
    artifact = build_application_artifact(qualified_root)
    with zipfile.ZipFile(io.BytesIO(artifact.content)) as archive:
        manifest = json.loads(archive.read("application-artifact.json"))
    files = cast(list[dict[str, object]], manifest["files"])
    evil = b"raise SystemExit('evil')\n"
    files.append(
        {
            "path": "evil.py",
            "sha256": "sha256:" + __import__("hashlib").sha256(evil).hexdigest(),
            "size": len(evil),
            "mode": "0644",
        }
    )
    files.sort(key=lambda item: cast(str, item["path"]))
    preimage = dict(manifest)
    preimage.pop("application_revision")
    manifest["application_revision"] = _revision(preimage)
    forged = _rewrite(
        artifact.content,
        {
            "application-artifact.json": json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode(),
            "evil.py": evil,
        },
    )
    with pytest.raises(
        ApplicationArtifactError, match="paths are not canonical|unknown or missing path"
    ):
        verify_application_artifact(forged)

    tampered = _rewrite(artifact.content, {"valkeyrie/routing.py": b"tampered\n"})
    with pytest.raises(ApplicationArtifactError, match="tampered"):
        verify_application_artifact(tampered)


def test_verifier_requires_caller_expected_content_identity(qualified_root: Path) -> None:
    artifact = build_application_artifact(qualified_root)
    with pytest.raises(ApplicationArtifactError, match="expected content identity"):
        verify_application_artifact(
            artifact.content,
            expected_application_revision="sha256:" + "0" * 64,
        )


def test_vendored_native_extensions_must_match_the_declared_architecture() -> None:
    """A build on the wrong host must fail here, not as an ImportError after deployment.

    Dependencies are vendored from the host environment while the artifact and the function both
    declare x86_64 Linux, and the digest is computed over whatever bytes were copied. So a macOS
    arm64 build produces extensions the runtime cannot import and still verifies. The ELF header
    is read rather than the filename trusted, because the filename is what was already wrong.
    """
    # A 64-bit little-endian ELF for x86_64 (e_machine 0x3E) is the only accepted shape.
    linux_x86_64 = (
        b"\x7fELF\x02\x01\x01\x00"
        + bytes(8)
        + (2).to_bytes(2, "little")
        + (0x3E).to_bytes(2, "little")
    )
    _assert_target_abi("rpds/rpds.cpython-311-x86_64-linux-gnu.so", linux_x86_64)

    rejected = {
        # Darwin arm64 Mach-O, which is what an Apple silicon build produces.
        "mach-o": b"\xcf\xfa\xed\xfe\x0c\x00\x00\x01" + bytes(12),
        # Linux aarch64 ELF (e_machine 0xB7): right OS, wrong machine.
        "aarch64": b"\x7fELF\x02\x01\x01\x00"
        + bytes(8)
        + (2).to_bytes(2, "little")
        + (0xB7).to_bytes(2, "little"),
        # 32-bit ELF: right machine family, wrong class.
        "elf32": b"\x7fELF\x01\x01\x01\x00"
        + bytes(8)
        + (2).to_bytes(2, "little")
        + (0x03).to_bytes(2, "little"),
    }
    for label, content in rejected.items():
        with pytest.raises(ApplicationArtifactError, match="Linux x86_64"):
            _assert_target_abi("rpds/rpds.cpython-311-x86_64-linux-gnu.so", content)
        assert label

    # Pure-Python files carry no ABI and must pass through untouched.
    _assert_target_abi("yaml/loader.py", b"import os\n")
    _assert_target_abi("attrs/__init__.py", b"")
