import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from infra.evaluate_generation import (
    GenerationEvaluationError,
    build_generation_evaluation_report,
    main,
)
from valkeyrie.evaluations import load_evaluation_suite, verify_evaluation_report

ROOT = Path(__file__).resolve().parents[1]
GENERATION = "sha256:" + "6" * 64


def test_builds_deterministic_passing_report_for_exact_generation() -> None:
    first = build_generation_evaluation_report(ROOT, GENERATION)
    second = build_generation_evaluation_report(ROOT, GENERATION)

    assert first == second
    assert first["candidate_revision"] == GENERATION
    assert first["result"] == "pass"
    summary = cast(Mapping[str, object], first["summary"])
    assert summary["model_runs"] == 291
    assert summary["retrieval_fixtures"] == 8
    verify_evaluation_report(first, load_evaluation_suite(ROOT))


def test_rejects_malformed_generation_before_reading_evidence() -> None:
    with pytest.raises(GenerationEvaluationError, match="generation ID"):
        build_generation_evaluation_report(ROOT, "not-a-generation")


def test_cli_writes_once_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    arguments = [
        "--root",
        str(ROOT),
        "--generation-id",
        GENERATION,
        "--output",
        str(output),
    ]

    assert main(arguments) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report == build_generation_evaluation_report(ROOT, GENERATION)
    assert main(arguments) == 1
