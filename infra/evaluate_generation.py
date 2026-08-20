"""Build one exact generation evaluation report from verified qualification evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from valkeyrie.evaluations import (
    EvaluationError,
    ModelRun,
    RetrievalResult,
    evaluate_candidate,
    load_evaluation_suite,
)
from valkeyrie.live_qualification import (
    LiveQualificationError,
    verify_live_qualification_artifacts,
)


class GenerationEvaluationError(RuntimeError):
    """Verified qualification evidence cannot produce an exact generation report."""


_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_EVIDENCE_BYTES = 96 * 1024 * 1024
_SELECTED_MODEL = "us.anthropic.claude-fable-5"


def build_generation_evaluation_report(
    root: Path,
    generation_id: str,
) -> Mapping[str, object]:
    """Evaluate ``generation_id`` using the exact selected qualification observations.

    Model and fixed retrieval observations are reused byte-for-byte from the freshly
    verified model qualification. Candidate activation still performs live,
    generation-filtered Bedrock retrieval smoke before changing the active pointer.
    """
    project_root = root.expanduser().resolve(strict=True)
    if not project_root.is_dir() or project_root.is_symlink():
        raise GenerationEvaluationError("project root must be a real directory")
    if _DIGEST.fullmatch(generation_id) is None:
        raise GenerationEvaluationError("generation ID is malformed")

    verified = verify_live_qualification_artifacts(project_root)
    selection = verified.selection_record
    if selection.get("selected_model_revision") != _SELECTED_MODEL:
        raise GenerationEvaluationError("verified qualification did not select required Fable")
    profile_revision = selection.get("selected_profile_revision")
    if not isinstance(profile_revision, str) or _DIGEST.fullmatch(profile_revision) is None:
        raise GenerationEvaluationError("selected profile revision is malformed")

    evidence_path = (
        project_root / "evals/reports/evidence" / f"{profile_revision.removeprefix('sha256:')}.json"
    )
    evidence = _load_json(evidence_path)
    if evidence.get("complete") is not True or evidence.get("pending") is not None:
        raise GenerationEvaluationError("selected qualification evidence is incomplete")
    profile = evidence.get("profile")
    if not isinstance(profile, Mapping) or profile.get("profile_revision") != profile_revision:
        raise GenerationEvaluationError("selected qualification evidence profile mismatches")

    observations = evidence.get("observations")
    if not isinstance(observations, list):
        raise GenerationEvaluationError("qualification observations are malformed")
    model_runs: list[ModelRun] = []
    for raw in observations:
        if not isinstance(raw, Mapping):
            raise GenerationEvaluationError("qualification observation is malformed")
        case_id = raw.get("case_id")
        run = raw.get("run")
        metrics = raw.get("metrics")
        cost = raw.get("cost_usd")
        if (
            not isinstance(case_id, str)
            or not isinstance(run, int)
            or isinstance(run, bool)
            or not isinstance(metrics, Mapping)
            or not isinstance(cost, str)
        ):
            raise GenerationEvaluationError("qualification observation fields are malformed")
        try:
            cost_value = float(cost)
        except ValueError as error:
            raise GenerationEvaluationError(
                "qualification observation cost is malformed"
            ) from error
        model_runs.append(
            ModelRun(
                case_id=case_id,
                run=run,
                metrics=dict(cast(Mapping[str, object], metrics)),
                grading_method="deterministic",
                grader_calls=0,
                cost_usd=cost_value,
            )
        )

    reuse = evidence.get("retrieval_reuse")
    if not isinstance(reuse, Mapping) or reuse.get("live_retrieval_performed") is not False:
        raise GenerationEvaluationError("qualification retrieval reuse identity is malformed")
    raw_results = reuse.get("results")
    if not isinstance(raw_results, list):
        raise GenerationEvaluationError("qualification retrieval results are malformed")
    retrieval_results: list[RetrievalResult] = []
    for raw in raw_results:
        if not isinstance(raw, Mapping):
            raise GenerationEvaluationError("qualification retrieval result is malformed")
        fixture_id = raw.get("fixture_id")
        metrics = raw.get("metrics")
        if not isinstance(fixture_id, str) or not isinstance(metrics, Mapping):
            raise GenerationEvaluationError("qualification retrieval result fields are malformed")
        retrieval_results.append(
            RetrievalResult(fixture_id, dict(cast(Mapping[str, object], metrics)))
        )

    started_at = evidence.get("started_at")
    completed_at = evidence.get("completed_at")
    if not isinstance(started_at, str) or not isinstance(completed_at, str):
        raise GenerationEvaluationError("qualification timestamps are malformed")
    suite = load_evaluation_suite(project_root)
    report = evaluate_candidate(
        suite,
        candidate_revision=generation_id,
        started_at=started_at,
        completed_at=completed_at,
        model_runs=model_runs,
        retrieval_results=retrieval_results,
    )
    return cast(Mapping[str, object], report)


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise GenerationEvaluationError("cannot read qualification evidence") from error
    if not content or len(content) > _MAX_EVIDENCE_BYTES:
        raise GenerationEvaluationError("qualification evidence is outside its byte bound")

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
        raise GenerationEvaluationError("qualification evidence is not canonical JSON") from error
    if not isinstance(value, Mapping):
        raise GenerationEvaluationError("qualification evidence root is malformed")
    return cast(Mapping[str, object], value)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a generation-bound evaluation report from verified qualification."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    values = parser.parse_args(arguments)
    try:
        report = build_generation_evaluation_report(values.root, values.generation_id)
        output = values.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(_canonical(report))
            stream.flush()
    except (
        EvaluationError,
        FileExistsError,
        GenerationEvaluationError,
        LiveQualificationError,
        OSError,
        ValueError,
    ) as error:
        print(f"generation evaluation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
