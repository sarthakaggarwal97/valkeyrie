#!/usr/bin/env python3
"""Run the exact A-02 live qualification after explicit operator invocation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

_REGION = "us-east-1"
_FABLE = "us.anthropic.claude-fable-5"
_NOVA = "amazon.nova-pro-v1:0"
_BOTO3_VERSION = "1.43.74"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed non-production A-02 Bedrock qualification suite."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--profile", default="valkeyrie-personal")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--archive-stale",
        choices=("dry-run", "apply"),
        help="Preview or apply archival of the authoritative stale qualification attempt.",
    )
    parser.add_argument(
        "--archive-status",
        choices=(
            "obsolete_prompt_and_inference_identities",
            "obsolete_response_normalization_policy_identity",
        ),
        default="obsolete_prompt_and_inference_identities",
    )
    arguments = parser.parse_args(argv)

    repository_root = Path(__file__).resolve().parents[1]
    source_root = str(repository_root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from valkeyrie.live_qualification import (
        LiveQualificationError,
        archive_stale_qualification_attempt,
        preview_stale_qualification_attempt,
        qualification_identity_from_aws,
        run_live_qualification,
    )

    if arguments.archive_stale is not None:
        if arguments.resume:
            parser.error("--resume cannot be combined with --archive-stale")
        try:
            manifest = preview_stale_qualification_attempt(
                arguments.root.resolve(), status=arguments.archive_status
            )
            archive_path = (
                arguments.root.resolve()
                / "evals/qualification-attempts"
                / cast(str, manifest["archive_id"]).removeprefix("sha256:")
            )
            if arguments.archive_stale == "apply":
                archive_path = archive_stale_qualification_attempt(
                    arguments.root.resolve(), status=arguments.archive_status
                )
        except (LiveQualificationError, OSError, ValueError) as error:
            print(f"qualification archive failed: {error}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "action": "archive_stale_qualification",
                    "applied": arguments.archive_stale == "apply",
                    "archive_path": str(archive_path),
                    "manifest": manifest,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0

    try:
        import boto3  # type: ignore[import-not-found]  # Operator supplies the exact batch SDK.

        if version("boto3") != _BOTO3_VERSION:
            raise LiveQualificationError(f"qualification requires boto3=={_BOTO3_VERSION}")
        session = boto3.Session(profile_name=arguments.profile, region_name=_REGION)
        caller = cast(dict[str, object], session.client("sts").get_caller_identity())
        bedrock = session.client("bedrock", region_name=_REGION)
        fable = cast(
            dict[str, object],
            bedrock.get_inference_profile(inferenceProfileIdentifier=_FABLE),
        )
        nova = cast(
            dict[str, object],
            bedrock.get_foundation_model(modelIdentifier=_NOVA),
        )
        identity = qualification_identity_from_aws(caller, fable, nova, region=_REGION)
        runtime: Any = session.client("bedrock-runtime", region_name=_REGION)
        result = run_live_qualification(
            arguments.root.resolve(),
            runtime,
            identity,
            monotonic=time.monotonic,
            clock=lambda: datetime.now(UTC),
            resume=arguments.resume,
        )
    except (LiveQualificationError, OSError, ValueError) as error:
        print(f"qualification failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result.selection_record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
