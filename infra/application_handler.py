"""Private Lambda entrypoint for one immutable qualified application artifact."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from valkeyrie.application_runtime import AwsRuntimeServices, RuntimeServices, run_runtime_event

_MANIFEST = Path(__file__).resolve().parents[1] / "application-artifact.json"
_ROOT = Path(__file__).resolve().parents[1]


def handler(
    event: object,
    _context: object,
    services: RuntimeServices | None = None,
) -> dict[str, object]:
    """Run one private health or answer request against the packaged identity."""
    try:
        value = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("application artifact manifest is unavailable") from error
    if not isinstance(value, dict):
        raise RuntimeError("application artifact manifest is malformed")
    dependencies = services
    if dependencies is None and isinstance(event, Mapping) and event.get("action") == "answer":
        dependencies = AwsRuntimeServices()
    return run_runtime_event(
        event, dependencies, root=_ROOT, manifest=cast(dict[str, object], value)
    )
