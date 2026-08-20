"""Small helpers for checking reviewed data files."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import cast

from valkeyrie.sources import classify_path
from valkeyrie.sources import load_yaml_mapping as load_yaml

__all__ = ["classify_path", "compute_prompt_revision", "load_yaml"]


def compute_prompt_revision(document: Mapping[str, object], root: Path) -> str:
    """Independently hash canonical manifest semantics and exact reviewed prompt bytes."""
    prompt_root = (root / "prompts").resolve(strict=True)
    prompts = cast(list[dict[str, object]], document["prompts"])
    loaded: list[tuple[dict[str, object], bytes]] = []
    for prompt in sorted(prompts, key=lambda item: cast(str, item["file"])):
        name = cast(str, prompt["name"])
        relative = cast(str, prompt["file"])
        path = PurePosixPath(relative)
        candidate = root / path
        resolved = candidate.resolve(strict=True)
        current = root
        contains_symlink = False
        for part in path.parts:
            current /= part
            contains_symlink = contains_symlink or current.is_symlink()
        if (
            path.is_absolute()
            or ".." in path.parts
            or not relative.startswith("prompts/")
            or not resolved.is_relative_to(prompt_root)
            or contains_symlink
            or not resolved.is_file()
        ):
            raise ValueError(f"prompt path escapes prompts/ or uses a symlink: {relative}")
        content = resolved.read_bytes()
        if not content.strip():
            raise ValueError(f"prompt is empty: {relative}")
        loaded.append(({"name": name, "file": relative}, content))

    canonical = dict(document)
    revision = dict(cast(Mapping[str, object], canonical["revision"]))
    revision.pop("prompt_revision")
    canonical["revision"] = revision
    canonical["prompts"] = [entry for entry, _ in loaded]
    manifest_bytes = json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    digest = hashlib.sha256()
    for value in (
        b"valkeyrie-prompt-package/1",
        manifest_bytes,
        *(
            part
            for entry, content in loaded
            for part in (
                cast(str, entry["name"]).encode("utf-8"),
                cast(str, entry["file"]).encode("utf-8"),
                content,
            )
        ),
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return f"sha256:{digest.hexdigest()}"
