#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from valkeyrie.git_acquisition import (
    GitAcquisitionError,
    build_locked_corpus,
    create_source_lock,
    load_source_lock,
)
from valkeyrie.retrieval_config import RetrievalConfigError, load_retrieval_config
from valkeyrie.sources import SourceInventoryError, load_source_inventory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve and verify a complete immutable Valkeyrie corpus generation."
    )
    parser.add_argument("--root", type=Path, required=True, help="Valkeyrie project root")
    commands = parser.add_subparsers(dest="command", required=True)

    lock = commands.add_parser("lock", help="resolve every reviewed source exactly once")
    lock.add_argument("--output", type=Path, required=True)

    build = commands.add_parser("build", help="build and verify the complete locked generation")
    build.add_argument("--lock", type=Path, required=True)
    build.add_argument("--cache", type=Path, required=True)
    build.add_argument("--created-at", required=True)
    build.add_argument("--report", type=Path)
    return parser


def _write_once(path: Path, content: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise GitAcquisitionError(f"refusing to replace existing output {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


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


def _root(value: Path) -> Path:
    root = value.expanduser().resolve(strict=True)
    required = ("sources.yaml", "retrieval-config.yaml", "evals")
    if not root.is_dir() or any(not (root / item).exists() for item in required):
        raise GitAcquisitionError("operator root is not a Valkeyrie project")
    return root


def _lock(root: Path, output: Path) -> Mapping[str, object]:
    inventory = load_source_inventory(root / "sources.yaml")
    document = create_source_lock(inventory)
    locked = load_source_lock(inventory, document)
    _write_once(output, document)
    return {
        "operation": "lock",
        "lock_id": locked.lock_id,
        "source_policy_digest": locked.source_policy_digest,
        "repository_count": len(locked.revisions),
        "output": str(output.expanduser().resolve()),
    }


def _build(
    root: Path,
    lock_path: Path,
    cache: Path,
    created_at: str,
    report_path: Path | None,
) -> Mapping[str, object]:
    sources_yaml = (root / "sources.yaml").read_bytes()
    inventory = load_source_inventory(root / "sources.yaml")
    lock_document = lock_path.expanduser().resolve(strict=True).read_bytes()
    locked = load_source_lock(inventory, lock_document)
    retrieval = load_retrieval_config(root / "retrieval-config.yaml", project_root=root)
    bundle = build_locked_corpus(
        sources_yaml,
        inventory,
        retrieval,
        lock_document,
        cache.expanduser().resolve(),
        created_at=created_at,
    )
    result: dict[str, object] = {
        "operation": "build",
        "lock_id": locked.lock_id,
        "generation_id": bundle.generation_id,
        "manifest_digest": bundle.manifest.digest,
        "documents": len(bundle.documents),
        "metadata_sidecars": len(bundle.metadata_sidecars),
        "structured_records": len(bundle.structured_records),
        "source_commits": {item.repository: item.commit for item in locked.revisions},
    }
    if report_path is not None:
        _write_once(report_path, _canonical(result))
        result["report"] = str(report_path.expanduser().resolve())
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    parser = _parser()
    values = parser.parse_args(arguments)
    try:
        root = _root(cast(Path, values.root))
        if values.command == "lock":
            result = _lock(root, cast(Path, values.output))
        elif values.command == "build":
            result = _build(
                root,
                cast(Path, values.lock),
                cast(Path, values.cache),
                cast(str, values.created_at),
                cast(Path | None, values.report),
            )
        else:  # pragma: no cover - argparse owns the command set
            raise GitAcquisitionError("operator command is unsupported")
    except (
        GitAcquisitionError,
        RetrievalConfigError,
        SourceInventoryError,
        OSError,
        ValueError,
    ) as error:
        print(f"corpus operator failed: {error}", file=sys.stderr)
        return 1
    print(_canonical(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
