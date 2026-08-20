"""Fail-closed loading and path classification for the reviewed source inventory."""

from __future__ import annotations

import re
from collections.abc import Hashable, Mapping
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import pathspec
import yaml
from jsonschema import Draft202012Validator, FormatChecker
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver


class SourceInventoryError(ValueError):
    """The source inventory is malformed or internally inconsistent."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""

    reject_merge_keys = False


_UniqueKeyLoader.yaml_implicit_resolvers = {
    key: [resolver for resolver in resolvers if resolver[0] != "tag:yaml.org,2002:bool"]
    for key, resolvers in _UniqueKeyLoader.yaml_implicit_resolvers.items()
}
_UniqueKeyLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
)


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: MappingNode, deep: bool = False
) -> dict[object, object]:
    seen: set[object] = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            if loader.reject_merge_keys:
                raise ConstructorError(
                    "mapping key",
                    node.start_mark,
                    "merge keys are not allowed",
                    key_node.start_mark,
                )
            continue
        key = cast(object, loader.construct_object(key_node, deep=deep))
        if not isinstance(key, Hashable):
            raise ConstructorError(
                "mapping key", node.start_mark, "unhashable key", key_node.start_mark
            )
        if key in seen:
            raise ConstructorError(
                "mapping key", node.start_mark, f"duplicate key: {key!r}", key_node.start_mark
            )
        seen.add(key)
    loader.flatten_mapping(node)
    return cast(dict[object, object], yaml.SafeLoader.construct_mapping(loader, node, deep))


_UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _load_yaml_mapping(
    text: str, source: str, *, reject_merge_keys: bool = False
) -> dict[str, object]:
    loader = _UniqueKeyLoader(text)
    loader.reject_merge_keys = reject_merge_keys
    try:
        value = cast(object, loader.get_single_data())
    finally:
        loader.dispose()
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain one mapping with string keys")
    _require_string_mapping_keys(value, source)
    return cast(dict[str, object], value)


def _require_string_mapping_keys(value: object, source: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{source} contains a non-string mapping key: {key!r}")
            _require_string_mapping_keys(item, source)
    elif isinstance(value, list):
        for item in value:
            _require_string_mapping_keys(item, source)


def load_yaml_mapping(path: Path) -> dict[str, object]:
    """Load one duplicate-safe YAML mapping without applying a domain schema."""
    return _load_yaml_mapping(path.read_text(encoding="utf-8"), str(path))


_SCHEMA_BUNDLE = cast(
    dict[str, Any],
    _load_yaml_mapping(
        files("valkeyrie").joinpath("schemas", "contracts.schema.json").read_text(encoding="utf-8"),
        "packaged contracts schema",
    ),
)
_SOURCE_SCHEMA = {
    "$schema": _SCHEMA_BUNDLE["$schema"],
    "$defs": _SCHEMA_BUNDLE["$defs"],
    "$ref": "#/$defs/source_inventory",
}
Draft202012Validator.check_schema(_SOURCE_SCHEMA)
_SOURCE_VALIDATOR = Draft202012Validator(_SOURCE_SCHEMA, format_checker=FormatChecker())
_FORBIDDEN_REF = re.compile(r"[\x00-\x20\x7f~^:?*\[]")


def load_source_inventory(path: Path) -> dict[str, object]:
    """Load and validate one source inventory, wrapping parse failures consistently."""
    try:
        document = _load_yaml_mapping(
            path.read_text(encoding="utf-8"), str(path), reject_merge_keys=True
        )
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as error:
        raise SourceInventoryError(f"cannot load source inventory {path}: {error}") from error
    validate_source_inventory(document)
    return document


def validate_source_inventory(document: Mapping[str, object]) -> None:
    """Validate the shared schema and C-01 cross-field invariants."""
    errors = sorted(
        _SOURCE_VALIDATOR.iter_errors(document),
        key=lambda error: "/".join(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "$"
        raise SourceInventoryError(f"schema validation failed at {location}: {error.message}")

    hard_exclusions = cast(list[dict[str, object]], document["hard_exclusions"])
    for group in hard_exclusions:
        for pattern in cast(list[str], group["patterns"]):
            _validate_pattern(pattern, "hard_exclusions")

    policies = cast(dict[str, dict[str, object]], document["path_policies"])
    for name, policy in policies.items():
        _validate_policy(name, cast(list[dict[str, object]], policy["rules"]))

    repositories = cast(list[dict[str, object]], document["repositories"])
    _require_unique(repositories, "name", case_insensitive=True)
    _require_unique(repositories, "url", case_insensitive=True)
    for repository in repositories:
        _validate_repository(repository, policies)

    inventory = cast(dict[str, object], document["inventory"])
    expected_counts = {
        "active_nonfork_repositories": len(repositories),
        "curated_or_structured": sum(
            item["classification"] in {"curated", "structured_exact"} for item in repositories
        ),
        "live_only_or_excluded": sum(
            item["classification"] in {"live_only", "excluded"} for item in repositories
        ),
    }
    for field, expected in expected_counts.items():
        if inventory[field] != expected:
            raise SourceInventoryError(
                f"inventory.{field} is {inventory[field]!r}; computed value is {expected}"
            )


def classify_path(
    document: Mapping[str, object], repository_name: str, path: str
) -> Literal["include", "exclude"]:
    """Apply reviewed deny-overrides Git-ignore rules to one safe repository path."""
    candidate = PurePosixPath(path)
    if (
        not path
        or candidate.is_absolute()
        or ".." in candidate.parts
        or "\\" in path
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
    ):
        raise SourceInventoryError(f"unsafe repository path: {path!r}")

    repositories = cast(list[dict[str, object]], document["repositories"])
    entry = next((item for item in repositories if item["name"] == repository_name), None)
    if entry is None:
        raise SourceInventoryError(f"unknown repository: {repository_name}")

    if _matches(_hard_exclusion_patterns(document), path):
        return "exclude"
    policies = cast(dict[str, dict[str, object]], document["path_policies"])
    policy_name = cast(str, entry["path_policy"])
    if policy_name not in policies:
        raise SourceInventoryError(
            f"repository {repository_name} references unknown policy {policy_name}"
        )
    rules = cast(list[dict[str, object]], policies[policy_name]["rules"])
    exclusions = [
        pattern
        for rule in rules
        if rule["action"] == "exclude" and rule["patterns"] != ["**"]
        for pattern in cast(list[str], rule["patterns"])
    ]
    if _matches(exclusions, path):
        return "exclude"
    inclusions = [
        pattern
        for rule in rules
        if rule["action"] == "include"
        for pattern in cast(list[str], rule["patterns"])
    ]
    return "include" if _matches(inclusions, path) else "exclude"


def _validate_policy(name: str, rules: list[dict[str, object]]) -> None:
    if rules[-1]["action"] != "exclude" or rules[-1]["patterns"] != ["**"]:
        raise SourceInventoryError(f"path policy {name} must end with the ** exclusion")
    seen_include = False
    seen_patterns: dict[str, str] = {}
    for index, rule in enumerate(rules):
        action = cast(str, rule["action"])
        patterns = cast(list[str], rule["patterns"])
        fallback = index == len(rules) - 1
        if action == "exclude" and seen_include and not fallback:
            raise SourceInventoryError(
                f"path policy {name} places a specific exclusion after an inclusion"
            )
        for pattern in patterns:
            _validate_pattern(pattern, f"path policy {name}")
            if pattern in seen_patterns:
                raise SourceInventoryError(
                    f"path policy {name} repeats pattern {pattern!r} in "
                    f"{seen_patterns[pattern]} and {action} rules"
                )
            seen_patterns[pattern] = action
        seen_include = seen_include or action == "include"
    if name != "none" and not seen_include:
        raise SourceInventoryError(f"path policy {name} has no inclusion rule")


def _validate_pattern(pattern: str, owner: str) -> None:
    relative = PurePosixPath(pattern.removeprefix("/"))
    if (
        pattern.startswith("//")
        or ".." in relative.parts
        or "\\" in pattern
        or "\0" in pattern
        or "\n" in pattern
        or "\r" in pattern
        or pattern.startswith("!")
    ):
        raise SourceInventoryError(f"{owner} contains unsafe pattern {pattern!r}")
    try:
        compiled = pathspec.GitIgnoreSpec.from_lines([pattern]).patterns[0]
    except ValueError as error:
        raise SourceInventoryError(
            f"{owner} contains invalid pattern {pattern!r}: {error}"
        ) from error
    if compiled.include is None or compiled.regex is None:
        raise SourceInventoryError(f"{owner} contains no-op pattern {pattern!r}")


def _validate_repository(repository: Mapping[str, object], policies: Mapping[str, object]) -> None:
    name = cast(str, repository["name"])
    expected_url = f"https://github.com/valkey-io/{name}"
    if repository["url"] != expected_url:
        raise SourceInventoryError(f"repository {name} must use URL {expected_url}")
    policy = cast(str, repository["path_policy"])
    if policy not in policies:
        raise SourceInventoryError(f"repository {name} references unknown path policy {policy}")
    _validate_ref(name, cast(str, repository["requested_ref"]))

    classification = repository["classification"]
    authority = repository["authority"]
    ingestion_mode = repository["ingestion_mode"]
    if classification == "curated":
        valid = authority in {"canonical", "secondary"} and ingestion_mode == "documents"
        valid = valid and policy != "none"
    elif classification == "structured_exact":
        valid = authority == "structured" and ingestion_mode == "structured_records"
        valid = valid and policy != "none"
    else:
        valid = authority == "none" and ingestion_mode == "none" and policy == "none"
    if not valid:
        raise SourceInventoryError(
            f"repository {name} has incoherent classification, authority, ingestion mode, or policy"
        )


def _validate_ref(repository: str, ref: str) -> None:
    parts = ref.split("/")
    invalid = (
        len(ref) > 255
        or ref in {"", ".", "..", "@", "HEAD"}
        or ref.startswith(("/", "-"))
        or ref.endswith(("/", "."))
        or "//" in ref
        or ".." in ref
        or "@{" in ref
        or "\\" in ref
        or _FORBIDDEN_REF.search(ref) is not None
        or any(part.startswith(".") or part.endswith(".lock") for part in parts)
    )
    if invalid:
        raise SourceInventoryError(f"repository {repository} has unsafe requested_ref {ref!r}")


def _require_unique(
    repositories: list[dict[str, object]], field: str, *, case_insensitive: bool = False
) -> None:
    values = [cast(str, repository[field]) for repository in repositories]
    normalized = [value.casefold() for value in values] if case_insensitive else values
    if len(normalized) != len(set(normalized)):
        raise SourceInventoryError(f"repository {field} values must be unique")


def _hard_exclusion_patterns(document: Mapping[str, object]) -> list[str]:
    groups = cast(list[dict[str, object]], document["hard_exclusions"])
    return [pattern for group in groups for pattern in cast(list[str], group["patterns"])]


def _matches(patterns: list[str], path: str) -> bool:
    return bool(patterns) and pathspec.GitIgnoreSpec.from_lines(patterns).match_file(path)
