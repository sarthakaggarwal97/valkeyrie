from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import cast

import pytest

from valkeyrie.sources import (
    SourceInventoryError,
    classify_path,
    load_source_inventory,
    load_yaml_mapping,
    validate_source_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "sources.yaml"


def _inventory() -> dict[str, object]:
    return deepcopy(load_yaml_mapping(SOURCES))


def _repositories(document: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], document["repositories"])


def _policies(document: dict[str, object]) -> dict[str, dict[str, object]]:
    return cast(dict[str, dict[str, object]], document["path_policies"])


def test_reviewed_inventory_loads_and_classifies_fail_closed() -> None:
    document = load_source_inventory(SOURCES)
    assert len(_repositories(document)) == 46
    assert classify_path(document, "valkey", "src/server.c") == "include"
    assert classify_path(document, "valkey", "src/generated/server.c") == "exclude"
    assert classify_path(document, "valkey", "unreviewed/source.c") == "exclude"
    assert (
        classify_path(
            document,
            "valkey-glide",
            "examples/java/gradle/wrapper/gradle-wrapper.jar",
        )
        == "exclude"
    )
    assert classify_path(document, "valkey-glide", "examples/java/gradlew.bat") == "exclude"
    assert (
        classify_path(document, "valkey-java", "src/test/resources/truststore.jceks") == "exclude"
    )
    assert classify_path(document, "valkey-java", "src/test/resources/cert.pem") == "exclude"
    assert classify_path(document, "valkey-py", "tests/testdata/payload.csv.bz2") == "exclude"
    assert classify_path(document, "valkey-py", "tests/__init__.py") == "exclude"
    assert classify_path(document, "valkey-py", "tests/test_asyncio/__init__.py") == "exclude"
    assert classify_path(document, "valkey-py", "tests/test_graph_utils/__init__.py") == "exclude"
    with pytest.raises(SourceInventoryError, match="unknown repository"):
        classify_path(document, "unknown", "README.md")
    for unsafe_path in ("line\nbreak", "control\x7fpath"):
        with pytest.raises(SourceInventoryError, match="unsafe repository path"):
            classify_path(document, "valkey", unsafe_path)


def test_contract_schema_is_packaged_with_the_validator() -> None:
    schema = files("valkeyrie").joinpath("schemas", "contracts.schema.json")
    assert schema.is_file()
    assert '"source_inventory"' in schema.read_text(encoding="utf-8")


def test_source_loader_wraps_duplicate_yaml_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "sources.yaml"
    duplicate.write_text(
        "api_version: valkeyrie.io/sources/1\napi_version: unsafe\n", encoding="utf-8"
    )
    with pytest.raises(SourceInventoryError, match="duplicate key"):
        load_source_inventory(duplicate)


@pytest.mark.parametrize(
    ("yaml_text", "error"),
    [
        (
            "base: &base\n  name: inherited\nrepository:\n  <<: *base\n  name: override\n",
            "merge keys are not allowed",
        ),
        ("api_version: valkeyrie.io/sources/1\nnested:\n  1: value\n", "non-string"),
    ],
)
def test_source_loader_rejects_merge_and_non_string_keys(
    tmp_path: Path, yaml_text: str, error: str
) -> None:
    source = tmp_path / "sources.yaml"
    source.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(SourceInventoryError, match=error):
        load_source_inventory(source)


def test_source_schema_rejects_incompatible_or_missing_contract_fields() -> None:
    document = _inventory()
    document["api_version"] = "valkeyrie.io/sources/2"
    with pytest.raises(SourceInventoryError, match="schema validation failed"):
        validate_source_inventory(document)

    document = _inventory()
    excluded = next(
        repository
        for repository in _repositories(document)
        if repository["classification"] == "excluded"
    )
    del excluded["reason"]
    with pytest.raises(SourceInventoryError, match="schema validation failed"):
        validate_source_inventory(document)


def test_all_exclusion_reasons_must_be_non_blank() -> None:
    document = _inventory()
    groups = cast(list[dict[str, object]], document["hard_exclusions"])
    groups[0]["reason"] = "   "
    with pytest.raises(SourceInventoryError, match="schema validation failed"):
        validate_source_inventory(document)

    document = _inventory()
    rules = cast(list[dict[str, object]], _policies(document)["core_source"]["rules"])
    rules[0]["reason"] = "\t"
    with pytest.raises(SourceInventoryError, match="schema validation failed"):
        validate_source_inventory(document)


@pytest.mark.parametrize("unsafe_name", ["../evil", "Valkey"])
def test_repository_names_must_be_canonical_lowercase_identities(unsafe_name: str) -> None:
    document = _inventory()
    repository = _repositories(document)[0]
    repository["name"] = unsafe_name
    repository["url"] = f"https://github.com/valkey-io/{unsafe_name}"
    with pytest.raises(SourceInventoryError, match="schema validation failed"):
        validate_source_inventory(document)


def test_repository_names_and_urls_must_be_unique() -> None:
    document = _inventory()
    repositories = _repositories(document)
    repositories[1]["name"] = repositories[0]["name"]
    with pytest.raises(SourceInventoryError, match="name values must be unique"):
        validate_source_inventory(document)

    document = _inventory()
    repositories = _repositories(document)
    repositories[1]["url"] = repositories[0]["url"]
    with pytest.raises(SourceInventoryError, match="url values must be unique"):
        validate_source_inventory(document)


def test_repository_policy_and_inventory_counts_must_resolve() -> None:
    document = _inventory()
    _repositories(document)[0]["path_policy"] = "missing"
    with pytest.raises(SourceInventoryError, match="unknown path policy"):
        validate_source_inventory(document)

    document = _inventory()
    inventory = cast(dict[str, object], document["inventory"])
    inventory["active_nonfork_repositories"] = 45
    with pytest.raises(SourceInventoryError, match="computed value is 46"):
        validate_source_inventory(document)


def test_repository_classification_fields_must_be_coherent() -> None:
    document = _inventory()
    curated = next(
        repository
        for repository in _repositories(document)
        if repository["classification"] == "curated"
    )
    curated["authority"] = "none"
    with pytest.raises(SourceInventoryError, match="incoherent classification"):
        validate_source_inventory(document)


@pytest.mark.parametrize(
    "unsafe_ref",
    [
        "",
        "../main",
        "feature//x",
        ".hidden",
        "main.lock",
        "feature@{x}",
        "feature x",
        "feature\\x",
        "feature?x",
        "HEAD",
        "-unsafe",
    ],
)
def test_repository_refs_must_be_safe(unsafe_ref: str) -> None:
    document = _inventory()
    _repositories(document)[0]["requested_ref"] = unsafe_ref
    with pytest.raises(SourceInventoryError):
        validate_source_inventory(document)


def test_path_policies_require_one_final_fallback() -> None:
    document = _inventory()
    rules = cast(list[dict[str, object]], _policies(document)["core_source"]["rules"])
    rules.pop()
    with pytest.raises(SourceInventoryError, match=r"must end with the \*\* exclusion"):
        validate_source_inventory(document)


def test_path_policies_reject_specific_exclusions_after_inclusions() -> None:
    document = _inventory()
    rules = cast(list[dict[str, object]], _policies(document)["core_source"]["rules"])
    rules.insert(
        -1,
        {"action": "exclude", "patterns": ["private/**"], "reason": "unsafe_test_overlap"},
    )
    with pytest.raises(SourceInventoryError, match="specific exclusion after an inclusion"):
        validate_source_inventory(document)


def test_path_policies_reject_exact_pattern_conflicts() -> None:
    document = _inventory()
    rules = cast(list[dict[str, object]], _policies(document)["core_source"]["rules"])
    rules.insert(
        0,
        {"action": "exclude", "patterns": ["/README*"], "reason": "unsafe_test_overlap"},
    )
    with pytest.raises(SourceInventoryError, match=r"repeats pattern '/README\*'"):
        validate_source_inventory(document)


@pytest.mark.parametrize("unsafe_pattern", ["../README.md", "docs\\README.md", "!README.md"])
def test_path_policies_reject_unsafe_patterns(unsafe_pattern: str) -> None:
    document = _inventory()
    rules = cast(list[dict[str, object]], _policies(document)["core_source"]["rules"])
    patterns = cast(list[str], rules[0]["patterns"])
    patterns.append(unsafe_pattern)
    with pytest.raises(SourceInventoryError, match="unsafe pattern"):
        validate_source_inventory(document)


@pytest.mark.parametrize("no_op_pattern", ["# comment", "   ", "/"])
def test_path_policies_reject_no_op_patterns(no_op_pattern: str) -> None:
    document = _inventory()
    rules = cast(list[dict[str, object]], _policies(document)["core_source"]["rules"])
    patterns = cast(list[str], rules[0]["patterns"])
    patterns.append(no_op_pattern)
    with pytest.raises(SourceInventoryError, match="no-op pattern"):
        validate_source_inventory(document)


def test_valkey_doc_utf8_bom_topics_remain_reviewed_canonical_sources() -> None:
    document = load_source_inventory(SOURCES)
    paths = {
        "topics/geospatial.md",
        "topics/hashes.md",
        "topics/lists.md",
        "topics/sets.md",
        "topics/strings.md",
    }

    for path in paths:
        assert classify_path(document, "valkey-doc", path) == "include"
        assert classify_path(document, "valkey-glide-docs", path) == "include"
