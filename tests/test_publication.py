from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

from tests.helpers import load_yaml
from tests.test_generation import _bundle
from valkeyrie.generation import GenerationBundle
from valkeyrie.publication import (
    PublicationError,
    StoredObject,
    publish_generation,
    verify_sealed_generation,
)
from valkeyrie.retrieval_config import load_retrieval_config

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = cast(dict[str, Any], load_yaml(ROOT / "src/valkeyrie/schemas/contracts.schema.json"))


class MemoryPublicationStore:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.put_calls: list[str] = []
        self.race_keys: set[str] = set()

    def list_keys(self, prefix: str) -> tuple[str, ...]:
        return tuple(sorted(key for key in self.objects if key.startswith(prefix)))

    def head_object(self, key: str) -> StoredObject | None:
        value = self.objects.get(key)
        if value is None:
            return None
        return StoredObject(len(value[0]), dict(value[1]))

    def get_object(self, key: str) -> bytes:
        return self.objects[key][0]

    def put_object_if_absent(
        self,
        key: str,
        content: bytes,
        metadata: object,
    ) -> bool:
        self.put_calls.append(key)
        values = dict(cast(dict[str, str], metadata))
        if key in self.race_keys:
            self.race_keys.remove(key)
            self.objects[key] = (content, values)
            return False
        if key in self.objects:
            return False
        self.objects[key] = (content, values)
        return True


@pytest.fixture
def bundle() -> GenerationBundle:
    return _bundle(load_retrieval_config(ROOT / "retrieval-config.yaml"))


def _completion_key(bundle: GenerationBundle) -> str:
    return f"control/generations/{bundle.generation_id.removeprefix('sha256:')}/complete.json"


def test_fresh_publication_creates_manifest_then_completion_last_and_verifies(
    bundle: GenerationBundle,
) -> None:
    store = MemoryPublicationStore()

    result = publish_generation(store, bundle)

    assert result.generation_id == bundle.generation_id
    assert result.completion_key == _completion_key(bundle)
    assert result.object_count == len(store.objects)
    assert result.created_keys == tuple(store.put_calls)
    assert store.put_calls[-2:] == [bundle.manifest.object_key, result.completion_key]
    assert all(
        key.startswith(
            (
                f"kb-documents/generations/{bundle.generation_id.removeprefix('sha256:')}/",
                f"control/generations/{bundle.generation_id.removeprefix('sha256:')}/",
            )
        )
        for key in store.objects
    )
    completion = json.loads(store.objects[result.completion_key][0])
    assert completion == {
        "api_version": "valkeyrie.io/generation-completion/1",
        "generation_id": bundle.generation_id,
        "kind": "GenerationCompletion",
        "manifest_digest": bundle.manifest.digest,
        "published_object_count": result.object_count - 1,
    }
    Draft202012Validator(
        {
            "$schema": SCHEMA["$schema"],
            "$defs": SCHEMA["$defs"],
            "$ref": "#/$defs/generation_completion",
        }
    ).validate(completion)
    assert verify_sealed_generation(store, bundle) == replace(result, created_keys=())


def test_retry_creates_only_missing_objects_and_accepts_exact_existing(
    bundle: GenerationBundle,
) -> None:
    complete = MemoryPublicationStore()
    expected = publish_generation(complete, bundle)
    store = MemoryPublicationStore()
    first_key = expected.created_keys[0]
    store.objects[first_key] = complete.objects[first_key]

    result = publish_generation(store, bundle)

    assert first_key not in store.put_calls
    assert set(result.created_keys) == set(complete.objects) - {first_key}
    assert store.objects == complete.objects


def test_conditional_create_race_verifies_winner_without_overwrite(
    bundle: GenerationBundle,
) -> None:
    store = MemoryPublicationStore()
    generation = bundle.generation_id.removeprefix("sha256:")
    raced = f"kb-documents/generations/{generation}/{bundle.documents[0].object_key}"
    store.race_keys.add(raced)

    result = publish_generation(store, bundle)

    assert raced in store.put_calls
    assert raced not in result.created_keys
    verify_sealed_generation(store, bundle)


@pytest.mark.parametrize("corruption", ["size", "metadata", "content"])
def test_existing_object_mismatch_is_rejected_without_overwrite(
    bundle: GenerationBundle,
    corruption: str,
) -> None:
    complete = MemoryPublicationStore()
    expected = publish_generation(complete, bundle)
    store = MemoryPublicationStore()
    key = expected.created_keys[0]
    content, metadata = complete.objects[key]
    if corruption == "size":
        content += b"x"
    elif corruption == "metadata":
        metadata = {**metadata, "object-kind": "forged"}
    else:
        content = b"x" * len(content)
    store.objects[key] = (content, metadata)

    expected_mismatch = "digest" if corruption == "content" else corruption
    with pytest.raises(PublicationError, match=f"published object {expected_mismatch} mismatch"):
        publish_generation(store, bundle)
    assert store.put_calls == []


def test_existing_completion_marker_forces_read_only_full_verification(
    bundle: GenerationBundle,
) -> None:
    store = MemoryPublicationStore()
    first = publish_generation(store, bundle)
    store.put_calls.clear()

    second = publish_generation(store, bundle)

    assert second == replace(first, created_keys=())
    assert store.put_calls == []

    del store.objects[first.created_keys[0]]
    with pytest.raises(PublicationError, match="object set mismatch"):
        publish_generation(store, bundle)
    assert store.put_calls == []


def test_unexpected_generation_object_is_rejected_before_mutation(
    bundle: GenerationBundle,
) -> None:
    store = MemoryPublicationStore()
    generation = bundle.generation_id.removeprefix("sha256:")
    store.objects[f"control/generations/{generation}/unexpected.json"] = (
        b"{}",
        {},
    )

    with pytest.raises(PublicationError, match="unexpected"):
        publish_generation(store, bundle)
    assert store.put_calls == []


def test_invalid_bundle_is_rejected_before_store_mutation(bundle: GenerationBundle) -> None:
    store = MemoryPublicationStore()
    forged = replace(bundle, generation_id="sha256:" + "0" * 64)

    with pytest.raises(PublicationError, match="bundle verification failed"):
        publish_generation(store, forged)
    assert store.put_calls == []
