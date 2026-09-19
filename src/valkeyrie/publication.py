"""Resumable conditional publication and immutable generation sealing."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol

from valkeyrie.generation import (
    CanonicalObject,
    GenerationBundle,
    GenerationError,
    verify_generation_bundle,
)


class PublicationError(RuntimeError):
    """A generation cannot be published or verified without violating immutability."""


@dataclass(frozen=True)
class StoredObject:
    """Object metadata returned by a strongly consistent object-store head."""

    size: int
    metadata: Mapping[str, str]


class PublicationStore(Protocol):
    """Minimal S3 port; put-if-absent must use the exact ``If-None-Match: *`` condition."""

    def list_keys(self, prefix: str) -> tuple[str, ...]: ...

    def head_object(self, key: str) -> StoredObject | None: ...

    def get_object(self, key: str) -> bytes: ...

    def put_object_if_absent(
        self,
        key: str,
        content: bytes,
        metadata: Mapping[str, str],
    ) -> bool:
        """Create once and return true, or return false when the exact key already exists."""
        ...


@dataclass(frozen=True)
class PublicationResult:
    """Verified publication outcome for one sealed generation."""

    generation_id: str
    completion_key: str
    created_keys: tuple[str, ...]
    object_count: int


@dataclass(frozen=True)
class _PlannedObject:
    key: str
    content: bytes
    digest: str
    kind: str


_DIGEST: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMPLETION_API_VERSION: Final = "valkeyrie.io/generation-completion/1"


def publish_generation(store: PublicationStore, bundle: GenerationBundle) -> PublicationResult:
    """Resume publication and conditionally create the completion marker last."""
    verified = _verify_inputs(store, bundle)
    planned, completion = _publication_plan(verified)
    expected_without_completion = {item.key for item in planned}
    expected_all = expected_without_completion | {completion.key}
    observed = _listed_keys(store, verified.generation_id)

    if completion.key in observed:
        _verify_key_set(observed, expected_all)
        for item in (*planned, completion):
            _verify_object(store, verified.generation_id, item)
        return PublicationResult(
            verified.generation_id,
            completion.key,
            (),
            len(expected_all),
        )

    if not observed <= expected_without_completion:
        _raise_key_mismatch(observed, expected_without_completion)

    created: list[str] = []
    for item in planned:
        if _ensure_object(store, verified.generation_id, item):
            created.append(item.key)

    observed_before_seal = _listed_keys(store, verified.generation_id)
    if (
        not observed_before_seal <= expected_all
        or not expected_without_completion <= observed_before_seal
    ):
        _raise_key_mismatch(observed_before_seal, expected_without_completion)
    for item in planned:
        _verify_object(store, verified.generation_id, item)

    if _ensure_object(store, verified.generation_id, completion):
        created.append(completion.key)
    final_keys = _listed_keys(store, verified.generation_id)
    _verify_key_set(final_keys, expected_all)
    # Every planned object was byte-verified against memory moments ago, above, and the store is
    # write-once, so a second full re-read of all of them proved nothing new and cost as much as
    # the first: on the first manual refresh each pass was about an hour (7,600 objects, HEAD plus
    # GET each, sequentially from a runner). Only the completion marker, the one object written
    # since that verification, is read back.
    _verify_object(store, verified.generation_id, completion)
    return PublicationResult(
        verified.generation_id,
        completion.key,
        tuple(created),
        len(expected_all),
    )


def verify_sealed_generation(
    store: PublicationStore,
    bundle: GenerationBundle,
) -> PublicationResult:
    """Fully verify a sealed generation without making any mutation call."""
    verified = _verify_inputs(store, bundle)
    planned, completion = _publication_plan(verified)
    expected = {item.key for item in (*planned, completion)}
    observed = _listed_keys(store, verified.generation_id)
    _verify_key_set(observed, expected)
    if completion.key not in observed:
        raise PublicationError("generation completion marker is absent")
    for item in (*planned, completion):
        _verify_object(store, verified.generation_id, item)
    return PublicationResult(verified.generation_id, completion.key, (), len(expected))


def _verify_inputs(store: object, bundle: GenerationBundle) -> GenerationBundle:
    required = ("list_keys", "head_object", "get_object", "put_object_if_absent")
    if store is None or any(not callable(getattr(store, method, None)) for method in required):
        raise PublicationError(
            "publication store does not implement the required object operations"
        )
    try:
        return verify_generation_bundle(bundle)
    except GenerationError as error:
        raise PublicationError(f"generation bundle verification failed: {error}") from error


def _publication_plan(
    bundle: GenerationBundle,
) -> tuple[tuple[_PlannedObject, ...], _PlannedObject]:
    generation = bundle.generation_id.removeprefix("sha256:")
    knowledge_prefix = f"kb-documents/generations/{generation}/"
    control_prefix = f"control/generations/{generation}/"
    data: list[_PlannedObject] = []
    for document in bundle.documents:
        data.append(_planned(knowledge_prefix + document.object_key, document, "document"))
    for sidecar in bundle.metadata_sidecars:
        data.append(_planned(knowledge_prefix + sidecar.object_key, sidecar, "metadata"))
    for record in bundle.structured_records:
        data.append(_planned(control_prefix + record.object_key, record, "structured-record"))
    data.append(_planned(bundle.manifest.object_key, bundle.manifest, "manifest"))

    ordered = tuple(sorted(data, key=lambda item: (item.kind == "manifest", item.key)))
    completion_value = {
        "api_version": _COMPLETION_API_VERSION,
        "kind": "GenerationCompletion",
        "generation_id": bundle.generation_id,
        "manifest_digest": bundle.manifest.digest,
        "published_object_count": len(ordered),
    }
    content = _canonical_json(completion_value)
    completion = _PlannedObject(
        f"{control_prefix}complete.json",
        content,
        _digest(content),
        "completion",
    )
    return ordered, completion


def _planned(key: str, value: CanonicalObject, kind: str) -> _PlannedObject:
    if not key or key.startswith("/") or "//" in key or ".." in key.split("/"):
        raise PublicationError(f"unsafe publication key: {key}")
    if not _DIGEST.fullmatch(value.digest) or _digest(value.content) != value.digest:
        raise PublicationError(f"invalid canonical object digest for {key}")
    return _PlannedObject(key, value.content, value.digest, kind)


def _metadata(generation_id: str, item: _PlannedObject) -> dict[str, str]:
    return {
        "generation-id": generation_id,
        "object-digest": item.digest,
        "object-kind": item.kind,
    }


def _ensure_object(store: PublicationStore, generation_id: str, item: _PlannedObject) -> bool:
    existing = store.head_object(item.key)
    if existing is not None:
        _verify_object(store, generation_id, item, existing)
        return False
    created = store.put_object_if_absent(item.key, item.content, _metadata(generation_id, item))
    if not isinstance(created, bool):
        raise PublicationError("conditional object create returned a non-boolean result")
    if not created:
        _verify_object(store, generation_id, item)
    return created


def _verify_object(
    store: PublicationStore,
    generation_id: str,
    item: _PlannedObject,
    head: StoredObject | None = None,
) -> None:
    observed = head if head is not None else store.head_object(item.key)
    if observed is None:
        raise PublicationError(f"published object is missing: {item.key}")
    expected_metadata = _metadata(generation_id, item)
    if observed.size != len(item.content):
        raise PublicationError(f"published object size mismatch: {item.key}")
    if dict(observed.metadata) != expected_metadata:
        raise PublicationError(f"published object metadata mismatch: {item.key}")
    content = store.get_object(item.key)
    if not isinstance(content, bytes):
        raise PublicationError(f"published object content is not bytes: {item.key}")
    if _digest(content) != item.digest or content != item.content:
        raise PublicationError(f"published object digest mismatch: {item.key}")


def _listed_keys(store: PublicationStore, generation_id: str) -> set[str]:
    generation = generation_id.removeprefix("sha256:")
    keys = (
        *store.list_keys(f"kb-documents/generations/{generation}/"),
        *store.list_keys(f"control/generations/{generation}/"),
    )
    if any(not isinstance(key, str) or not key for key in keys):
        raise PublicationError("object listing returned a malformed key")
    if len(keys) != len(set(keys)):
        raise PublicationError("object listing returned duplicate keys")
    return set(keys)


def _verify_key_set(observed: set[str], expected: set[str]) -> None:
    if observed != expected:
        _raise_key_mismatch(observed, expected)


def _raise_key_mismatch(observed: set[str], expected: set[str]) -> None:
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    raise PublicationError(
        f"generation object set mismatch; missing={missing!r}; unexpected={unexpected!r}"
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"
