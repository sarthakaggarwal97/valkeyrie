"""Guard the independently defined canonical-serialization and digest helpers.

Several modules define their own ``_canonical_json`` and ``_sha256`` because a
content-addressed identity should not depend on another subsystem's module.
That independence is deliberate: editing one module's serializer must not
silently re-hash every identity in the system.

The hazard is the opposite failure, silent drift. If one copy diverges, two
subsystems would compute incompatible identities for the same value and nothing
would notice. These tests pin the exact bytes and digests every copy must
produce, so any divergence fails here instead of corrupting stored identities.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import pytest

from infra import release_corpus
from valkeyrie import (
    application_runtime,
    deployed_evaluation,
    evaluations,
    evidence,
    generation,
    git_acquisition,
    live_github,
    live_qualification,
    normalization,
    promotion,
    publication,
    request_audit,
    structured,
)

# A value chosen to exercise key ordering, non-ASCII escaping, floats, booleans,
# null, and empty containers, which is where canonical encoders usually differ.
_VALUE = {
    "z": 1,
    "a": [True, False, None, "\u00fc", 1.5],
    "nested": {"k": "v"},
    "empty_object": {},
    "empty_array": [],
}
_EXPECTED_JSON = (
    b'{"a":[true,false,null,"\xc3\xbc",1.5],"empty_array":[],"empty_object":{},'
    b'"nested":{"k":"v"},"z":1}'
)
_EXPECTED_ABC = "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
_EXPECTED_EMPTY = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

_CANONICAL_JSON: tuple[tuple[str, Callable[[object], bytes]], ...] = (
    ("normalization", normalization._canonical_json),
    ("publication", publication._canonical_json),
    ("request_audit", request_audit._canonical_json),
    ("structured", structured._canonical_json),
)
_DIGESTS: tuple[tuple[str, Callable[[bytes], str]], ...] = (
    ("deployed_evaluation", deployed_evaluation._sha256),
    ("evidence", evidence._sha256),
    ("generation", generation._sha256),
    ("git_acquisition", git_acquisition._digest),
    ("live_qualification", live_qualification._sha256),
    ("normalization", normalization._sha256),
    ("request_audit", request_audit._sha256),
)


_CALENDAR_TIMESTAMP_COPIES: Final = (
    ("application_runtime", application_runtime._is_calendar_timestamp),
    ("evaluations", evaluations._is_calendar_timestamp),
    ("generation", generation._is_calendar_timestamp),
    ("live_github", live_github._is_calendar_timestamp),
    ("promotion", promotion._is_calendar_timestamp),
    ("release_corpus", release_corpus._is_calendar_timestamp),
    ("request_audit", request_audit._is_calendar_timestamp),
)


@pytest.mark.parametrize(("module", "encode"), _CANONICAL_JSON)
def test_every_canonical_json_copy_emits_the_pinned_bytes(
    module: str, encode: Callable[[object], bytes]
) -> None:
    assert encode(_VALUE) == _EXPECTED_JSON, f"{module} canonical JSON drifted"


@pytest.mark.parametrize(("module", "digest"), _DIGESTS)
def test_every_digest_copy_emits_the_pinned_identity(
    module: str, digest: Callable[[bytes], str]
) -> None:
    assert digest(b"abc") == _EXPECTED_ABC, f"{module} digest drifted"
    assert digest(b"") == _EXPECTED_EMPTY, f"{module} empty digest drifted"


def test_canonical_json_rejects_values_that_would_break_identity_stability() -> None:
    for module, encode in _CANONICAL_JSON:
        with pytest.raises(ValueError):
            encode({"nan": float("nan")})
        with pytest.raises(ValueError):
            encode({"inf": float("inf")})
        assert encode({"b": 1, "a": 2}) == encode({"a": 2, "b": 1}), (
            f"{module} is not order-independent"
        )


@pytest.mark.parametrize(("module", "validator"), _CALENDAR_TIMESTAMP_COPIES)
def test_every_calendar_timestamp_copy_rejects_impossible_instants(
    module: str, validator: Callable[[str], bool]
) -> None:
    """Each copy must reject what the shape regex admits, or one module drifts permissive.

    The regex pins digit layout only, so an impossible calendar date matches it. A module that
    keeps the regex alone would accept 2026-99-99T99:99:99Z and persist it.
    """
    assert validator("2026-09-17T22:00:00Z") is True, module
    assert validator("2026-09-17T22:00:00.123Z") is True, module
    for impossible in ("2026-99-99T99:99:99Z", "2026-02-30T00:00:00Z", "2026-13-01T00:00:00Z"):
        assert validator(impossible) is False, f"{module} accepted {impossible}"
