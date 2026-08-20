from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from valkeyrie.github import HttpResponse
from valkeyrie.inventory import (
    InventoryChange,
    InventoryError,
    InventoryLimits,
    InventoryReconciliationError,
    MissingRepositoryProbe,
    OrganizationInventory,
    RepositorySnapshot,
    fetch_organization_inventory,
    reconcile_source_inventory,
)
from valkeyrie.sources import load_source_inventory

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "sources.yaml"
OBSERVED_AT = datetime(2026, 8, 18, 12, 34, 56, tzinfo=UTC)
OBSERVED_TEXT = "2026-08-18T12:34:56Z"
NEXT_PAGE = (
    "<https://api.github.com/organizations/164458127/repos"
    '?type=all&per_page=100&page=2>; rel="next"'
)


class StubFetcher:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, float, int]] = []

    def __call__(self, url: str, timeout_seconds: float, max_bytes: int) -> HttpResponse:
        self.calls.append((url, timeout_seconds, max_bytes))
        return self.responses[len(self.calls) - 1]


def _api_repository(
    name: str = "alpha",
    repository_id: int = 1,
    **overrides: object,
) -> dict[str, object]:
    value: dict[str, object] = {
        "id": repository_id,
        "name": name,
        "full_name": f"valkey-io/{name}",
        "html_url": f"https://github.com/valkey-io/{name}",
        "default_branch": "main",
        "fork": False,
        "archived": False,
        "disabled": False,
        "size": 10,
        "private": False,
        "visibility": "public",
        "owner": {"login": "valkey-io"},
    }
    value.update(overrides)
    return value


def _response(
    repositories: object,
    *,
    status: int = 200,
    link: str | None = None,
) -> HttpResponse:
    headers = {"Link": link} if link is not None else {}
    return HttpResponse(status, headers, json.dumps(repositories).encode())


def _source() -> dict[str, object]:
    return deepcopy(load_source_inventory(SOURCES))


def _snapshot_repository(
    name: str,
    repository_id: int,
    default_branch: str,
) -> RepositorySnapshot:
    return RepositorySnapshot(
        repository_id=repository_id,
        name=name,
        full_name=f"valkey-io/{name}",
        url=f"https://github.com/valkey-io/{name}",
        default_branch=default_branch,
        fork=False,
        archived=False,
        disabled=False,
        empty=False,
        size=10,
        private=False,
        visibility="public",
    )


def _reviewed_inventory() -> OrganizationInventory:
    source_repositories = cast(list[dict[str, object]], _source()["repositories"])
    repositories = tuple(
        _snapshot_repository(
            cast(str, repository["name"]),
            index,
            cast(str, repository["requested_ref"]),
        )
        for index, repository in enumerate(source_repositories, start=1)
    )
    return OrganizationInventory("valkey-io", OBSERVED_TEXT, repositories)


def _changed_inventory(
    baseline: OrganizationInventory,
    old: RepositorySnapshot,
    new: RepositorySnapshot | None,
) -> OrganizationInventory:
    repositories = list(baseline.repositories)
    index = repositories.index(old)
    if new is None:
        repositories.pop(index)
    else:
        repositories[index] = new
    return replace(baseline, repositories=tuple(repositories))


def _with_repository_state(repository: RepositorySnapshot, state: str) -> RepositorySnapshot:
    if state == "fork":
        return replace(repository, fork=True)
    if state == "archived":
        return replace(repository, archived=True)
    if state == "disabled":
        return replace(repository, disabled=True)
    raise AssertionError(f"unsupported test repository state: {state}")


def _only_change(
    error: pytest.ExceptionInfo[InventoryReconciliationError],
) -> InventoryChange:
    changes = error.value.report.changes
    assert len(changes) == 1
    return changes[0]


def test_fetches_one_complete_page_and_records_stable_public_state() -> None:
    fetcher = StubFetcher([_response([_api_repository(size=0), _api_repository("beta", 2)])])

    inventory = fetch_organization_inventory(fetch_page=fetcher, observed_at=OBSERVED_AT)

    assert inventory.organization == "valkey-io"
    assert inventory.observed_at == OBSERVED_TEXT
    assert [repository.name for repository in inventory.repositories] == ["alpha", "beta"]
    assert inventory.repositories[0].repository_id == 1
    assert inventory.repositories[0].empty is True
    assert inventory.repositories[0].size == 0
    assert inventory.repositories[0].visibility == "public"
    assert len(fetcher.calls) == 1
    url, timeout, max_bytes = fetcher.calls[0]
    assert url == "https://api.github.com/orgs/valkey-io/repos?type=all&per_page=100&page=1"
    assert timeout <= 15.0
    assert max_bytes == 4 * 1024 * 1024


def test_follows_declared_pagination_without_following_link_target() -> None:
    fetcher = StubFetcher(
        [
            _response([_api_repository("beta", 2)], link=NEXT_PAGE),
            _response([_api_repository("alpha", 1)]),
        ]
    )

    inventory = fetch_organization_inventory(fetch_page=fetcher, observed_at=OBSERVED_AT)

    assert [repository.repository_id for repository in inventory.repositories] == [1, 2]
    assert len(fetcher.calls) == 2
    assert fetcher.calls[1][0].endswith("per_page=100&page=2")


@pytest.mark.parametrize(
    ("limits", "responses", "error"),
    [
        (
            InventoryLimits(max_pages=1),
            [_response([_api_repository()], link=NEXT_PAGE)],
            "pagination is incomplete",
        ),
        (
            InventoryLimits(max_items=1),
            [_response([_api_repository(), _api_repository("beta", 2)])],
            "item bound",
        ),
        (
            InventoryLimits(max_response_bytes=1),
            [_response([])],
            "response-byte bound",
        ),
        (
            InventoryLimits(per_page=1),
            [_response([_api_repository(), _api_repository("beta", 2)])],
            "exceeded per_page",
        ),
    ],
)
def test_enforces_page_item_byte_and_page_size_bounds(
    limits: InventoryLimits,
    responses: list[HttpResponse],
    error: str,
) -> None:
    with pytest.raises(InventoryError, match=error):
        fetch_organization_inventory(
            fetch_page=StubFetcher(responses), limits=limits, observed_at=OBSERVED_AT
        )


def test_enforces_total_elapsed_time_even_when_fetcher_returns() -> None:
    ticks = iter((0.0, 0.0, 2.0))

    with pytest.raises(InventoryError, match="time bound"):
        fetch_organization_inventory(
            fetch_page=StubFetcher([_response([])]),
            limits=InventoryLimits(timeout_seconds=1.0),
            observed_at=OBSERVED_AT,
            elapsed_clock=lambda: next(ticks),
        )


@pytest.mark.parametrize(
    "limits",
    [
        InventoryLimits(max_pages=0),
        InventoryLimits(max_items=0),
        InventoryLimits(max_response_bytes=0),
        InventoryLimits(timeout_seconds=0),
        InventoryLimits(per_page=0),
        InventoryLimits(per_page=101),
    ],
)
def test_rejects_non_bounding_limit_configuration(limits: InventoryLimits) -> None:
    with pytest.raises(InventoryError):
        fetch_organization_inventory(
            fetch_page=StubFetcher([]), limits=limits, observed_at=OBSERVED_AT
        )


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (HttpResponse(500, {}, b"[]"), "HTTP 500"),
        (HttpResponse(200, {}, b"not json"), "not valid JSON"),
        (_response({"repositories": []}), "must be a JSON array"),
        (_response(["not an object"]), "non-object item"),
        (_response([], link=NEXT_PAGE), "next page after an empty page"),
        (HttpResponse(200, {"Link": "not-a-link"}, b"[]"), "malformed Link"),
        (
            HttpResponse(
                200,
                {"Link": '<https://example.com/repos?page=2>; rel="next"'},
                b"[]",
            ),
            "escaped the public API origin",
        ),
        (
            _response(
                [],
                link='<https://api.github.com/repos?type=all&per_page=100&page=2>; rel="next"',
            ),
            "does not match the expected",
        ),
        (
            _response(
                [],
                link=(
                    "<https://api.github.com/orgs/other/repos"
                    '?type=all&per_page=100&page=2>; rel="next"'
                ),
            ),
            "does not match the expected",
        ),
        (
            _response(
                [],
                link=(
                    "<https://api.github.com/orgs/valkey-io/repos"
                    '?type=all&per_page=100&page=3>; rel="next"'
                ),
            ),
            "does not match the expected",
        ),
        (
            _response(
                [],
                link=(
                    "<https://api.github.com/orgs/valkey-io/repos"
                    '?type=all&per_page=100&page=2&extra=true>; rel="next"'
                ),
            ),
            "does not match the expected",
        ),
    ],
)
def test_rejects_http_json_and_pagination_protocol_failures(
    response: HttpResponse, error: str
) -> None:
    with pytest.raises(InventoryError, match=error):
        fetch_organization_inventory(fetch_page=StubFetcher([response]), observed_at=OBSERVED_AT)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"id": True}, "field 'id'"),
        ({"name": " "}, "field 'name'"),
        ({"default_branch": None}, "field 'default_branch'"),
        ({"size": -1}, "field 'size'"),
        ({"fork": "false"}, "field 'fork'"),
        ({"owner": {"login": "other"}}, "outside canonical"),
        ({"full_name": "other/alpha"}, "outside canonical"),
        ({"html_url": "https://example.com/alpha"}, "outside canonical"),
        ({"private": True, "visibility": "private"}, "unexpectedly non-public"),
    ],
)
def test_rejects_malformed_or_non_public_repository_fields(
    overrides: dict[str, object], error: str
) -> None:
    repository = _api_repository()
    repository.update(overrides)
    with pytest.raises(InventoryError, match=error):
        fetch_organization_inventory(
            fetch_page=StubFetcher([_response([repository])]),
            observed_at=OBSERVED_AT,
        )


@pytest.mark.parametrize(
    "repositories",
    [
        [_api_repository("alpha", 1), _api_repository("beta", 1)],
        [_api_repository("alpha", 1), _api_repository("ALPHA", 2)],
    ],
)
def test_rejects_duplicate_stable_ids_and_case_insensitive_names(
    repositories: list[dict[str, object]],
) -> None:
    with pytest.raises(InventoryError, match="duplicate repository"):
        fetch_organization_inventory(
            fetch_page=StubFetcher([_response(repositories)]), observed_at=OBSERVED_AT
        )


def test_rejects_naive_observation_time() -> None:
    with pytest.raises(InventoryError, match="timezone-aware"):
        fetch_organization_inventory(
            fetch_page=StubFetcher([_response([])]),
            observed_at=datetime(2026, 8, 18),
        )


def test_reviewed_active_inventory_reconciles_without_side_effects() -> None:
    source = _source()
    current = _reviewed_inventory()

    report = reconcile_source_inventory(source, current)

    assert report.promotion_allowed is True
    assert report.changes == ()
    assert report.active_nonfork_repositories == 46
    assert source == _source()


def test_reconciliation_rejects_incomplete_malformed_and_duplicate_sources() -> None:
    malformed = _source()
    malformed["repositories"] = "not-a-list"
    duplicate = _source()
    duplicate_repositories = cast(list[dict[str, object]], duplicate["repositories"])
    duplicate_repositories[1] = deepcopy(duplicate_repositories[0])

    for source in ({}, malformed, duplicate):
        with pytest.raises(InventoryError, match="source inventory is invalid"):
            reconcile_source_inventory(source, _reviewed_inventory())


def test_unclassified_active_repository_fails_closed() -> None:
    current = _reviewed_inventory()
    unknown = _snapshot_repository("new-project", 999, "main")
    current = replace(current, repositories=(*current.repositories, unknown))

    with pytest.raises(InventoryReconciliationError) as error:
        reconcile_source_inventory(_source(), current)

    change = _only_change(error)
    assert change.kind == "unclassified"
    assert change.observed_name == "new-project"
    assert error.value.report.promotion_allowed is False
    assert error.value.report.active_nonfork_repositories == 47


@pytest.mark.parametrize("state", ["fork", "archived", "disabled"])
def test_unclassified_inactive_or_fork_repository_does_not_block(state: str) -> None:
    current = _reviewed_inventory()
    unknown = _snapshot_repository("new-project", 999, "main")
    unknown = _with_repository_state(unknown, state)
    current = replace(current, repositories=(*current.repositories, unknown))

    report = reconcile_source_inventory(_source(), current)

    assert report.promotion_allowed is True
    assert report.active_nonfork_repositories == 46


def test_reviewed_empty_repository_remains_active_and_reconciles() -> None:
    current = _reviewed_inventory()
    target = current.repositories[0]
    empty = replace(target, empty=True, size=0)

    report = reconcile_source_inventory(_source(), _changed_inventory(current, target, empty))

    assert report.promotion_allowed is True
    assert report.active_nonfork_repositories == 46


@pytest.mark.parametrize(
    ("field", "kind"),
    [("fork", "forked"), ("archived", "archived"), ("disabled", "disabled")],
)
def test_reviewed_repository_state_change_blocks_promotion(field: str, kind: str) -> None:
    current = _reviewed_inventory()
    target = current.repositories[0]
    changed = _with_repository_state(target, field)

    with pytest.raises(InventoryReconciliationError) as error:
        reconcile_source_inventory(_source(), _changed_inventory(current, target, changed))

    assert _only_change(error).kind == kind


def test_default_branch_change_blocks_promotion() -> None:
    current = _reviewed_inventory()
    target = current.repositories[0]
    changed = replace(target, default_branch="new-default")

    with pytest.raises(InventoryReconciliationError) as error:
        reconcile_source_inventory(_source(), _changed_inventory(current, target, changed))

    assert _only_change(error).kind == "default_branch_changed"


def test_stable_repository_identity_change_blocks_promotion() -> None:
    previous = _reviewed_inventory()
    target = previous.repositories[0]
    replacement = replace(target, repository_id=999)
    current = _changed_inventory(previous, target, replacement)

    with pytest.raises(InventoryReconciliationError) as error:
        reconcile_source_inventory(_source(), current, previous=previous)

    change = _only_change(error)
    assert change.kind == "identity_changed"
    assert change.repository_id == 999
    assert change.source_name == target.name


def test_rename_is_detected_only_by_stable_id_from_previous_inventory() -> None:
    previous = _reviewed_inventory()
    target = previous.repositories[0]
    renamed = replace(
        target,
        name="renamed-project",
        full_name="valkey-io/renamed-project",
        url="https://github.com/valkey-io/renamed-project",
    )
    current = _changed_inventory(previous, target, renamed)

    with pytest.raises(InventoryReconciliationError) as error:
        reconcile_source_inventory(_source(), current, previous=previous)

    change = _only_change(error)
    assert change.kind == "renamed"
    assert change.repository_id == target.repository_id
    assert change.source_name == target.name
    assert change.observed_name == "renamed-project"


@pytest.mark.parametrize(
    ("probes", "kind", "observed_name"),
    [
        ((), "missing", None),
        ((MissingRepositoryProbe(1, "not_found"),), "deleted", None),
        (
            (MissingRepositoryProbe(1, "present", "other-owner/moved-project"),),
            "transferred",
            "other-owner/moved-project",
        ),
        (
            (MissingRepositoryProbe(1, "present", "valkey-io/still-present"),),
            "missing",
            "still-present",
        ),
    ],
)
def test_missing_transfer_and_deletion_require_explicit_probe_evidence(
    probes: tuple[MissingRepositoryProbe, ...],
    kind: str,
    observed_name: str | None,
) -> None:
    previous = _reviewed_inventory()
    target = previous.repositories[0]
    current = _changed_inventory(previous, target, None)

    with pytest.raises(InventoryReconciliationError) as error:
        reconcile_source_inventory(_source(), current, previous=previous, missing_probes=probes)

    change = _only_change(error)
    assert change.kind == kind
    assert change.observed_name == observed_name


@pytest.mark.parametrize(
    "probes",
    [
        (
            MissingRepositoryProbe(1, "not_found"),
            MissingRepositoryProbe(1, "not_found"),
        ),
        (MissingRepositoryProbe(1, "not_found", "owner/name"),),
        (MissingRepositoryProbe(1, "present"),),
        (MissingRepositoryProbe(1, "present", "owner/name/extra"),),
    ],
)
def test_rejects_ambiguous_or_inconsistent_missing_repository_probes(
    probes: tuple[MissingRepositoryProbe, ...],
) -> None:
    with pytest.raises(InventoryError, match="probe"):
        reconcile_source_inventory(_source(), _reviewed_inventory(), missing_probes=probes)


def test_reconciliation_revalidates_snapshot_uniqueness_and_public_state() -> None:
    current = _reviewed_inventory()
    duplicate = replace(
        current.repositories[1], repository_id=current.repositories[0].repository_id
    )
    duplicated = replace(current, repositories=(current.repositories[0], duplicate))
    with pytest.raises(InventoryError, match="IDs must be unique"):
        reconcile_source_inventory(_source(), duplicated)

    outside = replace(current.repositories[0], full_name="other/project")
    with pytest.raises(InventoryError, match="outside canonical"):
        reconcile_source_inventory(_source(), replace(current, repositories=(outside,)))

    private = replace(current.repositories[0], private=True)
    with pytest.raises(InventoryError, match="unexpectedly non-public"):
        reconcile_source_inventory(_source(), replace(current, repositories=(private,)))

    inconsistent_empty = replace(current.repositories[0], empty=True)
    with pytest.raises(InventoryError, match="inconsistent empty state"):
        reconcile_source_inventory(_source(), replace(current, repositories=(inconsistent_empty,)))


def test_reconciliation_rejects_runtime_snapshot_type_mismatches() -> None:
    current = _reviewed_inventory()
    target = current.repositories[0]
    malformed_repositories = (
        replace(target, repository_id=cast(int, True)),
        replace(target, size=cast(int, True)),
        replace(target, fork=cast(bool, "false")),
        replace(target, name=cast(str, 7)),
    )

    for malformed in malformed_repositories:
        snapshot = replace(current, repositories=(malformed,))
        with pytest.raises(InventoryError):
            reconcile_source_inventory(_source(), snapshot)
        with pytest.raises(InventoryError):
            reconcile_source_inventory(_source(), current, previous=snapshot)
