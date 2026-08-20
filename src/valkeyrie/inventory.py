"""Bounded public GitHub organization inventory and source reconciliation."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import Literal, cast
from urllib.parse import parse_qsl, urlencode, urlsplit

from valkeyrie.github import GitHubFetcher, GitHubReadError, fetch_public_github
from valkeyrie.sources import SourceInventoryError, validate_source_inventory


class InventoryError(ValueError):
    """A repository inventory could not be acquired or validated safely."""


@dataclass(frozen=True)
class InventoryLimits:
    """Hard bounds for one organization inventory request."""

    max_pages: int = 10
    max_items: int = 1_000
    max_response_bytes: int = 4 * 1024 * 1024
    timeout_seconds: float = 15.0
    per_page: int = 100


@dataclass(frozen=True)
class RepositorySnapshot:
    """Validated public state for one GitHub repository."""

    repository_id: int
    name: str
    full_name: str
    url: str
    default_branch: str
    fork: bool
    archived: bool
    disabled: bool
    empty: bool
    size: int
    private: bool
    visibility: Literal["public"]


@dataclass(frozen=True)
class OrganizationInventory:
    """A complete immutable observation of the public valkey-io organization."""

    organization: Literal["valkey-io"]
    observed_at: str
    repositories: tuple[RepositorySnapshot, ...]


@dataclass(frozen=True)
class MissingRepositoryProbe:
    """Explicit read-only evidence for a repository absent from the organization list."""

    repository_id: int
    outcome: Literal["not_found", "present"]
    full_name: str | None = None


ChangeKind = Literal[
    "archived",
    "default_branch_changed",
    "deleted",
    "disabled",
    "forked",
    "identity_changed",
    "missing",
    "renamed",
    "transferred",
    "unclassified",
]


@dataclass(frozen=True)
class InventoryChange:
    """One source-manifest divergence that blocks candidate promotion."""

    kind: ChangeKind
    repository_id: int | None
    source_name: str | None
    observed_name: str | None


@dataclass(frozen=True)
class ReconciliationReport:
    """A pure promotion decision; no generation state is read or mutated."""

    observed_at: str
    active_nonfork_repositories: int
    changes: tuple[InventoryChange, ...]

    @property
    def promotion_allowed(self) -> bool:
        """Return whether the observed inventory exactly matches reviewed sources."""
        return not self.changes


class InventoryReconciliationError(InventoryError):
    """The observed organization cannot safely promote with the reviewed sources."""

    def __init__(self, report: ReconciliationReport) -> None:
        self.report = report
        summary = ", ".join(
            f"{change.kind}:{change.source_name or change.observed_name}"
            for change in report.changes
        )
        super().__init__(f"organization inventory does not reconcile: {summary}")


_DEFAULT_LIMITS = InventoryLimits()
_REPOSITORY_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_ORGANIZATION_REPOSITORY_PATHS = {
    "/orgs/valkey-io/repos",
    "/organizations/164458127/repos",
}


def fetch_organization_inventory(
    *,
    fetch_page: GitHubFetcher | None = None,
    limits: InventoryLimits = _DEFAULT_LIMITS,
    observed_at: datetime | None = None,
    elapsed_clock: Callable[[], float] = monotonic,
) -> OrganizationInventory:
    """Fetch every public valkey-io repository through bounded GET-only pagination."""
    _validate_limits(limits)
    fetch = fetch_page or fetch_public_github
    started_at = elapsed_clock()
    remaining_bytes = limits.max_response_bytes
    repositories: list[RepositorySnapshot] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()

    for page_number in range(1, limits.max_pages + 1):
        elapsed = elapsed_clock() - started_at
        remaining_time = limits.timeout_seconds - elapsed
        if remaining_time <= 0:
            raise InventoryError("organization inventory exceeded its time bound")
        url = _organization_page_url(page_number, limits.per_page)
        try:
            response = fetch(url, remaining_time, remaining_bytes)
        except GitHubReadError as error:
            raise InventoryError(f"cannot fetch public GitHub inventory: {error}") from error
        if elapsed_clock() - started_at > limits.timeout_seconds:
            raise InventoryError("organization inventory exceeded its time bound")
        if response.status != 200:
            raise InventoryError(
                f"GitHub inventory page {page_number} returned HTTP {response.status}"
            )
        if len(response.body) > remaining_bytes:
            raise InventoryError("organization inventory exceeded its response-byte bound")
        remaining_bytes -= len(response.body)

        page = _decode_page(response.body, page_number)
        if len(page) > limits.per_page:
            raise InventoryError(
                f"GitHub inventory page {page_number} exceeded per_page={limits.per_page}"
            )
        for raw_repository in page:
            repository = _parse_repository(raw_repository)
            folded_name = repository.name.casefold()
            if repository.repository_id in seen_ids:
                raise InventoryError(
                    f"duplicate repository id {repository.repository_id} in organization inventory"
                )
            if folded_name in seen_names:
                raise InventoryError(
                    f"duplicate repository name {repository.name!r} in organization inventory"
                )
            seen_ids.add(repository.repository_id)
            seen_names.add(folded_name)
            repositories.append(repository)
            if len(repositories) > limits.max_items:
                raise InventoryError("organization inventory exceeded its item bound")

        has_next = _has_next_page(
            response.headers,
            expected_page=page_number + 1,
            per_page=limits.per_page,
        )
        if not has_next:
            break
        if not page:
            raise InventoryError("GitHub inventory declared a next page after an empty page")
        if page_number == limits.max_pages:
            raise InventoryError("organization inventory pagination is incomplete")
    else:  # pragma: no cover - every terminal path exits or raises above
        raise InventoryError("organization inventory pagination is incomplete")

    timestamp = _format_observed_at(observed_at or datetime.now(UTC))
    return OrganizationInventory(
        organization="valkey-io",
        observed_at=timestamp,
        repositories=tuple(
            sorted(
                repositories,
                key=lambda repository: (repository.name.casefold(), repository.repository_id),
            )
        ),
    )


def reconcile_source_inventory(
    source_inventory: Mapping[str, object],
    current: OrganizationInventory,
    *,
    previous: OrganizationInventory | None = None,
    missing_probes: Sequence[MissingRepositoryProbe] = (),
) -> ReconciliationReport:
    """Require the current active non-fork inventory to match reviewed source entries."""
    _validate_inventory_snapshot(current)
    if previous is not None:
        _validate_inventory_snapshot(previous)
    probes = _validate_probes(missing_probes)

    try:
        validate_source_inventory(source_inventory)
    except SourceInventoryError as error:
        raise InventoryError(f"source inventory is invalid: {error}") from error
    source_repositories = cast(list[dict[str, object]], source_inventory["repositories"])

    source_by_name = {
        cast(str, repository["name"]).casefold(): repository for repository in source_repositories
    }
    current_by_name = {
        repository.name.casefold(): repository for repository in current.repositories
    }
    current_by_id = {repository.repository_id: repository for repository in current.repositories}
    previous_by_name = (
        {repository.name.casefold(): repository for repository in previous.repositories}
        if previous is not None
        else {}
    )
    renamed_current_ids: set[int] = set()
    changes: list[InventoryChange] = []

    for folded_name, source_repository in source_by_name.items():
        source_name = cast(str, source_repository["name"])
        observed = current_by_name.get(folded_name)
        if observed is not None:
            previous_repository = previous_by_name.get(folded_name)
            if (
                previous_repository is not None
                and previous_repository.repository_id != observed.repository_id
            ):
                changes.append(
                    InventoryChange(
                        "identity_changed",
                        observed.repository_id,
                        source_name,
                        observed.name,
                    )
                )
                continue
            state_change = _inactive_change(observed, source_name)
            if state_change is not None:
                changes.append(state_change)
            elif observed.default_branch != source_repository["requested_ref"]:
                changes.append(
                    InventoryChange(
                        "default_branch_changed",
                        observed.repository_id,
                        source_name,
                        observed.name,
                    )
                )
            continue

        previous_repository = previous_by_name.get(folded_name)
        if previous_repository is not None:
            renamed = current_by_id.get(previous_repository.repository_id)
            if renamed is not None:
                renamed_current_ids.add(renamed.repository_id)
                changes.append(
                    InventoryChange(
                        "renamed",
                        renamed.repository_id,
                        source_name,
                        renamed.name,
                    )
                )
                continue
            changes.append(_missing_change(source_name, previous_repository.repository_id, probes))
            continue

        changes.append(InventoryChange("missing", None, source_name, None))

    for repository in current.repositories:
        if not _is_active_nonfork(repository):
            continue
        if repository.name.casefold() in source_by_name:
            continue
        if repository.repository_id in renamed_current_ids:
            continue
        changes.append(
            InventoryChange("unclassified", repository.repository_id, None, repository.name)
        )

    report = ReconciliationReport(
        observed_at=current.observed_at,
        active_nonfork_repositories=sum(
            _is_active_nonfork(repository) for repository in current.repositories
        ),
        changes=tuple(sorted(changes, key=_change_sort_key)),
    )
    if not report.promotion_allowed:
        raise InventoryReconciliationError(report)
    return report


def _validate_limits(limits: InventoryLimits) -> None:
    if limits.max_pages < 1:
        raise InventoryError("max_pages must be positive")
    if limits.max_items < 1:
        raise InventoryError("max_items must be positive")
    if limits.max_response_bytes < 1:
        raise InventoryError("max_response_bytes must be positive")
    if limits.timeout_seconds <= 0:
        raise InventoryError("timeout_seconds must be positive")
    if not 1 <= limits.per_page <= 100:
        raise InventoryError("per_page must be between 1 and 100")


def _organization_page_url(page_number: int, per_page: int) -> str:
    query = urlencode({"type": "all", "per_page": per_page, "page": page_number})
    return f"https://api.github.com/orgs/valkey-io/repos?{query}"


def _decode_page(body: bytes, page_number: int) -> list[Mapping[str, object]]:
    try:
        decoded = cast(object, json.loads(body.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InventoryError(f"GitHub inventory page {page_number} is not valid JSON") from error
    if not isinstance(decoded, list):
        raise InventoryError(f"GitHub inventory page {page_number} must be a JSON array")
    if not all(isinstance(item, dict) for item in decoded):
        raise InventoryError(f"GitHub inventory page {page_number} contains a non-object item")
    return cast(list[Mapping[str, object]], decoded)


def _parse_repository(raw: Mapping[str, object]) -> RepositorySnapshot:
    repository_id = _integer_field(raw, "id", minimum=1)
    name = _string_field(raw, "name")
    full_name = _string_field(raw, "full_name")
    url = _string_field(raw, "html_url")
    default_branch = _string_field(raw, "default_branch")
    size = _integer_field(raw, "size", minimum=0)
    fork = _boolean_field(raw, "fork")
    archived = _boolean_field(raw, "archived")
    disabled = _boolean_field(raw, "disabled")
    private = _boolean_field(raw, "private")
    visibility = _string_field(raw, "visibility")
    owner = raw.get("owner")
    owner_login = owner.get("login") if isinstance(owner, dict) else None

    if not _REPOSITORY_NAME.fullmatch(name):
        raise InventoryError(f"repository {repository_id} has invalid name {name!r}")
    expected_full_name = f"valkey-io/{name}"
    expected_url = f"https://github.com/{expected_full_name}"
    if owner_login != "valkey-io" or full_name != expected_full_name or url != expected_url:
        raise InventoryError(f"repository {repository_id} is outside canonical valkey-io identity")
    if private or visibility != "public":
        raise InventoryError(f"repository {full_name} is unexpectedly non-public")

    return RepositorySnapshot(
        repository_id=repository_id,
        name=name,
        full_name=full_name,
        url=url,
        default_branch=default_branch,
        fork=fork,
        archived=archived,
        disabled=disabled,
        empty=size == 0,
        size=size,
        private=private,
        visibility="public",
    )


def _integer_field(raw: Mapping[str, object], name: str, *, minimum: int) -> int:
    value = raw.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise InventoryError(f"GitHub repository field {name!r} must be an integer >= {minimum}")
    return value


def _string_field(raw: Mapping[str, object], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise InventoryError(f"GitHub repository field {name!r} must be a non-blank string")
    return value


def _boolean_field(raw: Mapping[str, object], name: str) -> bool:
    value = raw.get(name)
    if not isinstance(value, bool):
        raise InventoryError(f"GitHub repository field {name!r} must be a boolean")
    return value


def _has_next_page(headers: Mapping[str, str], *, expected_page: int, per_page: int) -> bool:
    link = next((value for name, value in headers.items() if name.casefold() == "link"), None)
    if link is None:
        return False
    next_links = 0
    expected_query = sorted(
        (("type", "all"), ("per_page", str(per_page)), ("page", str(expected_page)))
    )
    for segment in link.split(","):
        parts = [part.strip() for part in segment.strip().split(";")]
        if not parts or not parts[0].startswith("<") or not parts[0].endswith(">"):
            raise InventoryError("GitHub inventory returned a malformed Link header")
        target = urlsplit(parts[0][1:-1])
        if target.scheme != "https" or target.netloc != "api.github.com":
            raise InventoryError("GitHub inventory Link header escaped the public API origin")
        relations: set[str] = set()
        for parameter in parts[1:]:
            key, separator, value = parameter.partition("=")
            if separator and key.casefold() == "rel":
                if len(value) < 2 or value[0] != '"' or value[-1] != '"':
                    raise InventoryError("GitHub inventory returned a malformed Link relation")
                relations.update(value[1:-1].split())
        if "next" in relations:
            next_links += 1
            if (
                target.path not in _ORGANIZATION_REPOSITORY_PATHS
                or target.fragment
                or sorted(parse_qsl(target.query, keep_blank_values=True)) != expected_query
            ):
                raise InventoryError(
                    "GitHub inventory next link does not match the expected valkey-io page"
                )
    if next_links > 1:
        raise InventoryError("GitHub inventory returned multiple next-page links")
    return next_links == 1


def _format_observed_at(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise InventoryError("inventory observation time must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _validate_inventory_snapshot(inventory: OrganizationInventory) -> None:
    if inventory.organization != "valkey-io":
        raise InventoryError("inventory organization must be valkey-io")
    if not isinstance(inventory.observed_at, str):
        raise InventoryError("inventory observation time must be a string")
    try:
        parsed_time = datetime.fromisoformat(inventory.observed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise InventoryError("inventory observation time is invalid") from error
    if parsed_time.tzinfo is None or not inventory.observed_at.endswith("Z"):
        raise InventoryError("inventory observation time must be UTC")
    if not isinstance(inventory.repositories, tuple):
        raise InventoryError("inventory repositories must be an immutable tuple")

    ids: list[int] = []
    names: list[str] = []
    for repository in inventory.repositories:
        if not isinstance(repository, RepositorySnapshot):
            raise InventoryError("inventory contains an invalid repository snapshot")
        if (
            not isinstance(repository.repository_id, int)
            or isinstance(repository.repository_id, bool)
            or repository.repository_id < 1
        ):
            raise InventoryError("inventory repository IDs must be positive integers")
        if (
            not isinstance(repository.size, int)
            or isinstance(repository.size, bool)
            or repository.size < 0
        ):
            raise InventoryError(f"repository {repository.repository_id} has an invalid size")
        string_fields = (
            repository.name,
            repository.full_name,
            repository.url,
            repository.default_branch,
            repository.visibility,
        )
        if not all(isinstance(value, str) and value.strip() for value in string_fields):
            raise InventoryError(
                f"repository {repository.repository_id} has an invalid string field"
            )
        boolean_fields = (
            repository.fork,
            repository.archived,
            repository.disabled,
            repository.empty,
            repository.private,
        )
        if not all(isinstance(value, bool) for value in boolean_fields):
            raise InventoryError(
                f"repository {repository.repository_id} has an invalid boolean field"
            )
        if not _REPOSITORY_NAME.fullmatch(repository.name):
            raise InventoryError(f"repository has invalid name {repository.name!r}")
        expected_full_name = f"valkey-io/{repository.name}"
        expected_url = f"https://github.com/{expected_full_name}"
        if repository.full_name != expected_full_name or repository.url != expected_url:
            raise InventoryError(
                f"repository {repository.repository_id} is outside canonical valkey-io identity"
            )
        if repository.private or repository.visibility != "public":
            raise InventoryError(f"repository {repository.full_name} is unexpectedly non-public")
        if repository.empty != (repository.size == 0):
            raise InventoryError(f"repository {repository.full_name} has inconsistent empty state")
        ids.append(repository.repository_id)
        names.append(repository.name.casefold())

    if len(ids) != len(set(ids)):
        raise InventoryError("inventory repository IDs must be unique")
    if len(names) != len(set(names)):
        raise InventoryError("inventory repository names must be case-insensitively unique")


def _validate_probes(
    probes: Sequence[MissingRepositoryProbe],
) -> dict[int, MissingRepositoryProbe]:
    by_id: dict[int, MissingRepositoryProbe] = {}
    for probe in probes:
        if probe.repository_id < 1 or probe.repository_id in by_id:
            raise InventoryError("missing-repository probe IDs must be positive and unique")
        if probe.outcome not in {"not_found", "present"}:
            raise InventoryError("missing-repository probe outcome is invalid")
        if probe.outcome == "not_found" and probe.full_name is not None:
            raise InventoryError("not-found repository probes cannot have a full name")
        if probe.outcome == "present":
            parts = probe.full_name.split("/") if probe.full_name is not None else []
            if (
                len(parts) != 2
                or not _REPOSITORY_NAME.fullmatch(parts[0])
                or not _REPOSITORY_NAME.fullmatch(parts[1])
            ):
                raise InventoryError("present repository probes require an owner/name identity")
        by_id[probe.repository_id] = probe
    return by_id


def _inactive_change(repository: RepositorySnapshot, source_name: str) -> InventoryChange | None:
    if repository.fork:
        kind: ChangeKind = "forked"
    elif repository.archived:
        kind = "archived"
    elif repository.disabled:
        kind = "disabled"
    else:
        return None
    return InventoryChange(kind, repository.repository_id, source_name, repository.name)


def _is_active_nonfork(repository: RepositorySnapshot) -> bool:
    return not repository.fork and not repository.archived and not repository.disabled


def _missing_change(
    source_name: str,
    repository_id: int,
    probes: Mapping[int, MissingRepositoryProbe],
) -> InventoryChange:
    probe = probes.get(repository_id)
    if probe is None:
        return InventoryChange("missing", repository_id, source_name, None)
    if probe.outcome == "not_found":
        return InventoryChange("deleted", repository_id, source_name, None)
    full_name = probe.full_name
    if full_name is None:
        raise InventoryError("present repository probe has no full name")
    owner, _, name = full_name.partition("/")
    if owner.casefold() != "valkey-io":
        return InventoryChange("transferred", repository_id, source_name, full_name)
    return InventoryChange("missing", repository_id, source_name, name)


def _change_sort_key(change: InventoryChange) -> tuple[str, str, str, int]:
    return (
        change.source_name or "",
        change.observed_name or "",
        change.kind,
        change.repository_id or 0,
    )
