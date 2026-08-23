# Repository governance settings

Status: Reviewed local proposal; no remote setting is created or changed by this file.

## Default branch

For the current sole-owner private prototype, require the `phase0` status check and prevent force pushes
and deletion when remote settings are explicitly approved. DCO sign-off is a contribution policy verified
before any local commit; it is not represented as GitHub signed-commit protection or a nonexistent status
check.

Mandatory approving review, code-owner approval, stale-review dismissal, conversation resolution, and
administrator enforcement are deferred until an authorized second human owner exists. Before any Slack
pilot or official adoption, configure at least one independent human code owner and then require one
approving code-owner review with administrator enforcement.

The current private prototype must not apply any remote setting without explicit owner approval.

## Actions

- Default workflow token permission: read repository contents only.
- Pull requests from workflows: disabled.
- Every workflow declares top-level permissions explicitly.
- Third-party Actions use immutable 40-character commit SHAs.
- Phase 0 has no Slack credential and no project-mutation token. The scheduled corpus refresh is the only
  credentialed workflow: it assumes a corpus role through OIDC in the `corpus` environment and may publish,
  ingest, evaluate, and activate a corpus. It has no application or prompt deployment permission, and no
  workflow may combine both responsibilities.

## Dependency updates

Dependabot proposes bounded weekly updates for Python and GitHub Actions. Updates must preserve exact direct
dependency pins, refresh `uv.lock`, pass every local check, and receive normal review. Security updates do
not bypass compatibility or boundary tests.

## Protected environments

Two distinct future environments are reserved but are **not configured or used in Phase 0**:

- `application`: application and prompt deployment only; no corpus publication or activation permission.
- `corpus`: corpus publication, ingestion, evaluation, and conditional activation only; no application or
  prompt deployment permission.

Creating either environment, assigning reviewers, adding OIDC trust, or storing secrets is a remote
mutation requiring its later explicit approval gate. Environment separation is a policy invariant; one
workflow or role must never combine both responsibilities.
