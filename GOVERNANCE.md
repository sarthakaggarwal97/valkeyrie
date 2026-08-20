# Governance

## Current status

Valkeyrie is a private personal prototype owned by Sarthak Aggarwal (`sarthakaggarwal97`). It is not an
official Valkey project and does not speak for Valkey maintainers.

The prototype owner approves local changes and is the current infrastructure owner. Approval is recorded
in reviewable repository artifacts; it never authorizes a later deployment or traffic gate implicitly.

## Decision rules

- GitHub and exact public source commits remain authoritative for project facts.
- Source selection, prompts, evaluations, permissions, infrastructure, and rollout gates require human
  review.
- The model may explain validated evidence but cannot establish authority, policy, permissions, release
  criteria, or readiness.
- Existing release ownership remains unchanged: `valkey` owns Start Release; `valkey-ci-agent` owns
  authorization, reconciliation, qualification, and protected tag/release publication;
  `valkey-release-automation` owns builds, artifacts, packages, credentials, and downstream publication.
- Valkeyrie reads, cites, and explains only. Any project-state write requires a separate RFC and security
  review.

## Adoption

Moving to an official `valkey-io` repository, adding another maintainer, or opening a Slack pilot requires
explicit maintainer approval and the applicable owner, security, privacy, operations, and support gates.
Until then, no document in this repository may imply official adoption.
