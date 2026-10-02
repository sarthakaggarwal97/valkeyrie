# Decisions pending the owner's approval

Two items the third review wave raised need a decision rather than a fix. Each is written here as
the text that would be recorded if approved; nothing below is in force until the owner says so in
Home thread 338.

## 1. Amendment to PROTOTYPE_AUTHORIZATION.md: the Slack `run` command

Status: PROPOSED, not approved. The "Slack authorization" section says the app "is not authorized
to mutate Valkey, GitHub, release systems, or AWS, and it holds no capability to do so". Commit
284f23a (2026-09-25) added `tools/actions.py`, a `run <action>` command that dispatches GitHub
workflows, and the record was not amended. The code and the record disagree; one of them must
change. Proposed amendment text:

> ### Slack commands (amendment, 2026-10-02)
> The owner authorizes a closed catalog of GitHub workflow dispatches from Slack, implemented in
> `tools/actions.py` and declared in `tools/actions.yaml`. Bounds, each enforced in code:
> - Repositories: owned by `sarthakaggarwal97` only, compared case-insensitively. No action may
>   target `valkey-io/*`; a catalog entry that does fails to load.
> - Actions: only the catalog's named entries; each names its workflow file, a validated git ref,
>   and the allowed input keys with their value patterns. Nothing in a Slack message can name a
>   repository, workflow or input key outside the catalog.
> - Operators: only the Slack user ids listed in the catalog; the id comes from the event, never
>   from message text.
> - Rate: one dispatch per action per 120 seconds, process-wide; a dry run needs no credential.
> - Credential: `VALKEYRIE_ACTIONS_TOKEN`, a token with Actions write on the catalogued personal
>   repositories only, held by the bot process and never by the Lambda. Revocation disables the
>   command (it reports "nothing was triggered").
> - Audit: every dispatch is appended to the local audit file before the request is sent.
> This does not widen the answer path: the Lambda, the knowledge base and the GitHub read token
> are unchanged.

Alternative: remove the command (delete `tools/actions.py`, `tools/actions.yaml`, the `run`
branch in `tools/slack_bot.py`, and `tests/test_actions.py`), which makes the existing record true.

## 2. Pruning old corpus generations

Status: PROPOSED, not approved. Every corpus generation is written under
`kb-documents/generations/<generation>/` and nothing deletes one: the bucket lifecycle only aborts
incomplete multipart uploads, the publisher role is denied `s3:DeleteObject`, and the knowledge
base data source covers the whole `kb-documents/` prefix. Measured on 2026-09-28: ingestion scanned
22,838 documents across 6 generations, about 95 minutes, most of it re-reading generations that
will never be active again. Proposed policy:

- Keep the ACTIVE generation and its immediate predecessor (the rollback target). Delete every
  other generation prefix.
- Pruning runs as a separate workflow job after a successful promotion, under a role whose only
  write permission is `s3:DeleteObject` on `kb-documents/generations/*`, with an explicit deny on
  the two retained prefixes computed at run time from the active pointer and the lifecycle record.
  The publisher role stays denied deletion.
- A generation is deleted only if its lifecycle record is not `active` and not the predecessor,
  and it is older than 7 days (so a refresh that failed mid-way is not pruned under a concurrent
  run).
- Expected effect: ingestion scans two generations (about 7,600 documents), roughly a third of
  the time, and the bucket holds two corpora instead of six.

Alternative: an S3 lifecycle rule expiring objects under `generations/` after N days. Rejected
because it cannot tell the active generation from an old one: an active corpus older than N days
would be deleted under the knowledge base.
