# Contributing

Valkeyrie is currently a private personal prototype, not an official Valkey project. Contributions must
preserve the read-only boundary and the approval gates in `PROTOTYPE_AUTHORIZATION.md`.

## Development flow

1. Create a focused branch from `main`.
2. Install the locked environment with `uv sync --locked`.
3. Make the smallest complete change and add adversarial tests.
4. Run `make check`.
5. Review source authority, privacy, security, cost/runaway, and ownership implications.
6. Commit with a Developer Certificate of Origin sign-off: `git commit -s`.
7. Do not push, open a pull request, trigger a workflow, or change remote settings without explicit owner
   approval.

## Change boundaries

Changes to `sources.yaml`, prompts, evaluations, infrastructure, permissions, deployment policy, model or
retrieval configuration, live capabilities, Slack scope, retention, or release boundaries require the
specific review and gate described in the implementation backlog. Retrieved or model-generated text can
never change those controls.

Private repositories, security advisories, internal information, credentials, expected hidden answers,
and private Slack content must not be added to source, prompt, test, or log fixtures.

## Simplicity discipline

Every phase must implement the smallest complete solution that satisfies its current gate. Prefer native
managed capabilities and existing helpers. Add an abstraction, extension point, dependency, service, or
configuration option only for a demonstrated current requirement—not a speculative future use.

Reviewers must treat unnecessary complexity, duplicated responsibility, premature generalization, and
scope beyond the active phase as blocking findings. Extensibility means clean boundaries that can be
extended when approved, not code or infrastructure built before it is needed.
