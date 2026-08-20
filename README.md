# Valkeyrie

Valkeyrie is a version-aware, cited, read-only assistant for public Valkey project information.
This repository currently contains a **private prototype foundation only**. It does not authorize an
official Valkey service, Slack deployment, live GitHub credentials, AWS deployment, or any project-state
write.

## Current phase

P0-03 is approved for non-public local Phase 1 engineering:

- the personal prototype authorization is recorded in `PROTOTYPE_AUTHORIZATION.md`;
- all 46 active non-fork `valkey-io` repositories are classified in `sources.yaml`;
- reviewed prompt inputs and public seed evaluations live under `prompts/` and `evals/`;
- versioned shared contracts and golden fixtures live under `src/valkeyrie/schemas/` and
  `tests/fixtures/contracts/`;
- repository governance and proposed remote settings are documented locally; and
- the AWS CDK application synthesizes one isolated development knowledge-plane stack with three separated
  deployer/publisher/runtime roles; operator identity awaits an approved exact principal, and no service/data
  resources, stored credentials, service permissions, or deployment path exist.

D-01 is the next external-action gate and must approve the exact knowledge-plane change set before any
AWS mutation. Official Valkey adoption and repository approval remain hard prerequisites to any Slack
stack, credential, synthetic window, or pilot traffic.

## Development

Requirements:

- Python 3.11
- Node.js 22 or later (required by the Python CDK library through JSII)
- `uv` 0.11.19

Install the exact locked environment and run every local gate:

```bash
uv sync --locked
make check
```

Individual checks are available through `make format-check`, `make lint`, `make typecheck`, `make test`,
`make security`, and `make synth`. `make synth` writes only a local, identity-only stack assembly
under `cdk.out/`; it does not call AWS or provide a deployment workflow.

## Safety boundary

Valkeyrie is designed to read, retrieve, cite, explain, and navigate. The first beta cannot modify GitHub,
run workflows, merge pull requests, publish releases, decide release readiness, read private sources, or
browse arbitrary endpoints. Existing Valkey release systems retain their exact responsibilities.

See `CONTRIBUTING.md`, `GOVERNANCE.md`, `SECURITY.md`, and `docs/repository-governance.md` before making
changes.

## License

BSD 3-Clause. See `LICENSE`.
