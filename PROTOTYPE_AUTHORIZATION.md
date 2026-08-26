# Personal prototype authorization

Status: Authorized for local private-prototype implementation through Phase 1  
Decision date: 2026-08-18  
Backlog decisions: P0-01 and P0-03  
Prototype maintainer and infrastructure owner: Sarthak Aggarwal (`sarthakaggarwal97`)

## Bound design revision

This authorization binds the exact local design artifact below:

- File: `docs/rfcs/valkeyrie-maintainer-rfc.md`
- Status: `Final RFC for maintainer review`
- RFC date: `2026-08-18`
- SHA-256: `75510aa51200ebdc6acc6968ebbfe9f0eb73940149de5f4380148c05f19b9012`

Any byte change to that artifact requires renewed review and a new authorization record. The one-page
architecture and simplified phase plan are subordinate implementation references; the RFC controls if
they conflict.

## Approved prototype vehicle

- Repository: `https://github.com/sarthakaggarwal97/valkeyrie`
- Baseline commit: `0485ace3806e3fad69ea82dcd359ab9f2685dbbc`
- Visibility at authorization: private
- Project status: personal prototype; not adopted as an official Valkey project
- Recorded future AWS target: personal account `968533178160`, region `us-east-1`

Recording an AWS target does not authorize any AWS mutation. Any future resource must be dedicated,
`valkeyrie-*` namespaced, separately approved, and unable to modify existing account resources.

## P0-03 implementation approval

The instruction `Simplify the backlog gates and continue further implementation`, recorded in Home
thread 338 on 2026-08-18, accepts the reviewed product and read-only boundaries, complete source
classification, prompt package, evaluation inputs, owner, development environment, and qualification
expectations. It authorizes non-public local Phase 1 engineering.

This approval permits local schemas, validators, evaluation and threat-model work, source and corpus
tooling, synthesis-only infrastructure definitions, local application code, tests, and builds. It does
not authorize AWS deployment, Slack credentials or traffic, live GitHub credentials, remote GitHub
mutation, workflow execution, publication, or public beta.

## Authorized now

- The completed P0-01, F-01, F-02, and P0-02 foundation.
- Non-public local Phase 1 implementation and validation under the bound design.
- Read-only discovery of public Valkey repositories and the recorded personal AWS account.
- Local dependency installation, static analysis, unit tests, security checks, package builds, and CDK
  synthesis.
- Local commits with DCO sign-off when explicitly requested; no remote publication is implied.

The prototype remains read-only with respect to Valkey, GitHub, Slack, AWS, release systems, and every
other external project or service.

## Not authorized

This decision does **not** authorize:

- Slack credentials, app installation, deployment, synthetic traffic, real traffic, or channel access;
- live-state credentials or enabled request-time GitHub capabilities;
- GitHub or project-state writes, comments, reactions, workflow dispatch or rerun, merges, tags, releases,
  repository creation, branch-protection changes, or environment changes;
- pushes, pull requests, workflow execution, or publication of this private prototype;
- AWS provisioning, bootstrap, deployment, update, deletion, IAM mutation, secret creation, or reuse of
  existing resources; or
- public beta, release-readiness decisions, or invocation of release automation.

Official Valkey adoption and approval of the official repository home remain hard prerequisites to
P2-D1, which is the first Slack stack, credential, or synthetic-window authorization.

## Next external-action gate

D-01 must approve the exact synthesized development knowledge-plane change set, IAM diff, namespaced
resource collision check, expected cost and quotas, approver, and timestamp before the first AWS
mutation. D-02 separately protects the later application deployment because its deployable change set
does not exist until after the knowledge plane is available.

## Deployment and public endpoint approval

The earlier sections record what was authorized on 2026-08-18 and remain accurate as history. Their
non-public and no-deployment boundaries have since been superseded by later owner instructions in the
same thread, recorded here so the document does not contradict what is deployed.

AWS deployment was authorized in Home thread 338 and carried out against the recorded personal
account: the development knowledge plane, and application versions 6, 7, and 8. Version 7 runs the
qualified Fable model and version 8 runs Claude Opus 5 at explicit owner direction despite failing
qualification.

The instruction `Build the public URL with Fable`, recorded in Home thread 338 on 2026-08-26,
authorized a Lambda Function URL serving the qualified Fable model over the active corpus
generation. An unauthenticated url was attempted and is NOT in place: anonymous callers received
403, Palisade detected the function as world accessible, and epoxy-engage_mitigations automatically
removed the public permission. The endpoint is therefore scoped to this AWS account and requires
SigV4 request signing by an identity in account 968533178160.

World-accessible exposure of this prototype is treated as prohibited. Fronting the same function
with a public CloudFront distribution was considered and deliberately rejected, because it would
restore the world reach the automated mitigation removed.

This does not authorize Slack credentials or traffic, live GitHub mutation, publication to an
official Valkey repository, or release-readiness decisions, all of which remain gated.
