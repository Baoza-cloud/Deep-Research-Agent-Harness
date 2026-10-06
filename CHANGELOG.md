# Changelog

All notable changes to this project are documented in this file. The project follows
[Semantic Versioning](https://semver.org/).

## [1.2.1] - 2026-10-06

### Added

- Deterministic A/B evidence matrices for comparison research, including per-subject,
  per-dimension coverage metrics and prioritized Evidence Verifier repair queries.
- Transactional Blue Patch integrity gates for duplicate Claims, citation-boundary joins, and
  enumeration conflicts; unsafe patches are rolled back individually.
- Source commit metadata and package-derived engine versions for reproducible online runs.

### Changed

- Comparison coverage gaps now take precedence over repeated verification of already represented
  Claims, and keep the run in an evidence-gap state until both sides are covered.
- A deterministic Claim repair no longer skips a pending structured comparison-coverage repair.
- Unconfigured model pricing is reported as `cost_usd=null` with `cost_status=unknown`, rather
  than as a misleading zero-dollar run.

### Verified

- 130 offline tests and 84% branch coverage.
- Regression coverage for asymmetric comparisons, duplicate/format/enumeration Patch rollback,
  unknown-cost telemetry, and release identity metadata.

## [1.2.0] - 2026-10-06

### Added

- SQLite-backed Run, DAG node, task-state, budget, intermediate-report, and Trace persistence.
- Idempotent `run_id` submission, failed-node resume, completed-node reuse, crash recovery,
  global-timeout recovery, and deterministic Trace Replay with integrity verification.
- Versioned structured Trace envelopes with Trace, Event, and Span identifiers.
- Per-Agent and component telemetry for bounded input/output summaries, latency, provider Token
  usage, estimated cost, and retry counts.
- Explicit Dynamic Swarm role, Evidence Verifier, Fixed Harness fallback, final quality selection,
  and stop-condition decision events.
- Dependency-free local JSON/HTML Trace Viewer with DAG timeline, concurrency waterfall, Agent
  telemetry, control decisions, and a Harness-versus-ordinary-Agent comparison.
- `deep-research-trace` console command and `--trace-html` runtime output.

### Changed

- OpenAI-compatible backends now expose observable application-level retries and exact provider
  usage when returned by the backend; optional per-million-token rates drive Trace cost estimates.
- Resumed attempts receive a fresh wall-clock deadline while preserving persisted cumulative
  resource consumption.
- CI now verifies the Trace Viewer entry point in both editable and wheel-installed environments.
- Release manifests now pin source code, benchmark datasets, selected formal experiments, and the
  real-Web interview showcase rather than datasets and experiments alone.

### Verified

- 123 offline tests across Harness, Retrieval, Verifier, Red/Blue, Evaluation, Recovery, Replay,
  and structured tracing.
- 83% branch coverage under the locked Python 3.12 environment.
- Python 3.10, 3.11, and 3.12 compatibility plus fresh-environment wheel installation.
- Zero-key offline result and Replay bundles rendered as self-contained HTML Trace viewers.

## [1.1.0] - 2026-09-23

### Added

- GitHub Actions gates for Python compatibility, Ruff, coverage, package build, wheel install,
  module entry-point validation, and a zero-key offline smoke test.
- Locked CPython 3.12.7 reference environment and reproducible dependency set.
- `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`, and release artifact hashes.
- Retrieval-Gold v1.0 calibration data and final Frozen/Adversarial experiment artifacts.

### Changed

- Renamed the project to **Deep Research Agent Harness** and the Python distribution to
  `deep-research-agent-harness`.
- Repositioned local Hybrid RAG as a pluggable retrieval backend rather than the project core.
- Split the offline test suite by Harness, Retrieval, Verifier, Red/Blue, Evaluation, and
  Orchestrator concerns.
- Updated README, architecture, technical documentation, and project plan around the Harness
  contracts and reproducible evaluation protocol.

### Verified

- 112 offline tests.
- ResearchBench-Frozen v1.0: 35 questions × 3 repeats for Fixed Harness and Dynamic Swarm.
- ResearchBench-Adversarial v0.1: 35 cases × 3 repeats across four repair variants.

## [1.0.0] - 2026-09-22

### Added

- Harness contracts, DAG orchestration, nine-state task lifecycle, Dynamic Swarm, and budgets.
- Shared memory, context compression, Claim–Evidence verification, and Red/Blue repair.
- ResearchBench Frozen, Live, Adversarial, and retrieval calibration tracks.

[1.2.1]: https://github.com/Baoza-cloud/Deep-Research-Agent-Harness/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/Baoza-cloud/Deep-Research-Agent-Harness/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/Baoza-cloud/Deep-Research-Agent-Harness/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Baoza-cloud/Deep-Research-Agent-Harness/releases/tag/v1.0.0
