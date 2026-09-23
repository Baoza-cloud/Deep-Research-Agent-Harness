# Changelog

All notable changes to this project are documented in this file. The project follows
[Semantic Versioning](https://semver.org/).

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

[1.1.0]: https://github.com/Baoza-cloud/Deep-Research-Agent-Harness/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Baoza-cloud/Deep-Research-Agent-Harness/releases/tag/v1.0.0
