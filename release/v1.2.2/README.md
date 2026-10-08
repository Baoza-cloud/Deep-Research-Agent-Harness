# v1.2.2 release manifest

The Git tag `v1.2.2` is the canonical release identity. This patch release hardens atomic Claim
verification and deterministic Structured Blue repair without changing the pinned
ResearchBench-Frozen benchmark protocol.

`SHA256SUMS` pins four artifact groups:

- the source, packaging, CI, and regression-test files changed for v1.2.2;
- ResearchBench Frozen, Live, Adversarial, and Retrieval-Gold datasets;
- the formal Frozen and adversarial experiment artifacts cited by the documentation;
- the sanitized Kubernetes Ingress versus Gateway API Web rerun, its Trace bundle, HTML viewer,
  final report, online validation, and the preceding Blue Patch incident analysis.

Verify from the repository root:

```bash
shasum -a 256 -c release/v1.2.2/SHA256SUMS
```

The raw Tavily response bodies and persisted Replay state remain local and are intentionally not
redistributed. Their hashes and Replay-integrity result are retained in the public artifact
manifest.

The online rerun was executed immediately before the package-version metadata bump. Its immutable
Trace therefore records engine version `v1.2.1` plus source commit `d6805b2` and worktree patch hash
`33125a105a7201e06259eb66c7738dad15907b2927f0b0d13f175b69cea2449e`; that patch is committed in
`cb5c239`. This provenance is preserved instead of rewriting historical runtime metadata.

The rerun completed with zero final blockers, 97.96% citation coverage, 93.88% Claim support,
86.89% citation correctness, and an 83.33% deterministic Claim-Patch application rate. It remains
an auditable online case study with evidence gaps, not a replacement for the pinned 35-question ×
3-repeat Frozen benchmark.
