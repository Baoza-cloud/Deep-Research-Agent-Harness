# v1.2.1 release manifest

The Git tag `v1.2.1` is the canonical release identity. This release strengthens comparison
research and report repair with deterministic A/B evidence matrices, transactional Structured
Blue Patch integrity gates, source-version metadata, and explicit unknown-cost semantics.

`SHA256SUMS` pins four artifact groups:

- source, packaging, test, and demo-builder files changed for v1.2.1;
- ResearchBench Frozen, Live, Adversarial, and Retrieval-Gold datasets;
- the formal Frozen and adversarial experiment artifacts cited by the documentation;
- the sanitized Kubernetes Ingress versus Gateway API Web showcase, including the Trace bundle,
  self-contained HTML viewer, report, artifact manifest, one-page review, and demo script.

Verify from the repository root:

```bash
shasum -a 256 -c release/v1.2.1/SHA256SUMS
```

The raw Tavily response bodies and persisted Replay state remain local and are intentionally not
redistributed. Their hashes and Replay-integrity result are retained in the public manifest.

Formal benchmark figures remain tied to the pinned 35-question × 3-repeat ResearchBench-Frozen
run. The v1.2.1 online showcase is an auditable quality-gate incident, not a new benchmark: the
Harness correctly returned `completed_with_review_issues` after Red found a factual mismatch that
survived 100% Claim-support and citation-coverage scores.
