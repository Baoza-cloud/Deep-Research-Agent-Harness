# v1.2.0 release manifest

The Git tag `v1.2.0` is the canonical release identity. This release adds durable Run/DAG
recovery and deterministic Replay, plus versioned structured tracing and the local HTML Trace
Viewer.

`SHA256SUMS` pins four artifact groups:

- source and packaging files used by the Harness and local RAG;
- ResearchBench Frozen, Live, Adversarial, and Retrieval-Gold datasets;
- selected formal Frozen and adversarial experiment artifacts cited by the documentation;
- the complete real-Web interview showcase, including its raw run, Trace bundle, report, manifest,
  demo script, incident review, results sheet, and interview questions.

Verify from the repository root:

```bash
shasum -a 256 -c release/v1.2.0/SHA256SUMS
```

The GitHub Release records the exact tag commit SHA and publishes the wheel, source distribution,
`SHA256SUMS`, and a separate checksum file for those uploaded release assets.

Formal quality figures remain tied to the pinned 35-question × 3-repeat ResearchBench-Frozen run.
The recovery and tracing additions are covered by offline regression, replay-integrity, package,
and compatibility gates; they do not relabel the earlier benchmark as a newly executed v1.2.0 run.
