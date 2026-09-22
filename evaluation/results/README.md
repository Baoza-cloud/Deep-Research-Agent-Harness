# Versioned evaluation artifacts

This directory intentionally keeps only the canonical v1.0 release artifacts in Git.
Exploratory runs, smoke tests, failed resumptions, targeted reruns, and local diagnostics
are generated files and are ignored by default.

## Canonical ResearchBench-Frozen v1.0 run

- Dataset: 35 questions across 11 domains, three repeats per question.
- Variants: `fixed_harness` and `dynamic_swarm`.
- Model: `deepseek-v4-flash`.
- Final result: `formal_frozen_v1_35x3/ablation-20260922T135223799454Z.json`.
- Human-readable summary: `formal_frozen_v1_35x3/ablation-20260922T135223799454Z-summary.md`.
- Strict gate: the matching `quality-gate.json` and `quality-gate.md` files.
- Provenance: `frozen_release_20260921_v1/freeze-manifest.json` and
  `frozen_release_20260921_v1/formal-run-launch.json`.

The launch record discloses the resume history and the single targeted quality-gate
rerun. Do not combine ignored pilot artifacts with the canonical result when reporting
metrics.

## Curated adversarial summaries

The two Markdown reports under `adversarial/` preserve the formal 35-question ×
three-repeat Red/Blue evaluation and the deterministic Structured Patch comparison.
Their large intermediate/raw JSON files remain ignored because the Frozen release above
is the canonical raw artifact for v1.0.
