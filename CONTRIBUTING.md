# Contributing

## Development setup

Use Python 3.10–3.12. The locked reference environment is Python 3.12.7.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

## Required checks

```bash
ruff format --check src/research_engine evaluation scripts tests
ruff check src/research_engine evaluation scripts tests
coverage run -m pytest -q tests
coverage report
python scripts/offline_smoke.py
python -m build
```

Pull requests should explain the behavior change, include regression coverage, and keep the offline
suite independent of provider keys and network access.

## Evaluation changes

- Never modify a frozen question, evidence item, expectation, or statistical denominator to improve
  a result.
- Version benchmark changes and update the corresponding SHA256 manifest.
- Keep Frozen and Live metrics separate.
- Report uncertainty intervals and negative or non-significant results.
- Do not commit unreviewed online output as a formal result.

## Secrets and generated data

Do not commit `.env`, API keys, SQLite memory files, local indexes, caches, or unselected experiment
outputs. Follow `SECURITY.md` when a secret or private document may have been exposed.
