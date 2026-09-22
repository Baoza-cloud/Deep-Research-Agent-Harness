"""Allow ``python -m research_engine`` to invoke the supported CLI."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
