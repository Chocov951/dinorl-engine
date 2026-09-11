"""Module entry point for server-check tooling."""

from dinorl_engine.server_checks.cli import main

if __name__ == "__main__":  # pragma: no cover - exercised through subprocesses
    raise SystemExit(main())
