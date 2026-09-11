"""Versioned, auditable server dependency profiles."""

import hashlib
from pathlib import Path

import dinorl_engine.server_checks.runner as runner
from dinorl_engine.server_checks.profiles import (
    PYTHONANYWHERE_PROFILE,
    STANDARD_PROFILE,
    runtime_version_matches,
)


def test_pythonanywhere_profile_uses_its_own_pinned_lock_and_accepts_cpu_torch_build() -> None:
    assert STANDARD_PROFILE.runtime_lock_filename == "requirements.lock"
    assert PYTHONANYWHERE_PROFILE.runtime_lock_filename == "requirements-pythonanywhere.lock"
    assert PYTHONANYWHERE_PROFILE.requires_system_site_packages
    assert runtime_version_matches("2.3.1+cpu", "2.3.1")
    assert not runtime_version_matches("2.3.0+cpu", "2.3.1")


def test_pythonanywhere_archive_provenance_accepts_the_cpu_torch_build(
    tmp_path: Path, monkeypatch: object
) -> None:
    lock_path = tmp_path / PYTHONANYWHERE_PROFILE.runtime_lock_filename
    lock_contents = "torch==2.3.1\n"
    lock_path.write_text(lock_contents, encoding="utf-8")
    monkeypatch.setattr(runner, "_git", lambda *_args: "commit")  # type: ignore[attr-defined]

    runner.verify_import_provenance(
        tmp_path,
        PYTHONANYWHERE_PROFILE,
        {
            "git_commit": "commit",
            "lock_sha256": {
                "requirements-pythonanywhere.lock": hashlib.sha256(
                    lock_path.read_bytes()
                ).hexdigest()
            },
            "installed_packages": {"torch": "2.3.1+cpu"},
        },
    )
