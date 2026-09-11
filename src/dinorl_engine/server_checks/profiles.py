"""Versioned dependency profiles for auditable server gate environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "PYTHONANYWHERE_PROFILE",
    "STANDARD_PROFILE",
    "DependencyProfile",
    "ProfileError",
    "get_profile",
    "profile_names",
    "runtime_version_matches",
]


class ProfileError(ValueError):
    """Raised when a requested server dependency profile is unknown."""


@dataclass(frozen=True, slots=True)
class DependencyProfile:
    """One pinned dependency set accepted by a server-gate result."""

    name: str
    runtime_lock_filename: str
    provenance_lock_filenames: tuple[str, ...]
    requires_system_site_packages: bool = False


STANDARD_PROFILE: Final = DependencyProfile(
    "standard",
    "requirements.lock",
    ("requirements.lock", "requirements-dev.lock"),
)
PYTHONANYWHERE_PROFILE: Final = DependencyProfile(
    "pythonanywhere",
    "requirements-pythonanywhere.lock",
    ("requirements-pythonanywhere.lock",),
    requires_system_site_packages=True,
)
_PROFILES: Final = {
    STANDARD_PROFILE.name: STANDARD_PROFILE,
    PYTHONANYWHERE_PROFILE.name: PYTHONANYWHERE_PROFILE,
}


def profile_names() -> tuple[str, ...]:
    """Return the stable profile names accepted by the CLI."""

    return tuple(_PROFILES)


def get_profile(name: str) -> DependencyProfile:
    """Return a known profile without accepting an arbitrary lockfile path."""

    try:
        return _PROFILES[name]
    except KeyError as error:
        raise ProfileError(f"unsupported dependency profile: {name}") from error


def runtime_version_matches(installed: str, expected: str) -> bool:
    """Match a public pin while allowing a local build suffix.

    PythonAnywhere exposes its CPU Torch build as ``2.3.1+cpu`` while the
    public requirement is ``torch==2.3.1``.
    """

    return installed == expected or installed.partition("+")[0] == expected
