"""Executable documentation contract for the PythonAnywhere deployment."""

from pathlib import Path

ROOT = Path(__file__).parents[2]
DEPLOYMENT_GUIDE = ROOT / "deploy" / "README.md"


def test_pythonanywhere_guide_documents_reproducible_install_and_configuration() -> None:
    guide = DEPLOYMENT_GUIDE.read_text(encoding="utf-8")

    for required_text in (
        "python3.12",
        "requirements.lock",
        "pip install --no-deps -e .",
        "DINORL_ENGINE_TOKEN",
        "DINORL_ENGINE_PREVIOUS_TOKEN",
        "HTTPS",
        "Docker",
    ):
        assert required_text in guide


def test_pythonanywhere_guide_documents_smoke_tests_and_rollback() -> None:
    guide = DEPLOYMENT_GUIDE.read_text(encoding="utf-8")

    for required_text in (
        "/health",
        "/v1/matches/simulate",
        "401",
        "quatre",
        "cinq secondes",
        "Retour arrière",
    ):
        assert required_text in guide
