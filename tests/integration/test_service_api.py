"""HTTP service integration tests."""

from flask import Flask

from dinorl_engine.core.constants import ENGINE_VERSION, MAP_ID, PROTOCOL_VERSION, RULES_VERSION
from dinorl_engine.service.api import create_app


def test_health_returns_the_exact_public_service_description() -> None:
    client = create_app().test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.content_type == "application/json"
    assert response.get_json() == {
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "protocol_versions": [PROTOCOL_VERSION],
        "rules_versions": [RULES_VERSION],
        "maps": [MAP_ID],
    }


def test_pythonanywhere_wsgi_application_is_importable() -> None:
    from deploy.pythonanywhere_wsgi import application

    assert isinstance(application, Flask)
