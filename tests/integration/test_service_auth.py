"""Authentication and request-limit integration tests."""

from collections.abc import Callable

import pytest
from flask.testing import FlaskClient

from dinorl_engine.service import auth
from dinorl_engine.service.api import create_app

CURRENT_TOKEN = "c" * 32
PREVIOUS_TOKEN = "p" * 32
SIMULATE_PATH = "/v1/matches/simulate"


@pytest.fixture
def client() -> FlaskClient:
    app = create_app(
        {
            "DINORL_ENGINE_TOKEN": CURRENT_TOKEN,
            "DINORL_ENGINE_PREVIOUS_TOKEN": PREVIOUS_TOKEN,
        }
    )
    return app.test_client()


@pytest.mark.parametrize(
    "authorization",
    [None, "", "Basic credentials", "Bearer wrong", "Bearer"],
)
def test_simulation_fails_closed_without_a_valid_bearer_token(
    client: FlaskClient, authorization: str | None
) -> None:
    headers = {} if authorization is None else {"Authorization": authorization}

    response = client.post(SIMULATE_PATH, json={}, headers=headers)

    assert response.status_code == 401
    assert response.get_json() == {
        "error": {
            "code": "unauthorized",
            "message": "A valid bearer token is required.",
            "request_id": None,
        }
    }
    assert CURRENT_TOKEN not in response.get_data(as_text=True)
    assert PREVIOUS_TOKEN not in response.get_data(as_text=True)


@pytest.mark.parametrize("token", [CURRENT_TOKEN, PREVIOUS_TOKEN])
def test_current_and_previous_tokens_are_accepted(client: FlaskClient, token: str) -> None:
    response = client.post(
        SIMULATE_PATH,
        json={},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code != 401


def test_both_configured_tokens_are_compared_even_when_current_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparisons: list[tuple[bytes, bytes]] = []
    original: Callable[[bytes, bytes], bool] = auth.compare_digest

    def recording_compare(left: bytes, right: bytes) -> bool:
        comparisons.append((left, right))
        return original(left, right)

    monkeypatch.setattr(auth, "compare_digest", recording_compare)
    client = create_app(
        {
            "DINORL_ENGINE_TOKEN": CURRENT_TOKEN,
            "DINORL_ENGINE_PREVIOUS_TOKEN": PREVIOUS_TOKEN,
        }
    ).test_client()

    client.post(
        SIMULATE_PATH,
        json={},
        headers={"Authorization": f"Bearer {CURRENT_TOKEN}"},
    )

    assert len(comparisons) == 2


def test_request_body_over_32_kib_is_rejected_before_processing(client: FlaskClient) -> None:
    response = client.post(
        SIMULATE_PATH,
        data=b"x" * (32 * 1024 + 1),
        headers={
            "Authorization": f"Bearer {CURRENT_TOKEN}",
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 413
    assert response.get_json()["error"]["code"] == "request_too_large"


def test_request_limit_cannot_be_relaxed_by_application_configuration() -> None:
    app = create_app(
        {
            "DINORL_ENGINE_TOKEN": CURRENT_TOKEN,
            "MAX_CONTENT_LENGTH": 1024 * 1024,
        }
    )

    assert app.config["MAX_CONTENT_LENGTH"] == 32 * 1024


@pytest.mark.parametrize(
    "configuration",
    [
        {"DINORL_ENGINE_TOKEN": "short"},
        {
            "DINORL_ENGINE_TOKEN": CURRENT_TOKEN,
            "DINORL_ENGINE_PREVIOUS_TOKEN": "short",
        },
    ],
)
def test_configured_tokens_must_contain_at_least_32_bytes(configuration: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        create_app(configuration)


def test_health_remains_public_when_no_token_is_configured() -> None:
    assert create_app().test_client().get("/health").status_code == 200
