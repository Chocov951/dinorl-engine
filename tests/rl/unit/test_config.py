"""Strict configuration models for the RL public contract."""

import pytest

from dinorl_engine.rl.training.config import ConfigError, ExperimentConfig, PPOConfig


def _valid_mapping() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "experiment_id": "experiment-01",
        "parent_checkpoint_id": None,
        "seed": 12345,
        "budget_units": 20,
        "reward_source": "fn reward(t: Transition) -> Number { return 0.0; }",
        "ppo": {
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "entropy_coef": 0.01,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "batch_size": 256,
        },
    }


def test_experiment_configuration_is_frozen_and_round_trips_to_contract_json() -> None:
    configuration = ExperimentConfig.from_mapping(_valid_mapping())

    assert configuration.schema_version == "1.0.0"
    assert configuration.ppo == PPOConfig()
    assert configuration.to_mapping() == _valid_mapping()
    with pytest.raises(AttributeError):
        configuration.seed = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PPOConfig(learning_rate=float("inf")),
        lambda: PPOConfig(learning_rate="0.1"),  # type: ignore[arg-type]
        lambda: ExperimentConfig("invalid/id", None, 0, 1, "reward", PPOConfig()),
        lambda: ExperimentConfig("experiment-01", "invalid/id", 0, 1, "reward", PPOConfig()),
        lambda: ExperimentConfig("experiment-01", None, 0, 1, "", PPOConfig()),
        lambda: ExperimentConfig("experiment-01", None, 0, 1, "x" * (32 * 1024 + 1), PPOConfig()),
        lambda: ExperimentConfig("experiment-01", None, 0, 1, "reward", object()),  # type: ignore[arg-type]
    ],
)
def test_direct_models_enforce_the_same_strict_contract(factory: object) -> None:
    assert callable(factory)

    with pytest.raises(ConfigError):
        factory()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value.update(schema_version="2.0.0"),
        lambda value: value.update(seed=-1),
        lambda value: value.update(seed=2**32),
        lambda value: value.update(budget_units=0),
        lambda value: value["ppo"].update(learning_rate=0),
        lambda value: value["ppo"].update(learning_rate=0.2),
        lambda value: value["ppo"].update(gamma=-0.1),
        lambda value: value["ppo"].update(entropy_coef=1.1),
        lambda value: value["ppo"].update(gae_lambda=1.1),
        lambda value: value["ppo"].update(clip_range=0),
        lambda value: value["ppo"].update(batch_size=100),
    ],
)
def test_experiment_configuration_rejects_unknown_fields_and_invalid_domains(
    mutation: object,
) -> None:
    mapping = _valid_mapping()
    assert callable(mutation)
    mutation(mapping)

    with pytest.raises(ConfigError):
        ExperimentConfig.from_mapping(mapping)


def test_experiment_configuration_rejects_a_non_object_ppo() -> None:
    mapping = _valid_mapping()
    mapping["ppo"] = []

    with pytest.raises(ConfigError):
        ExperimentConfig.from_mapping(mapping)
