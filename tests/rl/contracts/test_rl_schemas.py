"""Versioned RL JSON Schema contract tests."""

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[3]
SCHEMA_DIRECTORY = ROOT / "schemas" / "rl"
FIXTURE_DIRECTORY = ROOT / "fixtures" / "rl" / "contracts"
SCHEMA_FILES = (
    "experiment-v1.schema.json",
    "run-v1.schema.json",
    "progress-v1.schema.json",
    "evaluation-v1.schema.json",
    "checkpoint-manifest-v1.schema.json",
    "snapshot-manifest-v1.schema.json",
)

type JsonObject = dict[str, Any]
type Mutation = Callable[[JsonObject], None]


def _read_json(path: Path) -> JsonObject:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def schemas() -> dict[str, JsonObject]:
    return {name: _read_json(SCHEMA_DIRECTORY / name) for name in SCHEMA_FILES}


@pytest.fixture(scope="module")
def validators(schemas: dict[str, JsonObject]) -> dict[str, Draft202012Validator]:
    return {name: Draft202012Validator(schema) for name, schema in schemas.items()}


def test_schemas_are_valid_draft_2020_12(schemas: dict[str, JsonObject]) -> None:
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("schema_name", SCHEMA_FILES)
def test_valid_fixture_is_accepted(
    validators: dict[str, Draft202012Validator], schema_name: str
) -> None:
    fixture_name = schema_name.removesuffix(".schema.json") + ".valid.json"
    instance = _read_json(FIXTURE_DIRECTORY / fixture_name)

    assert list(validators[schema_name].iter_errors(instance)) == []


@pytest.mark.parametrize("schema_name", SCHEMA_FILES)
@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(unexpected=True),
        lambda value: value.update(schema_version="2.0.0"),
    ],
)
def test_unknown_properties_and_wrong_versions_are_rejected(
    validators: dict[str, Draft202012Validator], schema_name: str, mutation: Mutation
) -> None:
    fixture_name = schema_name.removesuffix(".schema.json") + ".valid.json"
    instance = copy.deepcopy(_read_json(FIXTURE_DIRECTORY / fixture_name))
    mutation(instance)

    assert list(validators[schema_name].iter_errors(instance))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("ppo", "learning_rate"), 0),
        (("ppo", "learning_rate"), 0.1000001),
        (("ppo", "gamma"), -0.001),
        (("ppo", "entropy_coef"), 1.001),
        (("ppo", "gae_lambda"), 1.001),
        (("ppo", "clip_range"), 0),
        (("ppo", "batch_size"), 100),
        (("seed",), -1),
        (("seed",), 2**32),
        (("budget_units",), 0),
    ],
)
def test_experiment_hyperparameter_domains_are_enforced(
    validators: dict[str, Draft202012Validator], path: tuple[str, ...], value: object
) -> None:
    instance = _read_json(FIXTURE_DIRECTORY / "experiment-v1.valid.json")
    target: JsonObject = instance
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value

    assert list(validators["experiment-v1.schema.json"].iter_errors(instance))
