"""Local and server preflight for the reproducible RL-S5c-A2 campaign."""

from __future__ import annotations

from pathlib import Path

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.env.single_agent import DinoRLSingleAgentEnv
from dinorl_engine.rl.evaluation.policy import MaskablePolicyController
from dinorl_engine.rl.policies.local_mlp_v2 import LocalMLPV2Architecture
from dinorl_engine.rl.s5c.config import S5cConfig
from dinorl_engine.rl.s5c.provenance import (
    build_campaign_provenance,
    initialization_record,
    write_immutable_manifest,
)
from dinorl_engine.rl.s5c.rewards import ARCHETYPES, specialist_rewards
from dinorl_engine.rl.s5c.strength import verify_strong_pool
from dinorl_engine.rl.training.unit import create_maskable_ppo

__all__ = ["run_preflight"]


def _initialization_checks(config: S5cConfig) -> dict[str, object]:
    records: dict[str, object] = {}
    seed_hashes: dict[int, set[str]] = {seed: set() for seed in config.calibration_seeds}
    for seed in config.calibration_seeds:
        for archetype in ARCHETYPES:
            environment = DinoRLSingleAgentEnv(seed=seed)
            try:
                model = create_maskable_ppo(
                    environment,
                    seed=seed,
                    architecture=LocalMLPV2Architecture.COMPACT,
                )
                record = initialization_record(model, seed=seed)
            finally:
                environment.close()
            records[f"{archetype}:{seed}"] = record
            seed_hashes[seed].add(str(record["weights_sha256"]))
    if any(len(hashes) != 1 for hashes in seed_hashes.values()):
        raise RuntimeError("specialists do not share identical initial weights by seed")
    if len({next(iter(hashes)) for hashes in seed_hashes.values()}) != len(seed_hashes):
        raise RuntimeError("distinct training seeds produced identical initial weights")
    return records


def _controller_smoke() -> dict[str, object]:
    environment = DinoRLSingleAgentEnv(seed=19)
    model = create_maskable_ppo(
        environment,
        seed=19,
        architecture=LocalMLPV2Architecture.COMPACT,
    )
    try:
        environment.reset(options={"learner_actor": "A", "first_actor": "A"})
        environment.engine.state.raptor(Actor.A).position = (1, 1)
        environment.engine.state.raptor(Actor.B).position = (1, 2)
        legal = environment.engine.legal_actions()
        if not legal[Action.SHOVE] or not bool(environment.action_masks()[Action.SHOVE]):
            raise RuntimeError("SHOVE is legal in the engine but absent from the policy mask")
        controller = MaskablePolicyController(
            model,
            learner_actor=Actor.A,
            first_actor=Actor.A,
            deterministic=False,
            stochastic_seed=19,
        )
        actions = [
            controller.choose_action(environment.engine.snapshot_public(), legal)
            for _ in range(256)
        ]
        shoves = actions.count(Action.SHOVE)
        if shoves == 0:
            raise RuntimeError(
                "fresh policy never sampled a legal SHOVE in 256 deterministic draws"
            )
        return {"draws": len(actions), "shoves": shoves, "status": "passed"}
    finally:
        environment.close()


def run_preflight(
    config: S5cConfig,
    *,
    repository: Path,
    allow_dirty: bool,
) -> dict[str, object]:
    """Verify all non-performance A2 prerequisites and freeze the resolved manifest."""

    if config.phase != "RL-S5c-A2":
        raise ValueError("preflight requires an RL-S5c-A2 configuration")
    if config.rl_s5b_pool is None or config.rl_s5b_pool_sha256 is None:
        raise ValueError("preflight requires a frozen RL-S5b pool")
    pool = verify_strong_pool(
        config.rl_s5b_pool,
        expected_sha256=config.rl_s5b_pool_sha256,
    )
    rewards = specialist_rewards()
    provenance = build_campaign_provenance(
        repository=repository,
        config_sha256=config.resolved_sha256,
        reward_sha256={name: reward.cache_key for name, reward in rewards.items()},
        pool_sha256=str(pool["pool_sha256"]),
        allow_dirty=allow_dirty,
    )
    initializations = _initialization_checks(config)
    controller = _controller_smoke()
    manifest = {
        "format": "s5c-a2-run-manifest-v1",
        "phase": config.phase,
        "run_id": config.run_id,
        "config": config.to_mapping(),
        "config_sha256": config.resolved_sha256,
        "provenance": provenance,
        "initialization_preflight": initializations,
        "controller_smoke": controller,
        "reward_sha256": {name: reward.cache_key for name, reward in rewards.items()},
        "rl_s5b_pool_sha256": pool["pool_sha256"],
        "status": "READY",
    }
    write_immutable_manifest(config.output_directory / "manifest.json", manifest)
    return manifest
