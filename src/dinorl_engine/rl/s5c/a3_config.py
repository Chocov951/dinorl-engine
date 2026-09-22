"""Strict configuration for the isolated RL-S5c-A3 campaign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from dinorl_engine.rl.contracts import canonical_json_bytes


@dataclass(frozen=True, slots=True)
class A3Config:
    run_id: str
    output_directory: Path
    architecture: str
    max_rounds: int
    control_seed: int
    pilot_seed: int
    confirm_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    control_max_units: int
    specialist_max_units: int
    bootstrap_max_units: int
    rl_s5b_pool: Path
    rl_s5b_pool_sha256: str
    safe_feed_episode_cap: float

    @classmethod
    def load(cls, path: Path) -> A3Config:
        value = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "format",
            "phase",
            "run_id",
            "output_directory",
            "architecture",
            "max_rounds",
            "control_seed",
            "pilot_seed",
            "confirm_seeds",
            "evaluation_seeds",
            "control_max_units",
            "specialist_max_units",
            "bootstrap_max_units",
            "rl_s5b_pool",
            "rl_s5b_pool_sha256",
            "safe_feed_episode_cap",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("A3 config has missing or unexpected fields")
        if value["format"] != "dinorl-s5c-a3-config-v1" or value["phase"] != "RL-S5c-A3":
            raise ValueError("A3 config format or phase is invalid")
        base = path.parent.resolve()
        output = Path(value["output_directory"])
        pool = Path(value["rl_s5b_pool"])
        result = cls(
            run_id=str(value["run_id"]),
            output_directory=(base / output).resolve(),
            architecture=str(value["architecture"]),
            max_rounds=int(value["max_rounds"]),
            control_seed=int(value["control_seed"]),
            pilot_seed=int(value["pilot_seed"]),
            confirm_seeds=tuple(value["confirm_seeds"]),
            evaluation_seeds=tuple(value["evaluation_seeds"]),
            control_max_units=int(value["control_max_units"]),
            specialist_max_units=int(value["specialist_max_units"]),
            bootstrap_max_units=int(value["bootstrap_max_units"]),
            rl_s5b_pool=(base / pool).resolve(),
            rl_s5b_pool_sha256=str(value["rl_s5b_pool_sha256"]),
            safe_feed_episode_cap=float(value["safe_feed_episode_cap"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.architecture != "mlp-compact-v2" or self.max_rounds != 30:
            raise ValueError("A3 freezes mlp-compact-v2 and 30 rounds")
        if self.control_seed != 20 or self.pilot_seed != 20:
            raise ValueError("A3 control and pilot seed must be 20")
        if self.confirm_seeds != (19, 20, 21) or len(set(self.evaluation_seeds)) != 3:
            raise ValueError("A3 seed sets are invalid")
        if (self.control_max_units, self.specialist_max_units, self.bootstrap_max_units) != (
            147,
            197,
            147,
        ):
            raise ValueError("A3 budgets are invalid")
        if len(self.rl_s5b_pool_sha256) != 64 or self.safe_feed_episode_cap != 0.30:
            raise ValueError("A3 pool hash or safe-feed cap is invalid")

    @property
    def mapping(self) -> dict[str, object]:
        return {
            "format": "dinorl-s5c-a3-config-v1",
            "phase": "RL-S5c-A3",
            "run_id": self.run_id,
            "output_directory": str(self.output_directory),
            "architecture": self.architecture,
            "max_rounds": self.max_rounds,
            "control_seed": self.control_seed,
            "pilot_seed": self.pilot_seed,
            "confirm_seeds": list(self.confirm_seeds),
            "evaluation_seeds": list(self.evaluation_seeds),
            "control_max_units": self.control_max_units,
            "specialist_max_units": self.specialist_max_units,
            "bootstrap_max_units": self.bootstrap_max_units,
            "rl_s5b_pool": str(self.rl_s5b_pool),
            "rl_s5b_pool_sha256": self.rl_s5b_pool_sha256,
            "safe_feed_episode_cap": self.safe_feed_episode_cap,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.mapping)).hexdigest()
