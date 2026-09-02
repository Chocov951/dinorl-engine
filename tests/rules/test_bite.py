"""Bite legality, costs, damage and interruption tests."""

from collections.abc import Callable

import pytest

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import MAP_ID, Actor, EndReason
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.core.errors import IllegalActionError
from dinorl_engine.core.events import (
    ActionCost,
    ConsumptionInterruptedEffect,
    DamageDealtEffect,
)

type StateMutation = Callable[[DinoRLEnv], None]


def _environment() -> DinoRLEnv:
    env = DinoRLEnv(map_id=MAP_ID, seed=3)
    env.reset(first_actor=Actor.A)
    env.state.raptor(Actor.A).position = (1, 1)
    env.state.raptor(Actor.B).position = (1, 2)
    return env


def test_bite_pays_costs_deals_damage_and_keeps_turn_open() -> None:
    env = _environment()

    assert env.legal_actions()[Action.BITE] is True
    transition = env.step(Action.BITE)

    attacker = env.state.raptor(Actor.A)
    target = env.state.raptor(Actor.B)
    assert transition.action is Action.BITE
    assert transition.actor is Actor.A
    assert transition.cost == ActionCost(movement=1, endurance=2)
    assert transition.effects == (DamageDealtEffect(actor=Actor.A, target=Actor.B, amount=2),)
    assert transition.turn_ended is False
    assert attacker.movement_points == 2
    assert attacker.endurance == 3
    assert attacker.main_action_available is False
    assert target.hp == 4
    assert env.state.active_actor is Actor.A
    assert env.legal_actions()[Action.MOVE_SOUTH] is True
    assert env.legal_actions()[Action.BITE] is False


def _move_target_out_of_range(env: DinoRLEnv) -> None:
    env.state.raptor(Actor.B).position = (1, 3)


def _remove_movement_points(env: DinoRLEnv) -> None:
    env.state.raptor(Actor.A).movement_points = 0


def _remove_endurance(env: DinoRLEnv) -> None:
    env.state.raptor(Actor.A).endurance = 1


def _consume_main_action(env: DinoRLEnv) -> None:
    env.state.claim_main_action()


@pytest.mark.parametrize(
    "mutation",
    [_move_target_out_of_range, _remove_movement_points, _remove_endurance, _consume_main_action],
)
def test_illegal_bite_does_not_mutate_state(mutation: StateMutation) -> None:
    env = _environment()
    mutation(env)
    before = env.snapshot_public()

    assert env.legal_actions()[Action.BITE] is False
    with pytest.raises(IllegalActionError):
        env.step(Action.BITE)

    assert env.snapshot_public() == before


@pytest.mark.parametrize(
    ("carcass_id", "lateral_index"),
    [("carcass_left_a", 0), ("carcass_left_b", 1)],
)
def test_bite_interrupts_lateral_consumption(carcass_id: str, lateral_index: int) -> None:
    env = _environment()
    target = env.state.raptor(Actor.B)
    target.consumption_pending = carcass_id
    carcass = env.state.lateral_carcasses[lateral_index]
    carcass.status = "pending"
    carcass.pending_actor = Actor.B

    transition = env.step(Action.BITE)

    assert target.consumption_pending is None
    assert carcass.status == "available"
    assert carcass.pending_actor is None
    assert transition.effects == (
        DamageDealtEffect(actor=Actor.A, target=Actor.B, amount=2),
        ConsumptionInterruptedEffect(
            actor=Actor.A,
            target=Actor.B,
            carcass_id=carcass_id,
        ),
    )


def test_bite_interrupts_central_consumption_without_starting_recharge() -> None:
    env = _environment()
    target = env.state.raptor(Actor.B)
    target.consumption_pending = "carcass_center"
    central = env.state.central_carcass
    central.status = "pending"
    central.pending_actor = Actor.B

    transition = env.step(Action.BITE)

    assert target.consumption_pending is None
    assert central.status == "active"
    assert central.pending_actor is None
    assert central.reactivate_on_turn is None
    assert transition.effects[-1] == ConsumptionInterruptedEffect(
        actor=Actor.A,
        target=Actor.B,
        carcass_id="carcass_center",
    )


def test_bite_ko_is_immediate_and_prevents_further_actions() -> None:
    env = _environment()
    env.state.raptor(Actor.B).hp = 2

    env.step(Action.BITE)

    assert env.state.raptor(Actor.B).hp == 0
    assert env.state.terminal is True
    assert env.state.winner is Actor.A
    assert env.state.end_reason is EndReason.KO
    assert not any(env.legal_actions())
    terminal_snapshot = env.snapshot_public()
    with pytest.raises(IllegalActionError):
        env.step(Action.END_TURN)
    assert env.snapshot_public() == terminal_snapshot
