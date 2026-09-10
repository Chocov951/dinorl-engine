"""Snapshot tests for the deterministic terminal renderer."""

from dinorl_engine.core.constants import Actor
from dinorl_engine.core.engine import DinoRLEnv
from dinorl_engine.debug.text_renderer import render_state


def _initial_state() -> DinoRLEnv:
    env = DinoRLEnv("arena_mvp_v1", seed=7)
    env.reset(first_actor=Actor.A)
    return env


def test_initial_state_has_a_stable_complete_text_rendering() -> None:
    env = _initial_state()

    assert render_state(env.snapshot_public()) == (
        "Round 1 — Turn 0 — Actor A\n"
        "A: HP=6 END=5 PM=3 SCORE=0 MAIN=yes\n"
        "B: HP=6 END=5 PM=0 SCORE=0 MAIN=yes\n"
        "\n"
        ". . . . . . . . B\n"
        ". . . . . / . . .\n"
        ". . / . . . . ~ ~\n"
        "l . ~ . ~ / . . .\n"
        "/ / . . C . . / /\n"
        ". . . / ~ . ~ . l\n"
        "~ ~ . . . . / . .\n"
        ". . . / . . . . .\n"
        "A . . . . . . . .\n"
        "\n"
        "Central carcass: active\n"
        "Pending feed: none\n"
        "Pending rest: none\n"
        "Last action: none\n"
    )


def test_raptors_mask_tiles_while_panels_keep_pending_state_visible() -> None:
    env = _initial_state()
    state = env.state
    state.raptor(Actor.A).position = (4, 4)
    state.raptor(Actor.A).consumption_pending = "carcass_center"
    state.raptor(Actor.B).rest_pending = True
    state.central_carcass.status = "recharging"
    state.central_carcass.pending_actor = Actor.A
    state.central_carcass.reactivate_on_turn = 2
    event = {
        "type": "action_resolved",
        "actor": "B",
        "action": "SHOVE",
        "effects": [{"type": "target_shoved", "actor": "B", "target": "A"}],
    }

    rendered = render_state(env.snapshot_public(), event=event)

    assert rendered.splitlines()[8] == "/ / . . A . . / /"
    assert "Central carcass: recharging" in rendered
    assert "Pending feed: A carcass_center" in rendered
    assert "Pending rest: B" in rendered
    assert rendered.endswith("Last action: B SHOVE A\n")


def test_renderer_is_pure_deterministic_and_contains_no_ansi_sequences() -> None:
    snapshot = _initial_state().snapshot_public()

    first = render_state(snapshot)
    second = render_state(snapshot)

    assert first == second
    assert "\x1b" not in first
