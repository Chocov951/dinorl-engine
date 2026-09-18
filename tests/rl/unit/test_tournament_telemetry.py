"""Per-game evaluation telemetry remains deterministic and complete."""

from dinorl_engine.core.actions import Action
from dinorl_engine.core.constants import Actor
from dinorl_engine.rl.evaluation import tournament


class _EndTurnController:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def choose_action(self, *_: object) -> Action:
        return Action.END_TURN


def test_duel_telemetry_reports_round_limit_roles_and_last_turns(monkeypatch) -> None:
    monkeypatch.setattr(tournament, "MaskablePolicyController", _EndTurnController)
    outcome = tournament.run_policy_duel(
        object(),
        object(),
        seed=9,
        left_actor=Actor.A,
        first_actor=Actor.B,
        deterministic=True,
        matchup_id="left--right",
        max_rounds=2,
        left_policy_id="left",
        right_policy_id="right",
    )
    assert outcome.draws == 1
    assert outcome.telemetry is not None
    assert outcome.telemetry["terminal_reason"] == "round_limit"
    assert outcome.telemetry["winner"] == "draw"
    assert outcome.telemetry["policies"]["left"]["role"] == "second"
    assert outcome.telemetry["policies"]["right"]["role"] == "first"
    assert len(outcome.telemetry["last_five_turns"]) == 4
    assert outcome.telemetry["final"]["A"]["nutrition_pending"] is None
