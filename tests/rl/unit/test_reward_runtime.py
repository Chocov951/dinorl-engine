"""Reference interpreter and bounded compiled Reward DSL contracts."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dinorl_engine.rl.rewards.reference import REFERENCE_REWARD_SOURCE, reference_reward_public
from dinorl_engine.rl.rewards.runtime import RewardRuntimeError, compile_reward


@given(
    st.lists(
        st.tuples(st.sampled_from(("+", "-", "*", "/")), st.integers(min_value=1, max_value=5)),
        max_size=5,
    )
)
def test_generated_arithmetic_programs_match_between_ast_and_vm(
    terms: list[tuple[str, int]],
) -> None:
    expression = "1.0" + "".join(f" {operator} {value}.0" for operator, value in terms)
    reward = compile_reward(f"fn reward(t: Transition) -> Number {{ return {expression}; }}")

    assert reward.evaluate_vm({}) == reward.evaluate_reference({})


def test_compiled_vm_matches_the_reference_for_terminal_and_event_rewards() -> None:
    reward = compile_reward(
        """
        fn reward(t: Transition) -> Number {
            return terminal_score(t, 1.0, -1.0, 0.0)
                 + 0.05 * damage_dealt(t, SELF)
                 - 0.05 * damage_dealt(t, OPPONENT);
        }
        """
    )
    transition = {
        "outcome": "WIN",
        "events": {"damage_dealt": {"SELF": 2.0, "OPPONENT": 1.0}},
        "action": {"success": True},
        "before": {"self": {"position": (0, 0)}, "opponent": {"position": (1, 0)}},
    }

    assert reward.evaluate_reference(transition) == 1.05
    assert reward.evaluate_vm(transition) == 1.05
    assert reward.cache_key == compile_reward(reward.source).cache_key


def test_reference_reward_uses_the_closed_specialized_vm_instruction() -> None:
    reward = compile_reward(REFERENCE_REWARD_SOURCE)

    assert reward.reference_fast_path is True
    assert reward.vm_evaluator is reference_reward_public
    assert reward.bytecode[0].instructions[0].opcode == "REFERENCE_REWARD"


def test_vm_preserves_boolean_short_circuit() -> None:
    reward = compile_reward(
        "fn reward(t: Transition) -> Number { "
        "if t.action.success && 1.0 / 0.0 > 0.0 { return 1.0; } return 0.0; }"
    )

    transition = {"action": {"success": False}}

    assert reward.evaluate_reference(transition) == 0.0
    assert reward.evaluate_vm(transition) == 0.0


def test_cache_is_independent_of_comments_and_whitespace() -> None:
    compact = "fn reward(t: Transition) -> Number { return 1.0; }"
    documented = "// reward\nfn reward(t: Transition)->Number{return 1.0;}"

    assert compile_reward(compact).cache_key == compile_reward(documented).cache_key


def test_spatial_builtins_use_public_geometry_and_are_lazy() -> None:
    reward = compile_reward(
        "fn reward(t: Transition) -> Number { "
        "if t.action.success { return path_steps(t.before.self.position, "
        "t.before.opponent.position) + movement_cost(t.before.self.position, "
        "t.before.opponent.position); } return 0.0; }"
    )
    transition = {
        "action": {"success": True},
        "before": {"self": {"position": (0, 0)}, "opponent": {"position": (0, 2)}},
        "map": {"rows": 1, "columns": 3, "walls": (), "mud": ((0, 1),)},
    }
    assert reward.evaluate_reference(transition) == 5.0
    assert reward.evaluate_vm(transition) == 5.0

    lazy = compile_reward(
        "fn reward(t: Transition) -> Number { "
        "if t.action.success && path_steps(t.before.self.position, "
        "t.before.opponent.position) > 0.0 { return 1.0; } return 0.0; }"
    )
    assert lazy.evaluate_vm({"action": {"success": False}}) == 0.0


def test_normative_safe_feed_example_runs_in_reference_and_vm() -> None:
    reward = compile_reward(
        "fn safe_feed(t: Transition) -> Number { "
        "if feed_started(t, SELF) && "
        "manhattan(t.before.self.position, t.before.opponent.position) > 3.0 "
        "{ return 0.20; } return 0.00; } "
        "fn reward(t: Transition) -> Number { return safe_feed(t); }"
    )
    transition = {
        "events": {"feed_started": {"SELF": True, "OPPONENT": False}},
        "before": {"self": {"position": (0, 0)}, "opponent": {"position": (0, 4)}},
    }

    assert reward.evaluate_reference(transition) == 0.2
    assert reward.evaluate_vm(transition) == 0.2


def test_ast_and_vm_report_the_same_runtime_error_code() -> None:
    reward = compile_reward("fn reward(t: Transition) -> Number { return 1.0 / 0.0; }")

    with pytest.raises(RewardRuntimeError, match="REWARD_RUNTIME_DIVISION"):
        reward.evaluate_reference({})
    with pytest.raises(RewardRuntimeError, match="REWARD_RUNTIME_DIVISION"):
        reward.evaluate_vm({})


def test_ast_and_vm_abort_a_nonfinite_intermediate() -> None:
    huge = "1" + "0" * 308
    reward = compile_reward(f"fn reward(t: Transition) -> Number {{ return {huge} * {huge}; }}")

    with pytest.raises(RewardRuntimeError, match="REWARD_RUNTIME_NONFINITE"):
        reward.evaluate_reference({})
    with pytest.raises(RewardRuntimeError, match="REWARD_RUNTIME_NONFINITE"):
        reward.evaluate_vm({})


def test_vm_exposes_named_terms_without_a_second_program_evaluation() -> None:
    reward = compile_reward(
        "fn damage_bonus(t: Transition) -> Number { return damage_dealt(t, SELF); } "
        "fn reward(t: Transition) -> Number { "
        "let bonus: Number = damage_bonus(t); return bonus + 1.0; }"
    )

    value, terms = reward.evaluate_vm_terms(
        {"events": {"damage_dealt": {"SELF": 2.0, "OPPONENT": 0.0}}}
    )

    assert value == 3.0
    assert terms == {"damage_bonus": 2.0, "reward.bonus": 2.0}
