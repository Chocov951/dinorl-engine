"""Initial syntax contracts for the versioned Reward DSL."""

from __future__ import annotations

import pytest

from dinorl_engine.rl.rewards.dsl import (
    RewardParseError,
    RewardTypeError,
    parse_program,
    validate_program,
)


def test_parser_accepts_the_normative_reward_program() -> None:
    program = parse_program(
        """
        // Comments are ignored.
        fn safe_feed(t: Transition) -> Number {
            if feed_started(t, SELF) &&
               manhattan(t.before.self.position, t.before.opponent.position) > 3 {
                return 0.20;
            }
            return 0.00;
        }

        fn reward(t: Transition) -> Number {
            return terminal_score(t, 1.0, -1.0, 0.0)
                 + 0.05 * damage_dealt(t, SELF)
                 - 0.05 * damage_dealt(t, OPPONENT)
                 + safe_feed(t);
        }
        """
    )

    assert tuple(function.name for function in program.functions) == ("safe_feed", "reward")
    assert program.functions[-1].return_type == "Number"


def test_parser_reports_the_source_position_of_a_missing_semicolon() -> None:
    with pytest.raises(RewardParseError, match=r"REWARD_PARSE_SYNTAX at 1:49"):
        parse_program("fn reward(t: Transition) -> Number { return 0.0 }")


def test_validator_requires_the_exact_reward_entrypoint_and_rejects_cycles() -> None:
    with pytest.raises(RewardTypeError, match="REWARD_TYPE_ENTRYPOINT"):
        validate_program(parse_program("fn reward() -> Number { return 0.0; }"))

    with pytest.raises(RewardTypeError, match="REWARD_TYPE_CYCLE"):
        validate_program(
            parse_program(
                """
                fn first(t: Transition) -> Number { return second(t); }
                fn second(t: Transition) -> Number { return first(t); }
                fn reward(t: Transition) -> Number { return first(t); }
                """
            )
        )

    with pytest.raises(RewardTypeError, match="REWARD_TYPE_CYCLE"):
        validate_program(
            parse_program(
                "fn unreachable_a(t: Transition) -> Number { return unreachable_b(t); } "
                "fn unreachable_b(t: Transition) -> Number { return unreachable_a(t); } "
                "fn reward(t: Transition) -> Number { return 0.0; }"
            )
        )


def test_validator_rejects_invalid_types_and_a_missing_return_path() -> None:
    with pytest.raises(RewardTypeError, match="REWARD_TYPE_NAME"):
        validate_program(parse_program("fn reward(t: Transition) -> Number { return unknown; }"))

    with pytest.raises(RewardTypeError, match="REWARD_TYPE_RETURN"):
        validate_program(
            parse_program(
                "fn reward(t: Transition) -> Number { if t.action.success { return 1.0; } }"
            )
        )


def test_validator_accepts_each_public_raptor_field_without_opponent_movement() -> None:
    validate_program(
        parse_program(
            "fn reward(t: Transition) -> Number { "
            "if t.before.self.main_action_available && "
            "t.after.opponent.rest_pending { return t.before.self.movement_points "
            "+ t.after.opponent.hp + t.before.self.carcass_score; } return 0.0; }"
        )
    )

    validate_program(
        parse_program(
            "fn position_score(other: Transition) -> Number { return other.before.self.hp; } "
            "fn reward(t: Transition) -> Number { return position_score(t); }"
        )
    )

    with pytest.raises(RewardTypeError, match="REWARD_TYPE_FIELD"):
        validate_program(
            parse_program(
                "fn reward(t: Transition) -> Number { return t.before.opponent.movement_points; }"
            )
        )


def test_parser_and_validator_enforce_resource_limits() -> None:
    source = "fn reward(t: Transition) -> Number { return " + "(" * 65 + "0.0" + ")" * 65 + "; }"
    with pytest.raises(RewardParseError, match="REWARD_PARSE_LIMIT"):
        parse_program(source)

    functions = "\n".join(
        f"fn helper_{index}(t: Transition) -> Number {{ return helper_{index + 1}(t); }}"
        for index in range(32)
    )
    source = functions + "\nfn helper_32(t: Transition) -> Number { return 0.0; }\n"
    source += "fn reward(t: Transition) -> Number { return helper_0(t); }"
    with pytest.raises(RewardTypeError, match="REWARD_TYPE_LIMIT"):
        validate_program(parse_program(source))

    parameters = ", ".join(f"p{index}: Number" for index in range(2049))
    with pytest.raises(RewardParseError, match="REWARD_PARSE_LIMIT"):
        parse_program(f"fn helper({parameters}) -> Number {{ return 0.0; }}")
