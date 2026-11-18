from dinorl_engine.rl.s7_calibration import extract_curve_times, propose_daily_budget


def test_s7_extracts_visible_gate_and_plateau_from_a_fixed_curve() -> None:
    points = [
        {"unit": 5, "score": 0.35},
        {"unit": 10, "score": 0.4},
        {"unit": 15, "score": 0.56},
        {"unit": 20, "score": 0.565},
        {"unit": 25, "score": 0.57},
        {"unit": 30, "score": 0.575},
    ]
    assert extract_curve_times(points, visible_score=0.4, gate_score=0.55, plateau_delta=0.01) == {
        "T_visible": 10,
        "T_gate": 15,
        "T_plateau": 20,
    }


def test_s7_does_not_call_one_noisy_interval_a_plateau() -> None:
    points = [
        {"unit": 5, "score": 0.4},
        {"unit": 10, "score": 0.405},
        {"unit": 15, "score": 0.45},
        {"unit": 20, "score": 0.455},
    ]
    assert (
        extract_curve_times(points, visible_score=0.4, gate_score=0.55, plateau_delta=0.01)[
            "T_plateau"
        ]
        is None
    )


def test_s7_budget_obeys_the_documented_inequality() -> None:
    assert propose_daily_budget(visible=20, plateau=100) == 49
