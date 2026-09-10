"""Integration tests for local debugging CLI commands."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dinorl_engine.cli import main

ROOT = Path(__file__).parents[2]
VALID_REPLAY = ROOT / "fixtures" / "contracts" / "replay-v1.valid.json"


def test_inspect_map_renders_the_stable_initial_arena() -> None:
    output: list[str] = []

    exit_code = main(["inspect-map"], output=output.append)

    assert exit_code == 0
    assert len(output) == 1
    assert output[0].startswith("Round 1 — Turn 0 — Actor A\n")
    assert ". . . . . . . . B\n" in output[0]
    assert output[0].endswith("Last action: none\n")


def test_replay_can_render_one_selected_event_without_waiting() -> None:
    output: list[str] = []
    sleeps: list[float] = []

    exit_code = main(
        ["replay", str(VALID_REPLAY), "--event", "2"],
        output=output.append,
        sleeper=sleeps.append,
    )

    assert exit_code == 0
    assert len(output) == 1
    assert output[0].endswith("Last action: A BITE B\n")
    assert sleeps == []


def test_replay_speed_controls_injected_delays_between_events() -> None:
    output: list[str] = []
    sleeps: list[float] = []

    exit_code = main(
        ["replay", str(VALID_REPLAY), "--speed", "2"],
        output=output.append,
        sleeper=sleeps.append,
    )

    assert exit_code == 0
    assert len(output) == 4
    assert sleeps == [0.5, 0.5, 0.5]


def test_replay_step_uses_injected_reads_and_never_sleeps() -> None:
    reads: list[str] = []
    sleeps: list[float] = []

    exit_code = main(
        ["replay", str(VALID_REPLAY), "--step"],
        output=lambda _text: None,
        reader=lambda: reads.append("read") or "",
        sleeper=sleeps.append,
    )

    assert exit_code == 0
    assert reads == ["read", "read", "read"]
    assert sleeps == []


def test_invalid_replay_is_rejected_before_any_display(tmp_path: Path) -> None:
    replay: dict[str, Any] = json.loads(VALID_REPLAY.read_text(encoding="utf-8"))
    replay["replay_version"] = "9.0.0"
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(replay), encoding="utf-8")
    output: list[str] = []
    errors: list[str] = []

    exit_code = main(
        ["replay", str(invalid_path)],
        output=output.append,
        error_output=errors.append,
    )

    assert exit_code == 1
    assert output == []
    assert errors and errors[0].startswith("Replay error:")


def test_replay_rejects_an_event_index_outside_the_validated_document() -> None:
    output: list[str] = []
    errors: list[str] = []

    exit_code = main(
        ["replay", str(VALID_REPLAY), "--event", "42"],
        output=output.append,
        error_output=errors.append,
    )

    assert exit_code == 1
    assert output == []
    assert errors == ["Replay error: event index 42 is out of range\n"]


def test_play_supports_manual_against_bot_and_manual_against_manual() -> None:
    for controller_a, controller_b, response in (
        ("manual", "opportunist-v1", "REST"),
        ("manual", "manual", "END_TURN"),
    ):
        output: list[str] = []
        exit_code = main(
            [
                "play",
                "--seed",
                "5",
                "--controller-a",
                controller_a,
                "--controller-b",
                controller_b,
            ],
            output=output.append,
            reader=lambda selected=response: selected,
        )

        assert exit_code == 0
        summary = json.loads(output[-1])
        assert summary["result"]["actions"] >= 1
        assert summary["replay"] is None


def test_manual_cancellation_returns_a_nonzero_exit_code() -> None:
    errors: list[str] = []

    exit_code = main(
        [
            "play",
            "--controller-a",
            "manual",
            "--controller-b",
            "aggressive-v1",
        ],
        output=lambda _text: None,
        error_output=errors.append,
        reader=lambda: (_ for _ in ()).throw(EOFError()),
    )

    assert exit_code == 1
    assert errors == ["Match cancelled by manual controller\n"]


def _write_end_sequence(path: Path, controller_id: str, action: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "controller_id": controller_id,
                "turns": [{"actions": [action]} for _ in range(30)],
            }
        ),
        encoding="utf-8",
    )


def test_sequence_play_writes_a_path_free_replay(tmp_path: Path) -> None:
    sequence_path = tmp_path / "local plan.json"
    replay_path = tmp_path / "replay.json"
    _write_end_sequence(sequence_path, "resting-a", "REST")
    output: list[str] = []

    exit_code = main(
        [
            "play",
            "--seed",
            "5",
            "--controller-a",
            f"sequence:{sequence_path}",
            "--controller-b",
            "aggressive-v1",
            "--replay-out",
            str(replay_path),
        ],
        output=output.append,
    )

    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert replay["controllers"]["A"] == {"kind": "sequence", "id": "resting-a"}
    assert str(sequence_path) not in replay_path.read_text(encoding="utf-8")
    assert json.loads(output[-1])["replay_sha256"] is not None


def test_two_sequence_files_can_drive_a_complete_cli_match(tmp_path: Path) -> None:
    sequence_a = tmp_path / "a.json"
    sequence_b = tmp_path / "b.json"
    replay_path = tmp_path / "replay.json"
    _write_end_sequence(sequence_a, "end-a", "END_TURN")
    _write_end_sequence(sequence_b, "end-b", "END_TURN")

    exit_code = main(
        [
            "play",
            "--controller-a",
            f"sequence:{sequence_a}",
            "--controller-b",
            f"sequence:{sequence_b}",
            "--replay-out",
            str(replay_path),
        ],
        output=lambda _text: None,
    )

    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert replay["result"]["reason"] == "round_limit"
    assert replay["result"]["individual_turns"] == 60


def test_sequence_runtime_failure_returns_its_readable_diagnostic() -> None:
    errors: list[str] = []
    sequence_path = ROOT / "fixtures" / "debug" / "bite_then_retreat.json"

    exit_code = main(
        [
            "play",
            "--seed",
            "1",
            "--controller-a",
            f"sequence:{sequence_path}",
            "--controller-b",
            "aggressive-v1",
        ],
        output=lambda _text: None,
        error_output=errors.append,
    )

    assert exit_code == 1
    assert "Requested action: BITE" in errors[0]
    assert "Legal actions:" in errors[0]
    assert "State:\nRound" in errors[0]


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "dinorl_engine.cli", *arguments],
        cwd=ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def test_debug_commands_work_through_the_real_module_entrypoint() -> None:
    inspected = _run_cli("inspect-map")
    selected_event = _run_cli("replay", str(VALID_REPLAY), "--event", "2")

    assert inspected.returncode == 0
    assert inspected.stdout.startswith("Round 1 — Turn 0 — Actor A\n")
    assert inspected.stderr == ""
    assert selected_event.returncode == 0
    assert selected_event.stdout.endswith("Last action: A BITE B\n")
    assert selected_event.stderr == ""


def test_sequence_cli_replay_is_reproducible_across_processes(tmp_path: Path) -> None:
    sequence_a = tmp_path / "a.json"
    sequence_b = tmp_path / "b.json"
    first_replay = tmp_path / "first.json"
    second_replay = tmp_path / "second.json"
    _write_end_sequence(sequence_a, "stable-a", "END_TURN")
    _write_end_sequence(sequence_b, "stable-b", "END_TURN")
    common_arguments = (
        "play",
        "--seed",
        "17",
        "--controller-a",
        f"sequence:{sequence_a}",
        "--controller-b",
        f"sequence:{sequence_b}",
    )

    first = _run_cli(*common_arguments, "--replay-out", str(first_replay))
    second = _run_cli(*common_arguments, "--replay-out", str(second_replay))

    first_bytes = first_replay.read_bytes()
    second_bytes = second_replay.read_bytes()
    first_output = json.loads(first.stdout)
    assert first.returncode == second.returncode == 0
    assert first_bytes == second_bytes
    assert first_output["replay_sha256"] == hashlib.sha256(first_bytes).hexdigest()
    assert str(sequence_a).encode() not in first_bytes
    assert str(sequence_b).encode() not in first_bytes


def test_sequence_diagnostic_is_readable_without_a_traceback_in_subprocess() -> None:
    sequence_path = ROOT / "fixtures" / "debug" / "bite_then_retreat.json"

    completed = _run_cli(
        "play",
        "--seed",
        "1",
        "--controller-a",
        f"sequence:{sequence_path}",
        "--controller-b",
        "aggressive-v1",
    )

    assert completed.returncode == 1
    assert "Requested action: BITE" in completed.stderr
    assert "Legal actions:" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_core_never_imports_local_debug_adapters() -> None:
    for module_path in (ROOT / "src" / "dinorl_engine" / "core").glob("*.py"):
        source = module_path.read_text(encoding="utf-8")
        assert "dinorl_engine.debug" not in source
        assert "dinorl_engine.controllers.sequence" not in source
