"""CLI phase gates for RL-S5c-A3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dinorl_engine.rl.s5c.a3 import require_prior_checkpoint
from dinorl_engine.rl.s5c.a3_config import A3Config


def _run_directory(run_id: str) -> Path:
    return Path("artifacts/rl/s5c") / run_id


def run_a3_command(options: argparse.Namespace) -> dict[str, object]:
    command = str(options.s5c_command)
    if command in {"a3-status", "a3-report"}:
        run = _run_directory(str(options.run_id))
        control = run / "policy-control" / "result.json"
        pilot = run / "pilot" / "selection.json"
        confirm = run / "confirm" / "result.json"
        state = "READY_FOR_CONTROL"
        recovery_units = run / "policy-control" / "recovery" / "units"
        completed_units = (
            len([path for path in recovery_units.iterdir() if path.is_dir()])
            if recovery_units.is_dir()
            else 0
        )
        if completed_units:
            state = "CONTROL_RUNNING"
        for path, label in (
            (control, "CONTROL_COMPLETE"),
            (pilot, "PILOT_COMPLETE"),
            (confirm, "CONFIRM_COMPLETE"),
        ):
            if path.is_file():
                state = label
        result: dict[str, object] = {
            "format": "s5c-a3-status-v1",
            "run_id": options.run_id,
            "phase_status": state,
            "completed_units": completed_units,
        }
        if command == "a3-report":
            report = run / "report" / "rl-s5c-a3-checkpoint-report.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
            markdown = run / "report" / "rl-s5c-a3-checkpoint-report.md"
            pilot_allowed = False
            if control.is_file():
                control_value = json.loads(control.read_text(encoding="utf-8"))
                pilot_allowed = (
                    isinstance(control_value, dict)
                    and control_value.get("decision") == "PASSED_POLICY_CONTROL"
                )
            markdown.write_text(
                "# RL-S5c-A3 — rapport de checkpoint\n\n"
                f"- État : `{state}`\n"
                f"- Unités CONTROL validées : {completed_units}/147\n"
                f"- PILOT autorisé : {'oui' if pilot_allowed else 'non'}\n",
                encoding="utf-8",
            )
            result["report"] = str(report)
            result["markdown"] = str(markdown)
        return {"command": f"s5c {command}", **result, "status": "completed"}

    config = A3Config.load(options.config)
    if options.run_id is not None and options.run_id != config.run_id:
        raise ValueError("A3 run-id must match its frozen configuration")
    if command == "a3-preflight":
        from dinorl_engine.rl.s5c.a3_preflight import run_a3_preflight

        result = run_a3_preflight(config, repository=Path.cwd(), allow_dirty=options.allow_dirty)
        warning = result["provenance"].get("warning")
        if warning:
            print(str(warning), file=sys.stderr)
        return {
            "command": "s5c a3-preflight",
            "manifest": str(config.output_directory / "manifest.json"),
            "status": "completed",
        }
    if command == "a3-policy-control":
        from dinorl_engine.rl.s5c.a3_control import run_policy_control

        progress = None if options.quiet else lambda message: print(message, file=sys.stderr)
        result = run_policy_control(config, resume=options.resume, progress=progress)
        return {"command": "s5c a3-policy-control", **result, "status": "completed"}
    if command == "a3-pilot":
        require_prior_checkpoint(config.output_directory, phase="pilot")
        from dinorl_engine.rl.s5c.a3_pilot import run_pilot

        progress = None if options.quiet else lambda message: print(message, file=sys.stderr)
        return {
            "command": "s5c a3-pilot",
            **run_pilot(config, resume=options.resume, progress=progress),
            "status": "completed",
        }
    if command == "a3-pilot-r2":
        require_prior_checkpoint(config.output_directory, phase="pilot")
        from dinorl_engine.rl.s5c.a3_pilot_r2 import run_pilot_r2

        progress = None if options.quiet else lambda message: print(message, file=sys.stderr)
        return {
            "command": "s5c a3-pilot-r2",
            **run_pilot_r2(config, resume=options.resume, progress=progress),
            "status": "completed",
        }
    require_prior_checkpoint(config.output_directory, phase="confirm")
    from dinorl_engine.rl.s5c.a3_pilot import run_confirm

    return {
        "command": "s5c a3-confirm",
        **run_confirm(config, resume=options.resume),
        "status": "completed",
    }
