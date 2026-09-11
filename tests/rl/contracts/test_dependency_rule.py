"""The game core must remain independent from the RL subsystem."""

import ast
from pathlib import Path

CORE_DIRECTORY = Path(__file__).parents[3] / "src" / "dinorl_engine" / "core"


def _imports_rl(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                name.name == "dinorl_engine.rl" or name.name.startswith("dinorl_engine.rl.")
                for name in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "dinorl_engine.rl" or module.startswith("dinorl_engine.rl."):
                return True
    return False


def test_core_never_imports_rl() -> None:
    violations = [path for path in CORE_DIRECTORY.rglob("*.py") if _imports_rl(path)]

    assert violations == []
