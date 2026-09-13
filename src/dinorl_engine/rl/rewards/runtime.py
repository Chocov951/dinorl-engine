"""Pure reference evaluator and bounded compiled representation for Reward DSL."""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Final, TypeGuard

from dinorl_engine.rl.rewards.bytecode import BytecodeFunction, compile_bytecode
from dinorl_engine.rl.rewards.dsl import (
    Binary,
    Expression,
    FieldAccess,
    Function,
    Let,
    Literal,
    Name,
    Program,
    Return,
    Statement,
    Unary,
    canonical_source,
    parse_program,
    validate_program,
)
from dinorl_engine.rl.rewards.reference import REFERENCE_REWARD_SOURCE, reference_reward_public
from dinorl_engine.rl.versions import REWARD_CATALOG_VERSION, REWARD_DSL_VERSION

__all__ = ["CompiledReward", "RewardRuntimeError", "compile_reward"]

_MAX_INSTRUCTIONS: Final = 10_000
_MAX_REWARD: Final = 1_000_000.0
type Value = float | bool | str | tuple[int, int] | dict[str, object]
type RewardEvaluator = Callable[[dict[str, object]], float]


class RewardRuntimeError(RuntimeError):
    """A deterministic runtime failure that aborts the current PPO unit."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True, slots=True)
class CompiledReward:
    """Validated immutable program plus a compact instruction budget contract."""

    source: str
    program: Program
    cache_key: str
    warnings: tuple[str, ...]
    bytecode: tuple[BytecodeFunction, ...]
    reference_fast_path: bool
    vm_evaluator: RewardEvaluator = field(repr=False, compare=False)

    def evaluate_reference(self, transition: dict[str, object]) -> float:
        """Evaluate the AST directly, preserving IEEE operation order."""

        return _evaluate(self.program, transition)

    def evaluate_vm(self, transition: dict[str, object]) -> float:
        """Evaluate the validated instruction stream under its hard budget.

        V1 bytecode stores immutable AST operations; dispatch remains deliberately
        simple so AST and VM share exactly the same numeric semantics.
        """

        return self.vm_evaluator(transition)

    def evaluate_vm_terms(self, transition: dict[str, object]) -> tuple[float, dict[str, float]]:
        """Evaluate once and return numeric named-term contributions.

        Entries are recorded at immutable ``let`` bindings and at returns from
        component functions.  Repeated calls accumulate under the same stable
        label, so a caller can aggregate them across an episode without
        evaluating the DSL source a second time.
        """

        terms: dict[str, float] = {}
        return _evaluate_bytecode(self.bytecode, transition, terms=terms), terms


def compile_reward(source: str) -> CompiledReward:
    """Validate source then obtain its canonical cached compiled representation."""

    # Parse the authored text first so syntax/type diagnostics retain its source
    # positions; the immutable cache itself is keyed only by canonical tokens.
    validate_program(parse_program(source))
    return _compile_canonical(canonical_source(source))


@lru_cache(maxsize=256)
def _compile_canonical(source: str) -> CompiledReward:
    """Compile exactly one normalized source representation."""

    program = parse_program(source)
    warnings = validate_program(program)
    digest = hashlib.sha256()
    digest.update(REWARD_DSL_VERSION.encode())
    digest.update(b"\0")
    digest.update(REWARD_CATALOG_VERSION.encode())
    digest.update(b"\0")
    digest.update(source.encode())
    is_reference = source == canonical_source(REFERENCE_REWARD_SOURCE)
    bytecode = compile_bytecode(program, specialize_reference=is_reference)
    evaluator: RewardEvaluator = (
        reference_reward_public if is_reference else partial(_evaluate_bytecode, bytecode)
    )
    return CompiledReward(
        source,
        program,
        digest.hexdigest(),
        warnings,
        bytecode,
        is_reference,
        evaluator,
    )


def _number(value: Value) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RewardRuntimeError("REWARD_RUNTIME_TYPE", "expected Number")
    number = float(value)
    if not math.isfinite(number):
        raise RewardRuntimeError("REWARD_RUNTIME_NONFINITE", "non-finite number")
    return number


def _arithmetic(operator: str, left: Value, right: Value) -> float:
    """Apply one arithmetic operator and reject a non-finite intermediate."""

    left_number, right_number = _number(left), _number(right)
    if operator == "+":
        result = left_number + right_number
    elif operator == "-":
        result = left_number - right_number
    elif operator == "*":
        result = left_number * right_number
    else:
        if right_number == 0:
            raise RewardRuntimeError("REWARD_RUNTIME_DIVISION", "division by zero")
        result = left_number / right_number
    if not math.isfinite(result):
        raise RewardRuntimeError("REWARD_RUNTIME_NONFINITE", "non-finite arithmetic result")
    return result


def _mapping(value: Value | object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RewardRuntimeError("REWARD_RUNTIME_TYPE", "expected public object")
    return value


def _field(value: Value, field: str) -> Value:
    mapping = _mapping(value)
    result = mapping.get(field)
    if result is None:
        raise RewardRuntimeError("REWARD_RUNTIME_FIELD", f"missing public field {field}")
    if isinstance(result, tuple) and len(result) == 2 and all(type(item) is int for item in result):
        return result
    if isinstance(result, (dict, str, bool, int, float)):
        return result
    raise RewardRuntimeError("REWARD_RUNTIME_FIELD", f"invalid public field {field}")


def _event(transition: dict[str, object], name: str, actor: str) -> float | bool:
    events = _mapping(transition.get("events", {}))
    values = events.get(name, {})
    if not isinstance(values, dict):
        return (
            0.0 if name in {"damage_dealt", "shove_wall_damage", "carcass_points_gained"} else False
        )
    value = values.get(actor, 0.0)
    return (
        value
        if isinstance(value, bool)
        else float(value)
        if isinstance(value, int | float)
        else 0.0
    )


def _builtin(name: str, arguments: tuple[Value, ...], transition: dict[str, object]) -> Value:
    if name == "min":
        return min(_number(arguments[0]), _number(arguments[1]))
    if name == "max":
        return max(_number(arguments[0]), _number(arguments[1]))
    if name == "abs":
        return abs(_number(arguments[0]))
    if name == "clamp":
        return min(max(_number(arguments[0]), _number(arguments[1])), _number(arguments[2]))
    if name in {"manhattan", "path_steps", "movement_cost", "is_adjacent"}:
        first, second = arguments
        if not _position(first) or not _position(second):
            raise RewardRuntimeError("REWARD_RUNTIME_TYPE", "expected Position")
        distance = abs(first[0] - second[0]) + abs(first[1] - second[1])
        if name == "manhattan":
            return float(distance)
        if name == "is_adjacent":
            return distance == 1
        if name == "path_steps":
            return float(_path_steps(first, second, transition))
        return float(_movement_cost(first, second, transition))
    if name == "terminal_score":
        outcome = transition.get("outcome", "ONGOING")
        if not isinstance(outcome, str):
            raise RewardRuntimeError("REWARD_RUNTIME_TYPE", "invalid outcome")
        return (
            _number(arguments[{"WIN": 1, "LOSS": 2, "DRAW": 3}.get(outcome, 0)])
            if outcome != "ONGOING"
            else 0.0
        )
    if name in {"ended_by_ko", "ended_by_carcass_score", "ended_by_round_limit"}:
        return transition.get("end_reason") == name.removeprefix("ended_by_")
    if name in {"damage_dealt", "shove_wall_damage", "carcass_points_gained"}:
        return _event(transition, name, str(arguments[1]))
    if name in {
        "bite_attempted",
        "bite_hit",
        "shove_attempted",
        "shove_moved_target",
        "entered_mud",
        "exited_mud",
        "feed_started",
        "feed_interrupted",
        "feed_completed",
        "rest_started",
        "rest_completed",
    }:
        return bool(_event(transition, name, str(arguments[1])))
    if name == "is_legal":
        legal = _mapping(transition.get("legal_actions", {}))
        return bool(legal.get(str(arguments[1]), False))
    if name == "tile_at":
        tiles = _mapping(transition.get("tiles", {}))
        position = arguments[0]
        if not _position(position):
            raise RewardRuntimeError("REWARD_RUNTIME_TYPE", "expected Position")
        return str(tiles.get(_position_key(position), "EMPTY"))
    raise RewardRuntimeError("REWARD_RUNTIME_BUILTIN", f"unsupported builtin {name}")


def _position(value: Value) -> TypeGuard[tuple[int, int]]:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(type(component) is int for component in value)
    )


def _position_key(position: tuple[int, int]) -> str:
    return f"{position[0]},{position[1]}"


def _geometry(
    transition: dict[str, object],
) -> tuple[int, int, frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    """Read the static public geometry once for a spatial builtin."""

    geometry = _mapping(transition.get("map", {}))
    rows, columns = geometry.get("rows"), geometry.get("columns")
    walls_value, mud_value = geometry.get("walls"), geometry.get("mud")
    if (
        type(rows) is not int
        or type(columns) is not int
        or not isinstance(walls_value, tuple)
        or not isinstance(mud_value, tuple)
    ):
        raise RewardRuntimeError("REWARD_RUNTIME_MAP", "missing public map geometry")
    walls = frozenset(item for item in walls_value if _position(item))
    mud = frozenset(item for item in mud_value if _position(item))
    return rows, columns, walls, mud


def _path_steps(
    origin: tuple[int, int], destination: tuple[int, int], transition: dict[str, object]
) -> int:
    """Return the shortest wall-free route without treating raptors as walls."""

    rows, columns, walls, _mud = _geometry(transition)
    if origin in walls or destination in walls:
        raise RewardRuntimeError("REWARD_RUNTIME_MAP", "position is a wall")
    queue: deque[tuple[tuple[int, int], int]] = deque([(origin, 0)])
    visited = {origin}
    while queue:
        current, steps = queue.popleft()
        if current == destination:
            return steps
        for row_delta, column_delta in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            candidate = current[0] + row_delta, current[1] + column_delta
            if not (0 <= candidate[0] < rows and 0 <= candidate[1] < columns) or candidate in walls:
                continue
            if candidate not in visited:
                visited.add(candidate)
                queue.append((candidate, steps + 1))
    raise RewardRuntimeError("REWARD_RUNTIME_MAP", "positions are disconnected")


def _movement_cost(
    origin: tuple[int, int], destination: tuple[int, int], transition: dict[str, object]
) -> int:
    """Return the least engine movement cost through walls and mud."""

    rows, columns, walls, mud = _geometry(transition)
    if origin in walls or destination in walls:
        raise RewardRuntimeError("REWARD_RUNTIME_MAP", "position is a wall")
    queue: list[tuple[int, tuple[int, int]]] = [(0, origin)]
    best = {origin: 0}
    while queue:
        cost, current = heapq.heappop(queue)
        if current == destination:
            return cost
        if cost != best[current]:
            continue
        for row_delta, column_delta in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            candidate = current[0] + row_delta, current[1] + column_delta
            if not (0 <= candidate[0] < rows and 0 <= candidate[1] < columns) or candidate in walls:
                continue
            candidate_cost = cost + (2 if current in mud else 1)
            if candidate_cost < best.get(candidate, candidate_cost + 1):
                best[candidate] = candidate_cost
                heapq.heappush(queue, (candidate_cost, candidate))
    raise RewardRuntimeError("REWARD_RUNTIME_MAP", "positions are disconnected")


def _evaluate(
    program: Program, transition: dict[str, object], instruction_limit: int = _MAX_INSTRUCTIONS
) -> float:
    functions = {function.name: function for function in program.functions}
    used = 0

    def expression(node: Expression, environment: dict[str, Value]) -> Value:
        nonlocal used
        used += 1
        if used > instruction_limit:
            raise RewardRuntimeError("REWARD_RUNTIME_LIMIT", "VM instruction limit exceeded")
        if isinstance(node, Literal):
            return node.value
        if isinstance(node, Name):
            constants: dict[str, Value] = {
                "SELF": "SELF",
                "OPPONENT": "OPPONENT",
                "WIN": "WIN",
                "LOSS": "LOSS",
                "DRAW": "DRAW",
                "ONGOING": "ONGOING",
                "NO_DIRECTION": "NO_DIRECTION",
                "NORTH": "NORTH",
                "EAST": "EAST",
                "SOUTH": "SOUTH",
                "WEST": "WEST",
                "EMPTY": "EMPTY",
                "WALL": "WALL",
                "MUD": "MUD",
                "CARCASS": "CARCASS",
            }
            if node.value in environment:
                return environment[node.value]
            if node.value in constants:
                return constants[node.value]
            return node.value
        if isinstance(node, FieldAccess):
            return _field(expression(node.base, environment), node.field)
        if isinstance(node, Unary):
            value = expression(node.operand, environment)
            return not bool(value) if node.operator == "!" else -_number(value)
        if isinstance(node, Binary):
            left = expression(node.left, environment)
            if node.operator == "&&":
                return bool(left) and bool(expression(node.right, environment))
            if node.operator == "||":
                return bool(left) or bool(expression(node.right, environment))
            right = expression(node.right, environment)
            if node.operator in {"+", "-", "*", "/"}:
                return _arithmetic(node.operator, left, right)
            if node.operator == "==":
                return left == right
            if node.operator == "!=":
                return left != right
            return {
                "<": _number(left) < _number(right),
                "<=": _number(left) <= _number(right),
                ">": _number(left) > _number(right),
                ">=": _number(left) >= _number(right),
            }[node.operator]
        arguments = tuple(expression(argument, environment) for argument in node.arguments)
        if node.name in functions:
            return call(functions[node.name], arguments)
        return _builtin(node.name, arguments, transition)

    def statements(nodes: tuple[Statement, ...], environment: dict[str, Value]) -> Value | None:
        for node in nodes:
            if isinstance(node, Let):
                environment[node.name] = expression(node.expression, environment)
            elif isinstance(node, Return):
                return expression(node.expression, environment)
            elif bool(expression(node.condition, environment)):
                returned = statements(node.then_body, dict(environment))
                if returned is not None:
                    return returned
            elif node.else_body is not None:
                returned = statements(node.else_body, dict(environment))
                if returned is not None:
                    return returned
        return None

    def call(function: Function, arguments: tuple[Value, ...]) -> Value:
        environment: dict[str, Value] = dict(
            zip((parameter.name for parameter in function.parameters), arguments, strict=True)
        )
        returned = statements(function.body, environment)
        if returned is None:
            raise RewardRuntimeError("REWARD_RUNTIME_RETURN", "missing return")
        return returned

    value = _number(call(functions["reward"], (transition,)))
    if abs(value) > _MAX_REWARD:
        raise RewardRuntimeError("REWARD_RUNTIME_RANGE", "reward exceeds 1e6")
    return value


def _evaluate_bytecode(
    bytecode: tuple[BytecodeFunction, ...],
    transition: dict[str, object],
    *,
    terms: dict[str, float] | None = None,
) -> float:
    """Execute the compact stack representation with a global instruction limit."""

    functions = {function.name: function for function in bytecode}
    used = 0
    constants: dict[str, Value] = {
        "SELF": "SELF",
        "OPPONENT": "OPPONENT",
        "WIN": "WIN",
        "LOSS": "LOSS",
        "DRAW": "DRAW",
        "ONGOING": "ONGOING",
        "NO_DIRECTION": "NO_DIRECTION",
        "NORTH": "NORTH",
        "EAST": "EAST",
        "SOUTH": "SOUTH",
        "WEST": "WEST",
        "EMPTY": "EMPTY",
        "WALL": "WALL",
        "MUD": "MUD",
        "CARCASS": "CARCASS",
    }

    def binary(operator: str, left: Value, right: Value) -> Value:
        if operator in {"+", "-", "*", "/"}:
            return _arithmetic(operator, left, right)
        if operator == "==":
            return left == right
        if operator == "!=":
            return left != right
        if operator == "<":
            return _number(left) < _number(right)
        if operator == "<=":
            return _number(left) <= _number(right)
        if operator == ">":
            return _number(left) > _number(right)
        if operator == ">=":
            return _number(left) >= _number(right)
        if operator == "&&":
            return bool(left) and bool(right)
        return bool(left) or bool(right)

    def invoke(function: BytecodeFunction, arguments: tuple[Value, ...]) -> Value:
        nonlocal used
        environment = dict(zip(function.parameters, arguments, strict=True))
        stack: list[Value] = []
        counter = 0
        while counter < len(function.instructions):
            used += 1
            if used > _MAX_INSTRUCTIONS:
                raise RewardRuntimeError("REWARD_RUNTIME_LIMIT", "VM instruction limit exceeded")
            instruction = function.instructions[counter]
            counter += 1
            if instruction.opcode == "CONST":
                constant = instruction.operand
                if isinstance(constant, bool | float):
                    stack.append(constant)
                else:
                    raise RewardRuntimeError("REWARD_RUNTIME_BYTECODE", "invalid constant")
            elif instruction.opcode == "LOAD":
                name = instruction.operand
                if not isinstance(name, str):
                    raise RewardRuntimeError("REWARD_RUNTIME_BYTECODE", "invalid local name")
                stack.append(environment.get(name, constants.get(name, name)))
            elif instruction.opcode == "STORE":
                value = stack.pop()
                name = str(instruction.operand)
                environment[name] = value
                _record_term(terms, f"{function.name}.{name}", value)
            elif instruction.opcode == "FIELD":
                stack.append(_field(stack.pop(), str(instruction.operand)))
            elif instruction.opcode == "UNARY":
                value = stack.pop()
                stack.append(not bool(value) if instruction.operand == "!" else -_number(value))
            elif instruction.opcode == "BINARY":
                right, left = stack.pop(), stack.pop()
                stack.append(binary(str(instruction.operand), left, right))
            elif instruction.opcode == "POP":
                stack.pop()
            elif instruction.opcode == "CALL":
                call = instruction.operand
                if (
                    not isinstance(call, tuple)
                    or len(call) != 2
                    or not isinstance(call[0], str)
                    or not isinstance(call[1], int)
                ):
                    raise RewardRuntimeError("REWARD_RUNTIME_BYTECODE", "invalid call")
                name, count = call
                values = tuple(stack.pop() for _ in range(count))[::-1]
                stack.append(
                    invoke(functions[name], values)
                    if name in functions
                    else _builtin(name, values, transition)
                )
            elif instruction.opcode == "REFERENCE_REWARD":
                stack.append(reference_reward_public(transition))
            elif instruction.opcode == "JUMP":
                if not isinstance(instruction.operand, int):
                    raise RewardRuntimeError("REWARD_RUNTIME_BYTECODE", "invalid jump")
                counter = instruction.operand
            elif instruction.opcode == "JUMP_IF_FALSE":
                if not bool(stack.pop()):
                    if not isinstance(instruction.operand, int):
                        raise RewardRuntimeError("REWARD_RUNTIME_BYTECODE", "invalid jump")
                    counter = instruction.operand
            elif instruction.opcode == "JUMP_IF_FALSE_KEEP":
                if not bool(stack[-1]):
                    if not isinstance(instruction.operand, int):
                        raise RewardRuntimeError("REWARD_RUNTIME_BYTECODE", "invalid jump")
                    counter = instruction.operand
            elif instruction.opcode == "JUMP_IF_TRUE_KEEP":
                if bool(stack[-1]):
                    if not isinstance(instruction.operand, int):
                        raise RewardRuntimeError("REWARD_RUNTIME_BYTECODE", "invalid jump")
                    counter = instruction.operand
            elif instruction.opcode == "RETURN":
                value = stack.pop()
                if function.name != "reward":
                    _record_term(terms, function.name, value)
                return value
        raise RewardRuntimeError("REWARD_RUNTIME_RETURN", "missing return")

    value = _number(invoke(functions["reward"], (transition,)))
    if abs(value) > _MAX_REWARD:
        raise RewardRuntimeError("REWARD_RUNTIME_RANGE", "reward exceeds 1e6")
    return value


def _record_term(terms: dict[str, float] | None, label: str, value: Value) -> None:
    """Record a finite numeric component while keeping non-numeric terms private."""

    if terms is None or isinstance(value, bool) or not isinstance(value, int | float):
        return
    number = _number(value)
    terms[label] = terms.get(label, 0.0) + number
