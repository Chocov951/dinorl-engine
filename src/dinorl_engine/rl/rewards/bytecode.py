"""Immutable stack bytecode generated exclusively from validated Reward DSL ASTs."""

from __future__ import annotations

import math
from dataclasses import dataclass

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
)

__all__ = ["BytecodeFunction", "Instruction", "compile_bytecode"]


@dataclass(frozen=True, slots=True)
class Instruction:
    opcode: str
    operand: object | None = None


@dataclass(frozen=True, slots=True)
class BytecodeFunction:
    name: str
    parameters: tuple[str, ...]
    instructions: tuple[Instruction, ...]


def _constant_value(expression: Expression) -> float | bool | None:
    """Fold only literal subtrees, preserving evaluation order and failures."""

    if isinstance(expression, Literal):
        return expression.value
    if isinstance(expression, Unary):
        operand = _constant_value(expression.operand)
        if operand is None:
            return None
        if expression.operator == "!":
            return not bool(operand)
        return -operand if isinstance(operand, float) else None
    if not isinstance(expression, Binary):
        return None
    left = _constant_value(expression.left)
    if left is None:
        return None
    if expression.operator == "&&" and not bool(left):
        return False
    if expression.operator == "||" and bool(left):
        return True
    right = _constant_value(expression.right)
    if right is None:
        return None
    try:
        if expression.operator == "+":
            value: float | bool = float(left) + float(right)
        elif expression.operator == "-":
            value = float(left) - float(right)
        elif expression.operator == "*":
            value = float(left) * float(right)
        elif expression.operator == "/":
            if float(right) == 0.0:
                return None
            value = float(left) / float(right)
        elif expression.operator == "==":
            value = left == right
        elif expression.operator == "!=":
            value = left != right
        elif expression.operator == "<":
            value = float(left) < float(right)
        elif expression.operator == "<=":
            value = float(left) <= float(right)
        elif expression.operator == ">":
            value = float(left) > float(right)
        elif expression.operator == ">=":
            value = float(left) >= float(right)
        elif expression.operator == "&&":
            value = bool(left) and bool(right)
        else:
            value = bool(left) or bool(right)
    except (OverflowError, ValueError):
        return None
    return value if isinstance(value, bool) or math.isfinite(value) else None


def _expression(expression: Expression, code: list[Instruction]) -> None:
    constant = _constant_value(expression)
    if constant is not None:
        code.append(Instruction("CONST", constant))
        return
    if isinstance(expression, Literal):
        code.append(Instruction("CONST", expression.value))
    elif isinstance(expression, Name):
        code.append(Instruction("LOAD", expression.value))
    elif isinstance(expression, FieldAccess):
        _expression(expression.base, code)
        code.append(Instruction("FIELD", expression.field))
    elif isinstance(expression, Unary):
        _expression(expression.operand, code)
        code.append(Instruction("UNARY", expression.operator))
    elif isinstance(expression, Binary):
        _expression(expression.left, code)
        if expression.operator in {"&&", "||"}:
            jump = len(code)
            code.append(
                Instruction(
                    "JUMP_IF_FALSE_KEEP" if expression.operator == "&&" else "JUMP_IF_TRUE_KEEP",
                    None,
                )
            )
            code.append(Instruction("POP"))
            _expression(expression.right, code)
            code[jump] = Instruction(code[jump].opcode, len(code))
        else:
            _expression(expression.right, code)
            code.append(Instruction("BINARY", expression.operator))
    else:
        for argument in expression.arguments:
            _expression(argument, code)
        code.append(Instruction("CALL", (expression.name, len(expression.arguments))))


def _statements(statements: tuple[Statement, ...], code: list[Instruction]) -> None:
    for statement in statements:
        if isinstance(statement, Let):
            _expression(statement.expression, code)
            code.append(Instruction("STORE", statement.name))
        elif isinstance(statement, Return):
            _expression(statement.expression, code)
            code.append(Instruction("RETURN"))
        else:
            constant = _constant_value(statement.condition)
            if isinstance(constant, bool):
                _statements(statement.then_body if constant else statement.else_body or (), code)
                continue
            _expression(statement.condition, code)
            jump_false = len(code)
            code.append(Instruction("JUMP_IF_FALSE", None))
            _statements(statement.then_body, code)
            if statement.else_body is None:
                code[jump_false] = Instruction("JUMP_IF_FALSE", len(code))
            else:
                jump_end = len(code)
                code.append(Instruction("JUMP", None))
                code[jump_false] = Instruction("JUMP_IF_FALSE", len(code))
                _statements(statement.else_body, code)
                code[jump_end] = Instruction("JUMP", len(code))


def _function(function: Function, *, specialize_reference: bool) -> BytecodeFunction:
    if specialize_reference and function.name == "reward":
        return BytecodeFunction(
            function.name,
            tuple(parameter.name for parameter in function.parameters),
            (Instruction("REFERENCE_REWARD"), Instruction("RETURN")),
        )
    instructions: list[Instruction] = []
    _statements(function.body, instructions)
    return BytecodeFunction(
        function.name,
        tuple(parameter.name for parameter in function.parameters),
        tuple(instructions),
    )


def compile_bytecode(
    program: Program, *, specialize_reference: bool = False
) -> tuple[BytecodeFunction, ...]:
    """Compile every validated function without general Python code generation."""

    return tuple(
        _function(function, specialize_reference=specialize_reference)
        for function in program.functions
    )
