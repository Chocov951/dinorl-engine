"""Lexer and parser for the bounded, non-Python Reward DSL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

__all__ = [
    "Binary",
    "Call",
    "FieldAccess",
    "Function",
    "If",
    "Let",
    "Literal",
    "Name",
    "Program",
    "Return",
    "RewardParseError",
    "RewardTypeError",
    "Unary",
    "canonical_source",
    "parse_program",
    "validate_program",
]

_MAX_SOURCE_BYTES: Final = 32 * 1024
_MAX_FUNCTIONS: Final = 64
_MAX_AST_NODES: Final = 2048
_MAX_SYNTAX_DEPTH: Final = 64
_TYPES: Final = frozenset(
    {
        "Number",
        "Bool",
        "Position",
        "Actor",
        "ActionKind",
        "Direction",
        "TileKind",
        "Outcome",
        "Transition",
    }
)
_MULTI_SYMBOLS: Final = ("->", "&&", "||", "<=", ">=", "==", "!=")
_SINGLE_SYMBOLS: Final = frozenset("(){}:;,.-+*/!<>=")
_PRECEDENCE: Final = {
    "||": 1,
    "&&": 2,
    "==": 3,
    "!=": 3,
    "<": 4,
    "<=": 4,
    ">": 4,
    ">=": 4,
    "+": 5,
    "-": 5,
    "*": 6,
    "/": 6,
}
_CONSTANT_TYPES: Final = {
    "SELF": "Actor",
    "OPPONENT": "Actor",
    "NO_DIRECTION": "Direction",
    "ONGOING": "Outcome",
    "WIN": "Outcome",
    "LOSS": "Outcome",
    "DRAW": "Outcome",
    **{name: "Direction" for name in ("NORTH", "EAST", "SOUTH", "WEST")},
    **{name: "TileKind" for name in ("EMPTY", "WALL", "MUD", "CARCASS")},
    **{
        name: "ActionKind"
        for name in (
            "MOVE_NORTH",
            "MOVE_EAST",
            "MOVE_SOUTH",
            "MOVE_WEST",
            "BITE",
            "SHOVE",
            "FEED",
            "REST",
            "END_TURN",
        )
    },
}
_BUILTINS: Final[dict[str, tuple[tuple[str, ...], str]]] = {
    "min": (("Number", "Number"), "Number"),
    "max": (("Number", "Number"), "Number"),
    "abs": (("Number",), "Number"),
    "clamp": (("Number", "Number", "Number"), "Number"),
    "manhattan": (("Position", "Position"), "Number"),
    "path_steps": (("Position", "Position"), "Number"),
    "movement_cost": (("Position", "Position"), "Number"),
    "is_adjacent": (("Position", "Position"), "Bool"),
    "tile_at": (("Position",), "TileKind"),
    "is_legal": (("Transition", "ActionKind"), "Bool"),
    "damage_dealt": (("Transition", "Actor"), "Number"),
    "bite_attempted": (("Transition", "Actor"), "Bool"),
    "bite_hit": (("Transition", "Actor"), "Bool"),
    "shove_attempted": (("Transition", "Actor"), "Bool"),
    "shove_moved_target": (("Transition", "Actor"), "Bool"),
    "shove_wall_damage": (("Transition", "Actor"), "Number"),
    "entered_mud": (("Transition", "Actor"), "Bool"),
    "exited_mud": (("Transition", "Actor"), "Bool"),
    "feed_started": (("Transition", "Actor"), "Bool"),
    "feed_interrupted": (("Transition", "Actor"), "Bool"),
    "feed_completed": (("Transition", "Actor"), "Bool"),
    "rest_started": (("Transition", "Actor"), "Bool"),
    "rest_completed": (("Transition", "Actor"), "Bool"),
    "carcass_points_gained": (("Transition", "Actor"), "Number"),
    "ended_by_ko": (("Transition",), "Bool"),
    "ended_by_carcass_score": (("Transition",), "Bool"),
    "ended_by_round_limit": (("Transition",), "Bool"),
    "terminal_score": (("Transition", "Number", "Number", "Number"), "Number"),
}
_FIELDS: Final[dict[tuple[str, ...], str]] = {
    ("action", "actor"): "Actor",
    ("action", "kind"): "ActionKind",
    ("action", "direction"): "Direction",
    ("action", "movement_cost"): "Number",
    ("action", "endurance_cost"): "Number",
    ("action", "success"): "Bool",
    ("outcome",): "Outcome",
    **{
        (moment, "self", field): kind
        for moment in ("before", "after")
        for field, kind in {
            "position": "Position",
            "hp": "Number",
            "endurance": "Number",
            "movement_points": "Number",
            "carcass_score": "Number",
            "main_action_available": "Bool",
            "voluntary_move_done": "Bool",
        }.items()
    },
    **{
        (moment, "opponent", field): kind
        for moment in ("before", "after")
        for field, kind in {
            "position": "Position",
            "hp": "Number",
            "endurance": "Number",
            "carcass_score": "Number",
            "consumption_pending": "Bool",
            "rest_pending": "Bool",
        }.items()
    },
    **{
        (moment, field): kind
        for moment in ("before", "after")
        for field, kind in {
            "round": "Number",
            "turn": "Number",
            "active_actor": "Actor",
            "initiative": "Actor",
        }.items()
    },
    **{
        (moment, "carcasses", carcass, field): kind
        for moment in ("before", "after")
        for carcass in ("carcass_left_a", "carcass_center", "carcass_left_b")
        for field, kind in {
            "position": "Position",
            "available": "Bool",
            "reactivation_delay": "Number",
        }.items()
    },
}


@dataclass(frozen=True, slots=True)
class Span:
    """A stable one-based source position for a diagnostic."""

    line: int
    column: int


class RewardParseError(ValueError):
    """A lexical or syntactic diagnostic without executing user source."""

    def __init__(self, code: str, span: Span, message: str) -> None:
        super().__init__(f"{code} at {span.line}:{span.column}: {message}")
        self.code = code
        self.span = span


class RewardTypeError(ValueError):
    """A static typing or call-graph diagnostic for Reward DSL source."""

    def __init__(self, code: str, span: Span, message: str) -> None:
        super().__init__(f"{code} at {span.line}:{span.column}: {message}")
        self.code = code
        self.span = span


@dataclass(frozen=True, slots=True)
class Token:
    kind: str
    value: str
    span: Span


@dataclass(frozen=True, slots=True)
class Literal:
    value: float
    span: Span


@dataclass(frozen=True, slots=True)
class Name:
    value: str
    span: Span


@dataclass(frozen=True, slots=True)
class FieldAccess:
    base: Name | FieldAccess
    field: str
    span: Span


@dataclass(frozen=True, slots=True)
class Call:
    name: str
    arguments: tuple[Expression, ...]
    span: Span


@dataclass(frozen=True, slots=True)
class Unary:
    operator: str
    operand: Expression
    span: Span


@dataclass(frozen=True, slots=True)
class Binary:
    operator: str
    left: Expression
    right: Expression
    span: Span


type Expression = Literal | Name | FieldAccess | Call | Unary | Binary


@dataclass(frozen=True, slots=True)
class Let:
    name: str
    type_name: str
    expression: Expression
    span: Span


@dataclass(frozen=True, slots=True)
class If:
    condition: Expression
    then_body: tuple[Statement, ...]
    else_body: tuple[Statement, ...] | None
    span: Span


@dataclass(frozen=True, slots=True)
class Return:
    expression: Expression
    span: Span


type Statement = Let | If | Return


@dataclass(frozen=True, slots=True)
class Parameter:
    name: str
    type_name: str
    span: Span


@dataclass(frozen=True, slots=True)
class Function:
    name: str
    parameters: tuple[Parameter, ...]
    return_type: str
    body: tuple[Statement, ...]
    span: Span


@dataclass(frozen=True, slots=True)
class Program:
    functions: tuple[Function, ...]


def _tokens(source: str) -> tuple[Token, ...]:
    if len(source.encode()) > _MAX_SOURCE_BYTES:
        raise RewardParseError("REWARD_PARSE_LIMIT", Span(1, 1), "source exceeds 32 KiB")
    tokens: list[Token] = []
    index, line, column = 0, 1, 1
    while index < len(source):
        character = source[index]
        if character in " \t\r":
            index += 1
            column += 1
            continue
        if character == "\n":
            index += 1
            line += 1
            column = 1
            continue
        if source.startswith("//", index):
            while index < len(source) and source[index] != "\n":
                index += 1
                column += 1
            continue
        span = Span(line, column)
        if character.isascii() and (character.isalpha() or character == "_"):
            start = index
            while (
                index < len(source)
                and source[index].isascii()
                and (source[index].isalnum() or source[index] == "_")
            ):
                index += 1
                column += 1
            tokens.append(Token("identifier", source[start:index], span))
            continue
        if character.isascii() and character.isdigit():
            start = index
            while index < len(source) and source[index].isdigit():
                index += 1
                column += 1
            if index < len(source) and source[index] == ".":
                index += 1
                column += 1
                decimal_start = index
                while index < len(source) and source[index].isdigit():
                    index += 1
                    column += 1
                if index == decimal_start:
                    raise RewardParseError("REWARD_PARSE_LEX", span, "invalid decimal literal")
            value = source[start:index]
            if not math.isfinite(float(value)):
                raise RewardParseError("REWARD_PARSE_LEX", span, "number must be finite")
            tokens.append(Token("number", value, span))
            continue
        symbol = next((item for item in _MULTI_SYMBOLS if source.startswith(item, index)), None)
        if symbol is None and character in _SINGLE_SYMBOLS:
            symbol = character
        if symbol is not None:
            tokens.append(Token("symbol", symbol, span))
            index += len(symbol)
            column += len(symbol)
            continue
        raise RewardParseError("REWARD_PARSE_LEX", span, f"unexpected character {character!r}")
    tokens.append(Token("eof", "", Span(line, column)))
    return tuple(tokens)


class _Parser:
    def __init__(self, source: str) -> None:
        self.tokens = _tokens(source)
        self.index = 0
        self.nodes = 0
        self.depth = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def _node(self) -> None:
        self.nodes += 1
        if self.nodes > _MAX_AST_NODES:
            raise RewardParseError("REWARD_PARSE_LIMIT", self.current.span, "too many AST nodes")

    def _advance(self) -> Token:
        token = self.current
        self.index += 1
        return token

    def _expect(self, value: str) -> Token:
        if self.current.value != value:
            raise RewardParseError("REWARD_PARSE_SYNTAX", self.current.span, f"expected {value!r}")
        return self._advance()

    def _identifier(self) -> Token:
        if self.current.kind != "identifier":
            raise RewardParseError("REWARD_PARSE_SYNTAX", self.current.span, "expected identifier")
        return self._advance()

    def _type(self) -> str:
        token = self._identifier()
        if token.value not in _TYPES:
            raise RewardParseError("REWARD_PARSE_SYNTAX", token.span, "unknown type")
        return token.value

    def parse(self) -> Program:
        functions: list[Function] = []
        while self.current.kind != "eof":
            functions.append(self._function())
            if len(functions) > _MAX_FUNCTIONS:
                raise RewardParseError(
                    "REWARD_PARSE_LIMIT", self.current.span, "too many functions"
                )
        if not functions:
            raise RewardParseError("REWARD_PARSE_SYNTAX", self.current.span, "expected function")
        return Program(tuple(functions))

    def _function(self) -> Function:
        span = self._expect("fn").span
        name = self._identifier().value
        self._expect("(")
        parameters: list[Parameter] = []
        if self.current.value != ")":
            while True:
                parameter = self._identifier()
                self._expect(":")
                parameters.append(Parameter(parameter.value, self._type(), parameter.span))
                self._node()
                if self.current.value != ",":
                    break
                self._advance()
        self._expect(")")
        self._expect("->")
        return_type = self._type()
        body = self._block()
        self._node()
        return Function(name, tuple(parameters), return_type, body, span)

    def _block(self) -> tuple[Statement, ...]:
        self._expect("{")
        statements: list[Statement] = []
        while self.current.value != "}":
            if self.current.kind == "eof":
                raise RewardParseError("REWARD_PARSE_SYNTAX", self.current.span, "expected '}'")
            statements.append(self._statement())
        self._advance()
        return tuple(statements)

    def _statement(self) -> Statement:
        if self.current.value == "let":
            span = self._advance().span
            name = self._identifier().value
            self._expect(":")
            type_name = self._type()
            self._expect("=")
            expression = self._expression()
            self._expect(";")
            self._node()
            return Let(name, type_name, expression, span)
        if self.current.value == "if":
            span = self._advance().span
            condition = self._expression()
            then_body = self._block()
            else_body = self._block() if self.current.value == "else" and self._advance() else None
            self._node()
            return If(condition, then_body, else_body, span)
        if self.current.value == "return":
            span = self._advance().span
            expression = self._expression()
            self._expect(";")
            self._node()
            return Return(expression, span)
        raise RewardParseError("REWARD_PARSE_SYNTAX", self.current.span, "expected statement")

    def _expression(self, minimum_precedence: int = 1) -> Expression:
        self.depth += 1
        if self.depth > _MAX_SYNTAX_DEPTH:
            raise RewardParseError(
                "REWARD_PARSE_LIMIT", self.current.span, "syntax nesting exceeds 64"
            )
        try:
            left = self._unary()
            while (precedence := _PRECEDENCE.get(self.current.value, 0)) >= minimum_precedence:
                operator = self._advance()
                right = self._expression(precedence + 1)
                self._node()
                left = Binary(operator.value, left, right, operator.span)
            return left
        finally:
            self.depth -= 1

    def _unary(self) -> Expression:
        if self.current.value in {"!", "-"}:
            operator = self._advance()
            operand = self._unary()
            self._node()
            return Unary(operator.value, operand, operator.span)
        return self._primary()

    def _primary(self) -> Expression:
        token = self.current
        if token.kind == "number":
            self._advance()
            self._node()
            return Literal(float(token.value), token.span)
        if token.value == "(":
            self._advance()
            grouped_expression = self._expression()
            self._expect(")")
            return grouped_expression
        name = self._identifier()
        if self.current.value == "(":
            self._advance()
            arguments: list[Expression] = []
            if self.current.value != ")":
                while True:
                    arguments.append(self._expression())
                    if self.current.value != ",":
                        break
                    self._advance()
            self._expect(")")
            self._node()
            return Call(name.value, tuple(arguments), name.span)
        field_access: Name | FieldAccess = Name(name.value, name.span)
        self._node()
        while self.current.value == ".":
            self._advance()
            field = self._identifier()
            self._node()
            field_access = FieldAccess(field_access, field.value, field.span)
        return field_access


def parse_program(source: str) -> Program:
    """Parse source under the V1 lexical and structural resource limits."""

    if not isinstance(source, str):
        raise TypeError("reward source must be a string")
    return _Parser(source).parse()


def canonical_source(source: str) -> str:
    """Return the version-independent lexical form used for cache identity.

    Comments and insignificant whitespace must not create distinct compiled
    rewards.  Tokens are separated deliberately so two adjacent identifiers
    cannot be merged when this source is parsed again.
    """

    return " ".join(token.value for token in _tokens(source) if token.kind != "eof")


def _calls_in_expression(expression: Expression) -> tuple[Call, ...]:
    if isinstance(expression, Call):
        nested = tuple(
            call for argument in expression.arguments for call in _calls_in_expression(argument)
        )
        return (expression, *nested)
    if isinstance(expression, Unary):
        return _calls_in_expression(expression.operand)
    if isinstance(expression, Binary):
        return (*_calls_in_expression(expression.left), *_calls_in_expression(expression.right))
    return ()


def _calls_in_statements(statements: tuple[Statement, ...]) -> tuple[Call, ...]:
    calls: list[Call] = []
    for statement in statements:
        if isinstance(statement, Let | Return):
            calls.extend(_calls_in_expression(statement.expression))
        elif isinstance(statement, If):
            calls.extend(_calls_in_expression(statement.condition))
            calls.extend(_calls_in_statements(statement.then_body))
            if statement.else_body is not None:
                calls.extend(_calls_in_statements(statement.else_body))
    return tuple(calls)


def _field_path(expression: FieldAccess, variables: dict[str, str]) -> tuple[str, ...] | None:
    fields = [expression.field]
    base = expression.base
    while isinstance(base, FieldAccess):
        fields.append(base.field)
        base = base.base
    if variables.get(base.value) != "Transition":
        return None
    return tuple(reversed(fields))


def _type_expression(
    expression: Expression,
    variables: dict[str, str],
    functions: dict[str, Function],
) -> str:
    if isinstance(expression, Literal):
        return "Number"
    if isinstance(expression, Name):
        value_type = variables.get(expression.value) or _CONSTANT_TYPES.get(expression.value)
        if value_type is None:
            raise RewardTypeError("REWARD_TYPE_NAME", expression.span, "unknown name")
        return value_type
    if isinstance(expression, FieldAccess):
        path = _field_path(expression, variables)
        value_type = _FIELDS.get(path) if path is not None else None
        if value_type is None:
            raise RewardTypeError("REWARD_TYPE_FIELD", expression.span, "field is not public")
        return value_type
    if isinstance(expression, Unary):
        operand_type = _type_expression(expression.operand, variables, functions)
        unary_expected = "Bool" if expression.operator == "!" else "Number"
        if operand_type != unary_expected:
            raise RewardTypeError("REWARD_TYPE_OPERATOR", expression.span, "invalid unary operand")
        return unary_expected
    if isinstance(expression, Binary):
        left_type = _type_expression(expression.left, variables, functions)
        right_type = _type_expression(expression.right, variables, functions)
        if expression.operator in {"+", "-", "*", "/"}:
            if left_type != "Number" or right_type != "Number":
                raise RewardTypeError(
                    "REWARD_TYPE_OPERATOR", expression.span, "arithmetic needs Number"
                )
            return "Number"
        if expression.operator in {"<", "<=", ">", ">="}:
            if left_type != "Number" or right_type != "Number":
                raise RewardTypeError(
                    "REWARD_TYPE_OPERATOR", expression.span, "comparison needs Number"
                )
            return "Bool"
        if expression.operator in {"==", "!="}:
            if left_type != right_type:
                raise RewardTypeError(
                    "REWARD_TYPE_OPERATOR", expression.span, "equality needs matching types"
                )
            return "Bool"
        if left_type != "Bool" or right_type != "Bool":
            raise RewardTypeError(
                "REWARD_TYPE_OPERATOR", expression.span, "boolean operator needs Bool"
            )
        return "Bool"
    signature = _BUILTINS.get(expression.name)
    expected: tuple[str, ...]
    returned: str
    if signature is None:
        function = functions.get(expression.name)
        if function is None:
            raise RewardTypeError("REWARD_TYPE_CALL", expression.span, "unknown function")
        expected = tuple(parameter.type_name for parameter in function.parameters)
        returned = function.return_type
    else:
        expected, returned = signature
    actual = tuple(
        _type_expression(argument, variables, functions) for argument in expression.arguments
    )
    if actual != expected:
        raise RewardTypeError(
            "REWARD_TYPE_CALL", expression.span, "function arguments do not match"
        )
    return returned


def _block_returns(
    statements: tuple[Statement, ...],
    variables: dict[str, str],
    functions: dict[str, Function],
    return_type: str,
) -> bool:
    local_variables = dict(variables)
    for statement in statements:
        if isinstance(statement, Let):
            if (
                statement.name in local_variables
                or _type_expression(statement.expression, local_variables, functions)
                != statement.type_name
            ):
                raise RewardTypeError("REWARD_TYPE_LET", statement.span, "invalid immutable local")
            local_variables[statement.name] = statement.type_name
        elif isinstance(statement, Return):
            if _type_expression(statement.expression, local_variables, functions) != return_type:
                raise RewardTypeError(
                    "REWARD_TYPE_RETURN", statement.span, "return type does not match"
                )
            return True
        else:
            if _type_expression(statement.condition, local_variables, functions) != "Bool":
                raise RewardTypeError("REWARD_TYPE_IF", statement.span, "if condition must be Bool")
            then_returns = _block_returns(
                statement.then_body, local_variables, functions, return_type
            )
            else_returns = (
                _block_returns(statement.else_body, local_variables, functions, return_type)
                if statement.else_body is not None
                else False
            )
            if then_returns and else_returns:
                return True
    return False


def validate_program(program: Program) -> tuple[str, ...]:
    """Check mandatory entrypoint, unique functions and an acyclic call graph.

    The returned warnings are deliberately non-blocking V1 diagnostics. Full
    expression typing is performed by the compiler that consumes this AST.
    """

    functions: dict[str, Function] = {}
    for function in program.functions:
        if function.name in functions:
            raise RewardTypeError("REWARD_TYPE_FUNCTION", function.span, "duplicate function name")
        functions[function.name] = function
    reward = functions.get("reward")
    if (
        reward is None
        or reward.return_type != "Number"
        or len(reward.parameters) != 1
        or reward.parameters[0].type_name != "Transition"
    ):
        span = reward.span if reward is not None else Span(1, 1)
        raise RewardTypeError(
            "REWARD_TYPE_ENTRYPOINT", span, "expected fn reward(t: Transition) -> Number"
        )
    graph = {
        function.name: tuple(
            call.name for call in _calls_in_statements(function.body) if call.name in functions
        )
        for function in program.functions
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(function_name: str) -> None:
        if function_name in visiting:
            function = functions[function_name]
            raise RewardTypeError("REWARD_TYPE_CYCLE", function.span, "recursive call graph")
        if function_name in visited:
            return
        visiting.add(function_name)
        for callee in graph[function_name]:
            visit(callee)
        visiting.remove(function_name)
        visited.add(function_name)

    visit("reward")
    for function in program.functions:
        visit(function.name)
    call_depth: dict[str, int] = {}

    def depth(function_name: str) -> int:
        if function_name in call_depth:
            return call_depth[function_name]
        value = 1 + max((depth(callee) for callee in graph[function_name]), default=0)
        call_depth[function_name] = value
        return value

    if depth("reward") > 32:
        raise RewardTypeError("REWARD_TYPE_LIMIT", reward.span, "call nesting exceeds 32")
    for function in program.functions:
        variables = {parameter.name: parameter.type_name for parameter in function.parameters}
        if len(variables) != len(function.parameters):
            raise RewardTypeError("REWARD_TYPE_FUNCTION", function.span, "duplicate parameter name")
        if not _block_returns(function.body, variables, functions, function.return_type):
            raise RewardTypeError(
                "REWARD_TYPE_RETURN", function.span, "not all paths return a value"
            )
    reachable: set[str] = set()

    def mark(function_name: str) -> None:
        if function_name in reachable:
            return
        reachable.add(function_name)
        for callee in graph[function_name]:
            mark(callee)

    mark("reward")
    return tuple(
        f"unreachable function: {function.name}"
        for function in program.functions
        if function.name not in reachable
    )
