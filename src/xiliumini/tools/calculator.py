from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

from langchain_core.tools import tool

MAX_EXPRESSION_LENGTH = 256
MAX_ABSOLUTE_EXPONENT = 100
MAX_ABSOLUTE_RESULT = 1e100

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _bounded(value: int | float) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("non-numeric value")
    if not math.isfinite(value) or abs(value) > MAX_ABSOLUTE_RESULT:
        raise ValueError("result is outside the allowed range")
    return value


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("non-numeric value")
        return _bounded(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _bounded(_UNARY_OPERATORS[type(node.op)](_evaluate(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_ABSOLUTE_EXPONENT:
            raise ValueError("exponent is outside the allowed range")
        return _bounded(_BINARY_OPERATORS[type(node.op)](left, right))
    raise ValueError("syntax is not allowed")


def calculate_value(expression: str) -> str:
    """Evaluate a bounded arithmetic expression without executing Python code."""

    if not expression.strip():
        return "Error: expression is empty"
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return "Error: expression is too long"
    try:
        result = _evaluate(ast.parse(expression, mode="eval"))
    except (ArithmeticError, OverflowError, SyntaxError, TypeError, ValueError):
        return "Error: invalid or unsafe arithmetic expression"
    return str(result)


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression."""

    return calculate_value(expression)
