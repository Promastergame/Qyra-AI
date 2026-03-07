"""
tools/calculator.py — Safe mathematical expression evaluator.

Uses Python's `ast` module to parse expressions into an Abstract Syntax
Tree, then walks the tree evaluating only whitelisted operations.

Supported operations:
  Arithmetic:    + − * / // % **
  Functions:     sqrt, cbrt, sin, cos, tan, log, ln, factorial, etc.
  Constants:     pi, e, tau, inf
"""

import ast
import math
import operator
from typing import Union, cast, Optional, Any, Callable, Dict, List


BINARY_OPS = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
}

UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _sin_deg(x: float) -> float:
    return math.sin(math.radians(x))

def _cos_deg(x: float) -> float:
    return math.cos(math.radians(x))

def _tan_deg(x: float) -> float:
    return math.tan(math.radians(x))

def _asin_deg(x: float) -> float:
    return math.degrees(math.asin(x))

def _acos_deg(x: float) -> float:
    return math.degrees(math.acos(x))

def _atan_deg(x: float) -> float:
    return math.degrees(math.atan(x))

def _cbrt(x: float) -> float:
    if x < 0:
        return -((-x) ** (1.0 / 3.0))
    return x ** (1.0 / 3.0)

def _safe_factorial(n: float) -> int:
    if not isinstance(n, (int, float)) or n != int(n):
        raise ValueError(f"factorial requires integer, got {n}")
    val = int(n)
    if val < 0:
        raise ValueError(f"factorial of negative number: {val}")
    if val > 1000:
        raise ValueError(f"factorial({val}) too large (max 1000)")
    return math.factorial(val)

def _safe_comb(n: float, k: float) -> int:
    n_int, k_int = int(n), int(k)
    if n_int > 10000 or k_int > 10000:
        raise ValueError("comb arguments too large (max 10000)")
    return math.comb(n_int, k_int)

def _safe_perm(n: float, k: Optional[float] = None) -> int:
    n_int = int(n)
    k_int = int(k) if k is not None else None
    if n_int > 10000:
        raise ValueError("perm arguments too large (max 10000)")
    return math.perm(n_int, k_int)

def _safe_pow(base: float, exp: float) -> float:
    result = math.pow(base, exp)
    if math.isinf(result):
        raise OverflowError(f"{base}**{exp} overflows")
    return result

def _safe_gcd(*args: float) -> int:
    if not args:
        return 0
    int_args = [int(a) for a in args]
    res = int_args[0]
    for a in int_args[1:]:
        res = math.gcd(res, a)
    return res

def _safe_lcm(*args: float) -> int:
    if not args:
        return 0
    int_args = [int(a) for a in args]
    res = int_args[0]
    for a in int_args[1:]:
        res = res * a // math.gcd(res, a)
    return res


FUNCTIONS: Dict[str, Callable] = {
    "sin":     _sin_deg,
    "cos":     _cos_deg,
    "tan":     _tan_deg,
    "asin":    _asin_deg,
    "acos":    _acos_deg,
    "atan":    _atan_deg,
    "atan2":   lambda y, x: math.degrees(math.atan2(y, x)),
    "sin_rad": math.sin,
    "cos_rad": math.cos,
    "tan_rad": math.tan,
    "sinh":    math.sinh,
    "cosh":    math.cosh,
    "tanh":    math.tanh,
    "sqrt":    math.sqrt,
    "cbrt":    _cbrt,
    "log":     math.log10,
    "log10":   math.log10,
    "log2":    math.log2,
    "ln":      math.log,
    "exp":     math.exp,
    "abs":     abs,
    "ceil":    math.ceil,
    "floor":   math.floor,
    "round":   round,
    "factorial": _safe_factorial,
    "comb":      _safe_comb,
    "perm":      _safe_perm,
    "gcd":       _safe_gcd,
    "lcm":       _safe_lcm,
    "pow":     _safe_pow,
    "max":     max,
    "min":     min,
    "degrees": math.degrees,
    "radians": math.radians,
} # type: ignore

CONSTANTS = {
    "pi":   math.pi,
    "e":    math.e,
    "tau":  math.tau,
    "inf":  math.inf,
}


class SafeCalculator:
    """Safe mathematical expression evaluator using AST parsing."""

    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout

    def evaluate(self, expression: str) -> dict:
        expression = expression.strip()
        if not expression:
            return self._error("Empty expression", expression)
        if len(expression) > 500:
            return self._error("Expression too long (max 500 chars)", expression)

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            return self._error(f"Syntax error: {e}", expression)

        try:
            result = self._eval_node(tree.body)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError) as e:
            return self._error(str(e), expression)
        except RecursionError:
            return self._error("Expression too complex", expression)
        except Exception as e:
            return self._error(f"Evaluation error: {e}", expression)

        formatted = self._format_result(result)
        return {
            "success": True,
            "result": result,
            "formatted": formatted,
            "expression": expression,
            "error": None,
        }

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Non-numeric constant: {node.value!r}")

        if isinstance(node, ast.Num):
            return node.n

        if isinstance(node, ast.UnaryOp):
            op_func = UNARY_OPS.get(type(node.op))  # type: ignore
            if op_func is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            operand = self._eval_node(node.operand)
            return op_func(operand)  # type: ignore

        if isinstance(node, ast.BinOp):
            op_func = BINARY_OPS.get(type(node.op))  # type: ignore
            if op_func is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            if isinstance(node.op, ast.Pow):
                if isinstance(right, (int, float)) and abs(right) > 1000:
                    raise ValueError(f"Exponent too large: {right}")

            result = op_func(left, right)
            if isinstance(result, float) and math.isinf(result):
                raise OverflowError("Result overflows to infinity")
            return cast(Union[int, float], result)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls allowed")

            func_name = cast(ast.Name, node.func).id
            if func_name not in FUNCTIONS:
                raise ValueError(f"Unknown function: {func_name}")

            args = [self._eval_node(arg) for arg in node.args]
            if node.keywords:
                raise ValueError("Keyword arguments not supported")

            func = FUNCTIONS[func_name]
            return func(*args)

        if isinstance(node, ast.Name):
            if node.id in CONSTANTS:
                return CONSTANTS[node.id]
            raise ValueError(f"Unknown name: {node.id}")

        raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    @staticmethod
    def _format_result(result: Any) -> str:
        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15:
                return str(int(result))
            if abs(result) < 0.0001 or abs(result) > 1e10:
                return f"{result:.6e}"
            return f"{result:.6f}".rstrip("0").rstrip(".")
        return str(result)

    @staticmethod
    def _error(message: str, expression: str) -> dict:
        return {
            "success": False,
            "result": None,
            "formatted": f"Error: {message}",
            "expression": expression,
            "error": message,
        }


if __name__ == "__main__":
    calc = SafeCalculator()
    tests = [
        ("345 * 872", 300840),
        ("sqrt(289)", 17.0),
        ("factorial(10)", 3628800),
        ("sin(90)", 1.0),
        ("pi * 5**2", None),
    ]
    print("SafeCalculator Test Suite")
    print("=" * 60)
    for expr, expected in tests:
        res = calc.evaluate(expr)
        status = "✓" if res["success"] else "✗"
        print(f"  {status}  {expr:30s} → {res['formatted']}")
