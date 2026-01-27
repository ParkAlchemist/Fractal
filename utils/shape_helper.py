from __future__ import annotations
import ast
from typing import Dict, Tuple, List

_ALLOWED_NAMES = {"H", "W", "S", "C", "N", "M", "width", "height", "channels"}
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)


class ShapeExprError(ValueError):
    pass


def _eval_node(node: ast.AST, vars: Dict[str, int]) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name):
        if node.id not in _ALLOWED_NAMES:
            raise ShapeExprError(f"Unknown name '{node.id}' in shape_expr.")
        if node.id not in vars:
            raise ShapeExprError(
                f"Variable '{node.id}' not provided for shape_expr.")
        return int(vars[node.id])
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left = _eval_node(node.left, vars)
        right = _eval_node(node.right, vars)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0: raise ShapeExprError(
                "Division by zero in shape_expr.")
            return left // right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
        val = _eval_node(node.operand, vars)
        return val if isinstance(node.op, ast.UAdd) else -val
    if getattr(ast, "Expr", None) is not None and isinstance(node,
                                                             ast.Expr):  # defensive
        return _eval_node(node.value, vars)
    raise ShapeExprError(
        f"Unsupported expression node: {ast.dump(node, include_attributes=False)}")


def eval_shape_expr(expr: str, vars: Dict[str, int]) -> Tuple[int, ...]:
    """
    Evaluate shape expression like "H*S, W*S", "H, W, 3" with the given variables.
    Returns a tuple of positive ints (zero is clamped to >=1).
    """
    try:
        dims: List[int] = []
        for part in expr.split(","):
            sub = part.strip()
            if not sub:
                raise ShapeExprError("Empty sub-expression in shape_expr.")
            node = ast.parse(sub, mode="eval").body
            val = _eval_node(node, vars)
            if val <= 0:
                raise ShapeExprError(
                    f"Dimension '{sub}' -> {val} (must be > 0).")
            dims.append(int(val))
        return tuple(dims)
    except ShapeExprError:
        raise
    except Exception as e:
        raise ShapeExprError(f"Failed to parse shape_expr '{expr}': {e}")


def discover_shape_vars(expr: str) -> List[str]:
    """Return variable names referenced by expr (best-effort)."""
    names: set[str] = set()
    for part in expr.split(","):
        for tok in ast.walk(ast.parse(part.strip(), mode="eval").body):
            if isinstance(tok, ast.Name): names.add(tok.id)
    return sorted(names)
