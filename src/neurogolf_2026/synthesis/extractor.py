"""LLM 出力から weight_fn を AST whitelist で安全抽出。

design choices:
- import / eval / exec / __builtins__ アクセスは AST level で reject
- 関数本体は純式のみ許可 (= constant, name, BinOp, Compare, BoolOp, IfExp, return, if-elif-else)
- ast.Call は禁止 (= function call 不可、tuple/int/float リテラルのみ)
- attribute access (= ast.Attribute) も禁止
- closure 不可 (= nonlocal / global 不可)

抽出後は `compile(ast, '<weight_fn>', 'exec')` で `code object` 化、
`exec(code_obj, restricted_globals)` で `weight_fn` を取り出す。
restricted_globals は `{}` (= no builtins access)。
"""
from __future__ import annotations

import ast
import re
from collections.abc import Callable
from typing import Any

CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)

# 許可される AST node (= 関数本体に出現してよいもの)
# NOTE: ast.For / ast.While / ast.Comprehension は **意図的に除外** — loop が無いことで
# DoS (= 無限ループ / huge memory allocation) を構文 level で塞ぐ。 weight_fn は (channel_out,
# channel_in, kernel_coord) の有限 domain (= 10 × 10 × 9) で全 case を if-elif 列挙すれば足りる。
# loop を許可する変更を入れると sandbox の暗黙前提が崩れるので docstring + test 追記 必須。
_ALLOWED_NODES = {
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.If,
    ast.IfExp,
    ast.Compare,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.Load,
    ast.Store,
    ast.Subscript,
    ast.Index,
    ast.Slice,
    ast.Pass,
    ast.Expr,
    # operators
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.And, ast.Or, ast.Not,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Assign,  # 内部一時変数許可 (= 表現力のため)
    ast.AnnAssign,
}

# 禁止 attribute / name (= わずかでも sandbox 破りを防ぐ)
_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "__import__", "open", "input",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "__builtins__", "__class__", "__bases__", "__subclasses__", "__mro__",
    "subprocess", "os", "sys", "io", "pathlib", "shutil",
    "import", "from",
}


class ExtractError(ValueError):
    """LLM output から安全な weight_fn を取り出せなかった場合。"""


def _extract_code_block(raw_output: str) -> str:
    """```python ... ``` の中身を抽出。複数あれば最初の関数定義を含むもの。"""
    matches = CODE_BLOCK_RE.findall(raw_output)
    candidates = matches if matches else [raw_output]
    for c in candidates:
        if "def weight_fn" in c:
            return c.strip()
    raise ExtractError("no `def weight_fn` found in any code block")


def _validate_ast(tree: ast.AST) -> None:
    """全 node が whitelist に含まれることを確認、forbidden name を reject。"""
    for node in ast.walk(tree):
        # node 種別 whitelist
        if type(node) not in _ALLOWED_NODES:
            raise ExtractError(
                f"forbidden AST node: {type(node).__name__} (line {getattr(node, 'lineno', '?')})"
            )
        # Name node の identifier check
        if isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                raise ExtractError(f"forbidden name: {node.id}")
        # Attribute access は禁止 (= _ALLOWED_NODES から除外済だが念押し)
        if isinstance(node, ast.Attribute):
            raise ExtractError(
                f"attribute access not allowed: line {node.lineno}"
            )
        # Call は禁止 (= _ALLOWED_NODES から除外済だが念押し)
        if isinstance(node, ast.Call):
            raise ExtractError(
                f"function call not allowed: line {node.lineno}"
            )


def extract_weight_fn(raw_output: str) -> Callable[[int, int, tuple[int, int]], float]:
    """LLM raw 出力 → 安全に validate された weight_fn callable.

    Raises ExtractError if any safety check fails.
    """
    code = _extract_code_block(raw_output)

    # 文字列レベルでの粗い check (= AST 通過後でも識別子並びの safety)
    for forbidden in _FORBIDDEN_NAMES:
        if re.search(rf"\b{re.escape(forbidden)}\b", code):
            raise ExtractError(f"forbidden token in source: {forbidden}")

    # AST parse
    try:
        tree = ast.parse(code, filename="<weight_fn>")
    except SyntaxError as e:
        raise ExtractError(f"syntax error: {e!r}") from e

    # Module 直下が weight_fn 定義のみ
    body = tree.body
    if len(body) != 1 or not isinstance(body[0], ast.FunctionDef):
        raise ExtractError("module body must be a single FunctionDef")
    func_def = body[0]
    if func_def.name != "weight_fn":
        raise ExtractError(f"function name must be `weight_fn`, got `{func_def.name}`")

    # 引数 signature 確認
    args = func_def.args.args
    if len(args) != 3:
        raise ExtractError(f"weight_fn must take 3 args, got {len(args)}")
    arg_names = [a.arg for a in args]
    if arg_names != ["channel_out", "channel_in", "kernel_coord"]:
        raise ExtractError(f"arg names must be (channel_out, channel_in, kernel_coord), got {arg_names}")

    # AST whitelist check
    _validate_ast(tree)

    # Compile + exec in restricted scope
    code_obj = compile(tree, filename="<weight_fn>", mode="exec")
    restricted: dict[str, Any] = {"__builtins__": {}}
    try:
        exec(code_obj, restricted)  # noqa: S102 — sandbox は AST whitelist + restricted globals で担保
    except Exception as e:
        raise ExtractError(f"exec failed: {e!r}") from e

    fn = restricted.get("weight_fn")
    if not callable(fn):
        raise ExtractError("weight_fn not callable after exec")
    return fn
