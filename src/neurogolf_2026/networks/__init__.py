"""Per-task ONNX builder registry.

baseline 段階では task276 (= Color Swapping, Complexity 1) のみ独自 builder。
残り 399 task は fallback (= zero conv 1x1) で submission.zip 構造を満たす。

後続 plan で task002, task003, ... の builder を追加していく registry pattern。
"""
from __future__ import annotations

from collections.abc import Callable

import onnx

from . import fallback, task276

BuilderFn = Callable[[], onnx.ModelProto]

REGISTRY: dict[str, BuilderFn] = {
    "task276": task276.build,
}


def get_builder(task_id: str) -> BuilderFn:
    """task_id (= 'task001' .. 'task400') -> builder function.

    未登録 task は fallback.build (= zero conv) を返す。
    """
    return REGISTRY.get(task_id, fallback.build)
