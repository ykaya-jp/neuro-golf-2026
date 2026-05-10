"""Runner: 抽出 weight_fn → ONNX → score_network で functional correct + scorer_ok 判定."""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import onnx

from neurogolf_2026.build_submission import _validate_onnx
from neurogolf_2026.networks._helpers import single_layer_conv2d_network
from neurogolf_2026.validate import validate_task
from neurogolf_2026.networks import REGISTRY

# 並列 dispatch 時の REGISTRY mutation race を防ぐ (= 1 process 内 1 lock)
# NOTE: multi-process では別途 IPC 必要、 本実装は thread-safe のみ
_REGISTRY_LOCK = threading.Lock()


@contextmanager
def _temp_registry(task_id: str, model: onnx.ModelProto) -> Iterator[None]:
    """REGISTRY に task_id → builder を一時 inject、 finally で原状復帰.

    並列 unsafe (= 同一 process 内 multi-thread 想定の Lock 付き) — multi-process は
    別途 IPC 必要 (= 各 worker が独立 REGISTRY を持つ設計に refactor)。
    """
    with _REGISTRY_LOCK:
        original = REGISTRY.get(task_id)
        REGISTRY[task_id] = lambda: model
        try:
            yield
        finally:
            if original is None:
                REGISTRY.pop(task_id, None)
            else:
                REGISTRY[task_id] = original


@dataclass
class RunResult:
    task_id: str
    onnx_size: int | None = None
    constraint_violations: list[str] | None = None
    functional_correct: bool = False
    scorer_ok: bool = False
    cost: int | None = None
    score: float | None = None
    arc_agi_pass: int = 0
    arc_agi_fail: int = 0
    arc_gen_pass: int = 0
    arc_gen_fail: int = 0
    error: str | None = None
    # registry 採用判定 (= functional_correct + scorer_ok + 制約準拠)
    accepted: bool = False


def run_weight_fn(
    task_id: str,
    weight_fn: Callable[[int, int, tuple[int, int]], float],
    *,
    kernel_size: int = 1,
) -> RunResult:
    """weight_fn → ONNX → validate のフルパイプ。registry には書き込まない。

    AC-3 の banned op check は build_submission._validate_onnx を再利用。
    """
    result = RunResult(task_id=task_id)

    # ONNX 化
    try:
        model = single_layer_conv2d_network(weight_fn, kernel_size=kernel_size)
    except Exception as e:
        result.error = f"single_layer_conv2d_network failed: {e!r}"
        return result

    raw = model.SerializeToString()
    result.onnx_size = len(raw)

    # 制約 check (= banned op / size / kernel_time / multi-IO)
    violations = _validate_onnx(task_id, raw)
    if violations:
        result.constraint_violations = violations
        result.error = f"constraint violations: {violations}"
        return result
    result.constraint_violations = []

    # functional correct + scorer_ok を score_network で判定
    # context manager で REGISTRY 一時 inject (= thread-safe、 multi-process は別 IPC 必要)
    with _temp_registry(task_id, model):
        v = validate_task(task_id, strict=False, verbose=False)

    result.arc_agi_pass = v["arc_agi"]["right"]
    result.arc_agi_fail = v["arc_agi"]["wrong"]
    result.arc_gen_pass = v["arc_gen"]["right"]
    result.arc_gen_fail = v["arc_gen"]["wrong"]
    result.functional_correct = v["functional_correct"]
    result.scorer_ok = v["scorer_ok"]
    result.cost = v.get("cost")
    result.score = v.get("score")
    if v["errors"]:
        result.error = "; ".join(str(e) for e in v["errors"][:3])

    result.accepted = bool(
        result.functional_correct and result.scorer_ok and not violations
    )
    return result


__all__ = ["RunResult", "run_weight_fn"]
