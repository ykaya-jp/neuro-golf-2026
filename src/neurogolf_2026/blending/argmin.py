"""Per-task argmin selection across multiple ONNX sources.

戦略 v2 (2026-05-11): **真の cost (= score_network() の memory_bytes + params)** で
per-task min を選ぶ。 加えて以下を全て gate:
- banned op 不使用 (`_validate_onnx`)
- functional correct (= validate_task の `functional_correct: true`)
- scorer_ok (= score_network が None を返さない、 = scorer-poison op 不使用)

v1 (size proxy) の問題:
- size 小 != true cost 小 (例: ONNX header / metadata で乖離)
- size 小でも functional incorrect / scorer crash の source を採用 → 0 点で task ロス
- magmacot-new-blending / konbu17-blended-401-v117 は **全 task で scorer crash** 観測

License hedge: 公開 dataset の license は各作者依存だが、 NeuroGolf comp の
submission は kaggle competitions submit 経由で公開作品の流用が一般的に
許容される (= Halite II 等先例)。 主催者 ban announcement あれば即撤回。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnx

from neurogolf_2026.build_submission import _validate_onnx
from neurogolf_2026.synthesis.runner import _temp_registry
from neurogolf_2026.validate import validate_task


@dataclass(frozen=True)
class Source:
    """1 つの公開 dataset / kernel output の ONNX 提供 source."""

    name: str
    path_template: str  # e.g., "data/external/konbu17-may8/sub/task{n:03d}.onnx"
    license: str = "unknown"

    def candidate_path(self, task_num: int, repo_root: Path) -> Path:
        return repo_root / self.path_template.format(n=task_num)

    def has(self, task_num: int, repo_root: Path) -> bool:
        return self.candidate_path(task_num, repo_root).exists()

    def read(self, task_num: int, repo_root: Path) -> bytes:
        return self.candidate_path(task_num, repo_root).read_bytes()

    def size(self, task_num: int, repo_root: Path) -> int | None:
        p = self.candidate_path(task_num, repo_root)
        if not p.exists():
            return None
        return p.stat().st_size


def _evaluate_candidate(task_id: str, raw: bytes) -> tuple[bool, int | None, float | None, str]:
    """raw ONNX bytes に対して score_network() を経由して評価.

    Returns: (accepted, cost, score, reason)
      accepted = True iff functional_correct and scorer_ok and no banned-op violation
    """
    violations = _validate_onnx(task_id, raw)
    if violations:
        return False, None, None, f"banned op / constraint: {violations[:1]}"
    try:
        model = onnx.load_from_string(raw)
    except Exception as e:
        return False, None, None, f"onnx parse: {e!r}"
    try:
        with _temp_registry(task_id, model):
            v = validate_task(task_id, strict=False, verbose=False)
    except Exception as e:
        return False, None, None, f"validate_task crash: {e!r}"
    if not v["scorer_ok"]:
        return False, None, None, f"scorer crash: {v['errors'][:1]}"
    if not v["functional_correct"]:
        return False, v.get("cost"), v.get("score"), \
               f"functional incorrect (arc_agi {v['arc_agi']}, arc_gen {v['arc_gen']})"
    return True, v["cost"], v["score"], "accepted"


def select_per_task(
    task_num: int,
    sources: list[Source],
    repo_root: Path,
    *,
    self_raw: bytes | None = None,
    self_size: int | None = None,
    quick_mode: bool = False,
) -> tuple[str, bytes, dict] | None:
    """Per-task で **真の cost** で min を選び、 functional correct + scorer_ok を全候補で gate.

    Args:
        quick_mode: True なら size 昇順で 上から試行、 最初の accepted を採用 (= lazy)。
                    False なら全 candidate を score_network → 真の cost で argmin。

    Returns:
        (source_name, raw_bytes, eval_info) where eval_info has cost / score / size
        全 source / self が fail なら None。
    """
    candidates: list[tuple[int, str, bytes]] = []
    if self_raw is not None and self_size is not None:
        candidates.append((self_size, "self", self_raw))
    for src in sources:
        size = src.size(task_num, repo_root)
        if size is None:
            continue
        candidates.append((size, src.name, src.read(task_num, repo_root)))

    if not candidates:
        return None

    task_id = f"task{task_num:03d}"

    if quick_mode:
        # size 昇順で 1 件ずつ accept 判定、 最初の accepted を採用 (= lazy 早期 break)
        candidates.sort(key=lambda x: x[0])
        for size, name, raw in candidates:
            accepted, cost, score, reason = _evaluate_candidate(task_id, raw)
            if accepted:
                return name, raw, {"size": size, "cost": cost, "score": score, "source": name}
        return None

    # 全 candidate を評価して 真 cost で argmin
    evaluated: list[tuple[int, float, str, bytes]] = []
    for size, name, raw in candidates:
        accepted, cost, score, reason = _evaluate_candidate(task_id, raw)
        if accepted and cost is not None:
            evaluated.append((cost, score or 0.0, name, raw))

    if not evaluated:
        return None
    evaluated.sort(key=lambda x: x[0])  # min cost
    cost, score, name, raw = evaluated[0]
    return name, raw, {"cost": cost, "score": score, "source": name,
                       "size": len(raw), "candidates_evaluated": len(candidates)}
