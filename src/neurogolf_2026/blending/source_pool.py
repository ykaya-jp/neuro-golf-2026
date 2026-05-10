"""公開 source の inventory.

各 source は data/external/<name>/sub/task<NNN>.onnx 形式 (= kernel output) または
data/external/<name>/submission/task<NNN>.onnx (= dataset 形式)。
"""
from __future__ import annotations

from pathlib import Path

from .argmin import Source

# NOTE: 各 path_template は repo_root 起点の format string。 task_num を {n:03d} で。
DEFAULT_SOURCES: list[Source] = [
    # konbu17/may-8-updated kernel output: 400 ONNX, sample task276=217 byte (= 5/8 update、 比較的新)
    Source(
        name="konbu17-may8",
        path_template="data/external/konbu17-may8/sub/task{n:03d}.onnx",
        license="kaggle-kernel-output (= Apache-2.0 derived)",
    ),
    # afr1ste/5480-41 kernel output: 400 ONNX (LB 5480.41 score の現物)
    Source(
        name="afr1ste-5480-41",
        path_template="data/external/afr1ste-neurogolf-5480-41-current-rules-score/sub/task{n:03d}.onnx",
        license="kaggle-kernel-output",
    ),
    # jonathanchan/ngc26 blending: 401 ONNX
    Source(
        name="jonathanchan-ngc26",
        path_template="data/external/jonathanchan-ngc26-constraint-smart-logic-mix-blending/tmp_build/blend/task{n:03d}.onnx",
        license="kaggle-kernel-output",
    ),
    # magmacot/new-blending: 401 ONNX
    Source(
        name="magmacot-new-blending",
        path_template="data/external/magmacot-neurogolf-new-blending/sub/task{n:03d}.onnx",
        license="kaggle-kernel-output",
    ),
    # konbu17/blended-401-v117 dataset: 401 ONNX (LB 5331+, sample task001=5582 byte)
    Source(
        name="konbu17-blended-401-v117",
        path_template="data/external/neurogolf-2026-blended-401-v117/task{n:03d}.onnx",
        license="kaggle-dataset (= Apache-2.0 implied)",
    ),
    # karnakbaevarthur/task-transformation-library: 266 ONNX
    Source(
        name="karnakbaevarthur-task-library",
        path_template="data/external/neurogolf-2026-task-transformation-library/submission/task{n:03d}.onnx",
        license="kaggle-dataset",
    ),
    # karnakbaevarthur/logic-for-each-arc-task: 204 ONNX
    Source(
        name="karnakbaevarthur-logic",
        path_template="data/external/logic-for-each-arc-task/submission/task{n:03d}.onnx",
        license="kaggle-dataset",
    ),
    # agentzz/neurogolf-submit-6284-v2 kernel output: 400 ONNX (LB 6284 帯 = 最高 score 帯)
    Source(
        name="agentzz-6284-v2",
        path_template="data/external/agentzz-neurogolf-submit-6284-v2/sub/task{n:03d}.onnx",
        license="kaggle-kernel-output",
    ),
    # jazivxt/infinitesimals kernel output: 397 ONNX (5/10 最新 update, votes 76)
    Source(
        name="jazivxt-infinitesimals",
        path_template="data/external/jazivxt-infinitesimals/sub/task{n:03d}.onnx",
        license="kaggle-kernel-output",
    ),
    # imaadmahmood/neurogolf-2026-rule-based-onnx-solver: 400 ONNX (votes 75)
    Source(
        name="imaadmahmood-rule-based",
        path_template="data/external/imaadmahmood-neurogolf-2026-rule-based-onnx-solver/sub/task{n:03d}.onnx",
        license="kaggle-kernel-output",
    ),
    # hanifnoerrofiq/8-may-update: 400 ONNX (5/7 update, votes 46)
    Source(
        name="hanifnoerrofiq-8may",
        path_template="data/external/hanifnoerrofiq-8-may-update/sub/task{n:03d}.onnx",
        license="kaggle-kernel-output",
    ),
    # vyankteshdwivedi/neurogolf-multi-source-onnx-solver: 371 ONNX (votes 64)
    Source(
        name="vyankteshdwivedi-multi-source",
        path_template="data/external/vyankteshdwivedi-neurogolf-multi-source-onnx-solver/sub/task{n:03d}.onnx",
        license="kaggle-kernel-output",
    ),
    # konbu17/neurogolf-2026-blended-till-4-27 kernel output: 401 ONNX (votes 60)
    Source(
        name="konbu17-till-4-27",
        path_template="data/external/konbu17-neurogolf-2026-blended-till-4-27/sub/task{n:03d}.onnx",
        license="kaggle-kernel-output",
    ),
]


def inventory(repo_root: Path, sources: list[Source] | None = None) -> dict[str, dict[int, int]]:
    """各 source の per-task ONNX 在庫を集計.

    Returns:
        {source_name: {task_num: size_bytes}}
    """
    sources = sources or DEFAULT_SOURCES
    out: dict[str, dict[int, int]] = {}
    for src in sources:
        per_task: dict[int, int] = {}
        for n in range(1, 401):
            sz = src.size(n, repo_root)
            if sz is not None:
                per_task[n] = sz
        out[src.name] = per_task
    return out
