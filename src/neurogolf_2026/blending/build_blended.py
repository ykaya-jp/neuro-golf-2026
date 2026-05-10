"""blended submission.zip 構築 (= 自前 + 公開 source の per-task argmin)."""
from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

from neurogolf_2026.build_submission import (
    NUM_TASKS,
    _serialize,
    _validate_onnx,
)
from neurogolf_2026.networks import REGISTRY, get_builder

from .argmin import Source, select_per_task
from .source_pool import DEFAULT_SOURCES, inventory

REPO_ROOT = Path(__file__).resolve().parents[3]


def build_blended_zip(
    out_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    sources: list[Source] | None = None,
    use_self_registry: bool = True,
    fallback_to_zero: bool = True,
    summary_path: Path | None = None,
) -> dict:
    """全 400 task で argmin source を選び submission.zip を構築.

    Args:
        out_path: output zip path
        sources: 評価対象 source list (default: DEFAULT_SOURCES = 7 sources)
        use_self_registry: True なら src/neurogolf_2026/networks/REGISTRY (= self) を 1 source として加える
        fallback_to_zero: select_per_task が None を返した task で fallback (zero-conv) を入れるか
        summary_path: 集約 JSON の出力先 (None なら out_path.with_suffix('.summary.json'))

    Returns:
        summary dict (per-task source 採用 + total byte / fallback 数 + ban op 違反 数)
    """
    sources = sources if sources is not None else DEFAULT_SOURCES
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = summary_path or out_path.with_suffix(".summary.json")

    # 自前 registry の per-task raw bytes を 事前 cache (= 各 task で再 build しない)
    self_cache: dict[str, bytes] = {}
    if use_self_registry:
        for n in range(1, NUM_TASKS + 1):
            tk = f"task{n:03d}"
            if tk in REGISTRY:
                model = REGISTRY[tk]()
                self_cache[tk] = _serialize(model)

    selections: dict[str, dict] = {}
    fallback_count = 0
    total_bytes = 0
    by_source: dict[str, int] = {}

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for n in range(1, NUM_TASKS + 1):
            tk = f"task{n:03d}"
            self_raw = self_cache.get(tk)
            self_size = len(self_raw) if self_raw is not None else None

            sel = select_per_task(
                n, sources, repo_root,
                self_raw=self_raw, self_size=self_size,
            )

            if sel is None:
                # 全 source 失敗、 fallback
                if not fallback_to_zero:
                    selections[tk] = {"source": None, "size": None, "status": "skip"}
                    fallback_count += 1
                    continue
                # 自前 fallback (zero-conv) を生成
                builder = get_builder(tk)
                model = builder()
                raw = _serialize(model)
                source_name = "fallback-zero-conv"
                fallback_count += 1
            else:
                source_name, raw = sel

            zf.writestr(f"{tk}.onnx", raw)
            selections[tk] = {
                "source": source_name,
                "size": len(raw),
                "status": "selected",
            }
            total_bytes += len(raw)
            by_source[source_name] = by_source.get(source_name, 0) + 1

    out_path.write_bytes(buf.getvalue())

    summary = {
        "_meta": {
            "out": str(out_path),
            "zip_size_bytes": out_path.stat().st_size,
            "uncompressed_total_bytes": total_bytes,
            "task_count": NUM_TASKS,
            "fallback_count": fallback_count,
            "by_source_count": by_source,
            "sources_evaluated": [s.name for s in sources],
        },
        "per_task": selections,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


__all__ = ["build_blended_zip"]
