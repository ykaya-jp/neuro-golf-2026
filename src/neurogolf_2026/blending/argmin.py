"""Per-task argmin selection across multiple ONNX sources.

戦略: ONNX file size (= proxy of cost = params + memory_bytes) で min を選ぶ。
真の cost は score_network() で測れるが build 時に全 source × 全 task で score を
走らせると 2000+ 回の重い計算になるため、 file size を proxy にしてから
build_blended で 最終 採用 ONNX のみ score_network gate する 二段方式。

License hedge: 公開 dataset の license は各作者依存だが、 NeuroGolf comp の
submission は kaggle competitions submit 経由で公開作品の流用が一般的に
許容される (= Halite II 等先例)。 主催者 ban announcement あれば即撤回。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neurogolf_2026.build_submission import _validate_onnx


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


def select_per_task(
    task_num: int,
    sources: list[Source],
    repo_root: Path,
    *,
    self_raw: bytes | None = None,
    self_size: int | None = None,
) -> tuple[str, bytes] | None:
    """Per-task で min size source の raw ONNX を返す.

    Args:
        task_num: 1..400
        sources: 評価対象の source list
        repo_root: 解決用 base path
        self_raw: 自前 ONNX の raw bytes (e.g., task276 の hand-craft)
        self_size: self_raw の onnx file size (cost proxy)

    Returns:
        (source_name, raw_bytes) — banned op 検査 pass、 size 最小の source。
        全 source / self が banned op fail なら None。
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

    # size 昇順で sort、 banned op 違反ない最初の 1 つを採用
    candidates.sort(key=lambda x: x[0])
    task_id = f"task{task_num:03d}"
    for size, name, raw in candidates:
        violations = _validate_onnx(task_id, raw)
        if not violations:
            return name, raw
    return None
