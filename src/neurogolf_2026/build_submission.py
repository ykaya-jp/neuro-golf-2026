"""Build NeuroGolf 2026 submission.zip.

Usage:
    uv run python -m neurogolf_2026.build_submission [--out submissions/submission.zip]

全 400 task に対し networks.get_builder() で ONNX を取得、submission.zip に同梱。
冪等 (= 同 input で同 zip を生成)。
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import onnx

from neurogolf_2026.networks import get_builder

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "submissions" / "submission.zip"
NUM_TASKS = 400
ONNX_FILE_LIMIT_BYTES = int(1.44 * 1024 * 1024)
BANNED_OPS = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}


def _serialize(model: onnx.ModelProto) -> bytes:
    """ONNX → bytes (= raw_data 含む)。"""
    return model.SerializeToString()


def _validate_onnx(task_id: str, raw: bytes) -> list[str]:
    """ONNX bytes が AC-2 制約 (1.44 MB / banned op / kernel_time / Multi-IO) を満たすか。

    Returns list of violation messages (empty list = OK).
    """
    violations: list[str] = []
    if len(raw) > ONNX_FILE_LIMIT_BYTES:
        violations.append(f"size {len(raw)} byte > {ONNX_FILE_LIMIT_BYTES} byte limit")

    model = onnx.load_from_string(raw)

    # banned op check
    for node in model.graph.node:
        if node.op_type in BANNED_OPS:
            violations.append(f"banned op: {node.op_type} in node {node.name}")
        if "kernel_time" in node.name:
            violations.append(f"kernel_time substring in node name: {node.name}")
        for output_name in node.output:
            if "kernel_time" in output_name:
                violations.append(f"kernel_time substring in output: {output_name}")

    # tensor name check
    for init in model.graph.initializer:
        if "kernel_time" in init.name:
            violations.append(f"kernel_time substring in initializer: {init.name}")

    # Multi-input/output check (2026-05-06 ban)
    if len(model.graph.input) != 1:
        violations.append(f"multi-input graph: {len(model.graph.input)} inputs")
    if len(model.graph.output) != 1:
        violations.append(f"multi-output graph: {len(model.graph.output)} outputs")

    return violations


def build(
    out_path: Path = DEFAULT_OUT,
    num_tasks: int = NUM_TASKS,
    strict: bool = True,
    verbose: bool = False,
) -> dict[str, list[str]]:
    """Build submission.zip。

    Args:
        out_path: 出力 zip path
        num_tasks: 含める task 数 (default 400)
        strict: True なら制約違反時に raise
        verbose: True なら per-task progress 出力

    Returns:
        {task_id: list of violations} (= 全 task で violations 空なら success)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    violations_map: dict[str, list[str]] = {}

    # 一旦 BytesIO に書いてから path に書く (= atomicity + 冪等)
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for n in range(1, num_tasks + 1):
            task_id = f"task{n:03d}"
            builder = get_builder(task_id)
            try:
                model = builder()
            except Exception as e:
                msg = f"builder failed: {e!r}"
                violations_map[task_id] = [msg]
                if strict:
                    raise RuntimeError(f"{task_id}: {msg}") from e
                continue

            raw = _serialize(model)
            v = _validate_onnx(task_id, raw)
            if v:
                violations_map[task_id] = v
                if strict:
                    raise ValueError(f"{task_id}: {'; '.join(v)}")

            zf.writestr(f"{task_id}.onnx", raw)
            if verbose:
                print(f"  {task_id}: {len(raw)} byte")

    out_path.write_bytes(buf.getvalue())
    if verbose:
        print(f"\nwrote {out_path} ({out_path.stat().st_size:,} byte, {num_tasks} tasks)")

    return violations_map


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output zip path")
    ap.add_argument("--num-tasks", type=int, default=NUM_TASKS, help="number of tasks")
    ap.add_argument("--no-strict", action="store_true", help="continue on violations")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    violations = build(
        out_path=args.out,
        num_tasks=args.num_tasks,
        strict=not args.no_strict,
        verbose=args.verbose,
    )
    if violations:
        print(f"\n{len(violations)} task(s) with violations:", file=sys.stderr)
        for tk, vlist in list(violations.items())[:10]:
            print(f"  {tk}: {'; '.join(vlist)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
