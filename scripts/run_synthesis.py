"""CLI for synthesis pipeline (= exp002).

Usage:
    uv run python scripts/run_synthesis.py --client dummy \\
        --tasks task276,task004,task016,task140,task276 \\
        --out outputs/synthesis/dummy-$(date -u +%Y%m%dT%H%M%S)/

    # 既知 simple task 5 件で dry-run
    uv run python scripts/run_synthesis.py --client dummy --preset color5
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from neurogolf_2026.synthesis import (
    DummyClient,
    SynthesisResult,
    run_synthesis,
)
from neurogolf_2026.synthesis.clients import AgentDispatchClient

PRESETS = {
    "color5": ["task276", "task016", "task040", "task267", "task373"],
    "single": ["task276"],
}

CLIENTS = {
    "dummy": DummyClient,
    "agent": AgentDispatchClient,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--client", choices=list(CLIENTS), default="dummy")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tasks", help="comma-separated task ids (e.g. task276,task004)")
    g.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--out", type=Path, help="output dir (default: outputs/synthesis/<run_id>/)")
    ap.add_argument("--kernel-size", type=int, default=1)
    args = ap.parse_args()

    if args.preset:
        task_ids = PRESETS[args.preset]
    else:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if args.out is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        args.out = Path("outputs/synthesis") / f"{args.client}-{run_id}"

    client = CLIENTS[args.client]()
    results = run_synthesis(client, task_ids, args.out, kernel_size=args.kernel_size)

    accepted = [r for r in results if r.accepted]
    print(f"\n=== synthesis run summary ===")
    print(f"client: {type(client).__name__}")
    print(f"out: {args.out}")
    print(f"task_count: {len(results)}, accepted: {len(accepted)}")
    for r in results:
        rr = r.run_result or {}
        marker = "✓" if r.accepted else ("○" if r.status == "missing_explanation" else "✗")
        cost = rr.get("cost", "?")
        # score が None (= scorer fail) と 0.0 (= 真に 0 で計算済) を区別
        s = rr.get("score")
        score = f"{s:.3f}" if s is not None else "n/a"
        print(f"  {marker} {r.task_id} [{r.status}] cost={cost} score={score}")
    return 0 if any(r.accepted for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
