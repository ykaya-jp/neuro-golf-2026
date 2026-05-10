"""Build blended submission from public + self sources.

Usage:
    uv run python scripts/run_blending.py --out submissions/blended.zip
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neurogolf_2026.blending import build_blended_zip
from neurogolf_2026.blending.source_pool import DEFAULT_SOURCES, inventory


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("submissions/blended.zip"))
    ap.add_argument("--inventory-only", action="store_true",
                    help="print source inventory and exit")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if args.inventory_only:
        inv = inventory(repo_root)
        for name, per_task in inv.items():
            print(f"  {name}: {len(per_task)} tasks (sample sizes: "
                  f"{[v for k, v in sorted(per_task.items())[:3]]})")
        return 0

    summary = build_blended_zip(args.out, repo_root=repo_root)
    print(f"\nblended submission: {args.out}")
    print(f"zip size: {summary['_meta']['zip_size_bytes']:,} byte")
    print(f"uncompressed total: {summary['_meta']['uncompressed_total_bytes']:,} byte")
    print(f"fallback count: {summary['_meta']['fallback_count']}")
    print(f"\nby source count:")
    for src, cnt in sorted(summary["_meta"]["by_source_count"].items(),
                           key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
