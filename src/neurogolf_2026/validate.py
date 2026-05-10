"""Validate per-task ONNX via official score_network().

AC-1: 選択 baseline task が functional correct (= 全 pair pass)
AC-3: scorer-poison op を踏まない (= score_network が `>= 0` を返す)

Usage:
    # 1 task のみ functional correct を strict 検証 (= AC-1)
    uv run python -m neurogolf_2026.validate --task task276 --strict

    # 全 task で scorer-poison check (= AC-3)
    uv run python -m neurogolf_2026.validate --all --check-scorer-poison
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime

from neurogolf_2026.networks import REGISTRY, get_builder
from neurogolf_2026.networks._helpers import (
    convert_to_numpy,
    score_network,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"


def _load_task_examples(task_id: str) -> dict | None:
    """data/raw/<task_id>.json を読み込む。"""
    path = RAW_DIR / f"{task_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _make_session(model: onnx.ModelProto, task_num: int) -> tuple[onnxruntime.InferenceSession, str]:
    """sanitize node names + profiling 有効化した session を返す。

    neurogolf_utils.verify_network と同じ初期化手順を踏襲。
    """
    sanitized = onnx.load_from_string(model.SerializeToString())
    for node in sanitized.graph.node:
        node.name = node.output[0] if node.output else ""
        if "kernel_time" in node.name:
            raise ValueError(f"node name contains 'kernel_time': {node.name}")

    options = onnxruntime.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    # task ごとに profile prefix 分離 (2026-05-06 fix 由来)
    options.profile_file_prefix = f"{task_num:03d}"
    session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
    return session, "n/a"


def _verify_pairs(
    session: onnxruntime.InferenceSession,
    examples: list[dict],
) -> tuple[int, int, list[str]]:
    """examples 全件で functional correct を判定。

    Returns: (pass_count, fail_count, runtime_errors)
    runtime_errors は OrtException など runtime crash のみ (= scorer-poison 候補)。
    output mismatch は wrong に計上、runtime_errors には載せない。
    """
    right = wrong = 0
    runtime_errors: list[str] = []
    for i, ex in enumerate(examples):
        bench = convert_to_numpy(ex)
        if not bench:
            continue
        try:
            out = session.run(None, {"input": bench["input"]})[0]
            if np.array_equal(out, bench["output"]):
                right += 1
            else:
                wrong += 1
        except onnxruntime.OrtException as e:  # type: ignore[attr-defined]
            wrong += 1
            runtime_errors.append(f"OrtException at pair {i}: {e!r}")
    return right, wrong, runtime_errors


def validate_task(
    task_id: str,
    *,
    strict: bool = True,
    verbose: bool = False,
) -> dict:
    """Validate a single task ONNX.

    Returns dict with keys:
      task_id, params, memory_bytes, cost, score,
      arc_agi: {right, wrong}, arc_gen: {right, wrong},
      functional_correct: bool, scorer_ok: bool, errors: list[str]
    """
    result: dict = {
        "task_id": task_id,
        "params": None,
        "memory_bytes": None,
        "cost": None,
        "score": None,
        "arc_agi": {"right": 0, "wrong": 0},
        "arc_gen": {"right": 0, "wrong": 0},
        "functional_correct": False,
        "scorer_ok": False,
        "errors": [],
    }

    examples = _load_task_examples(task_id)
    if examples is None and strict:
        result["errors"].append(f"task data not found: {task_id}.json")
        return result

    builder = get_builder(task_id)
    try:
        model = builder()
    except Exception as e:
        result["errors"].append(f"builder failed: {e!r}")
        return result

    task_num = int(task_id.replace("task", ""))
    try:
        session, _ = _make_session(model, task_num)
    except Exception as e:
        result["errors"].append(f"session init failed: {e!r}")
        return result

    if examples is not None:
        agi = _verify_pairs(session, examples.get("train", []) + examples.get("test", []))
        gen = _verify_pairs(session, examples.get("arc-gen", []))
        result["arc_agi"] = {"right": agi[0], "wrong": agi[1]}
        result["arc_gen"] = {"right": gen[0], "wrong": gen[1]}
        result["functional_correct"] = (agi[1] == 0 and gen[1] == 0
                                        and agi[0] + gen[0] > 0)
        # runtime crash は scorer-poison 候補なので errors に明示
        result["errors"].extend(agi[2] + gen[2])

    # scorer (= score_network) を必ず走らせる (= AC-3 check_scorer_poison)
    sanitized = onnx.load_from_string(model.SerializeToString())
    for node in sanitized.graph.node:
        node.name = node.output[0] if node.output else ""
    try:
        # score_network は (memory, params) tuple を返す
        memory, params = score_network(sanitized, session.end_profiling())
        result["memory_bytes"] = memory
        result["params"] = params
        if memory is None or params is None:
            result["errors"].append("score_network returned None")
        elif memory < 0 or params < 0:
            result["errors"].append(f"score_network returned negative: memory={memory}, params={params}")
        else:
            result["cost"] = memory + params
            result["score"] = max(1.0, 25.0 - math.log(max(1.0, memory + params)))
            result["scorer_ok"] = True
    except Exception as e:
        result["errors"].append(f"score_network crashed: {e!r}")

    if verbose:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", help="task_id (e.g. task276)")
    g.add_argument("--all", action="store_true", help="iterate all 400 tasks")

    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if functional incorrect (= AC-1)")
    ap.add_argument("--check-scorer-poison", action="store_true",
                    help="exit non-zero if score_network crashes for any task (= AC-3)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.task:
        r = validate_task(args.task, strict=args.strict, verbose=args.verbose)
        if not args.verbose:
            print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
        # AC-1 strict: functional_correct 必須
        if args.strict and not r["functional_correct"]:
            print(f"\nFAIL ({args.task}): not functional correct", file=sys.stderr)
            return 1
        # AC-3 strict: scorer_ok 必須
        if args.check_scorer_poison and not r["scorer_ok"]:
            print(f"\nFAIL ({args.task}): scorer_ok = False", file=sys.stderr)
            return 1
        return 0

    # --all
    fc_count = poison_count = 0
    poison_tasks: list[str] = []
    summary: list[dict] = []
    for n in range(1, 401):
        tk = f"task{n:03d}"
        r = validate_task(tk, strict=False, verbose=False)
        summary.append(r)
        if r["functional_correct"]:
            fc_count += 1
        if not r["scorer_ok"]:
            poison_count += 1
            poison_tasks.append(tk)

    print(f"\nFunctional correct: {fc_count} / 400 (registry: {list(REGISTRY)})")
    print(f"Scorer OK: {400 - poison_count} / 400")
    if poison_tasks:
        print(f"Scorer-poison tasks: {poison_tasks[:10]}{'...' if len(poison_tasks) > 10 else ''}")
    # 上位 5 cost
    valid = [r for r in summary if r.get("cost") is not None]
    if valid:
        print(f"\nTop 5 by score:")
        for r in sorted(valid, key=lambda x: -x["score"])[:5]:
            print(f"  {r['task_id']}: score {r['score']:.3f}, cost {r['cost']}")

    if args.check_scorer_poison and poison_count > 0:
        print(f"\nFAIL: {poison_count} scorer-poison tasks", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
