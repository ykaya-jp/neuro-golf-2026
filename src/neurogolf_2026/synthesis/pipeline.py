"""Main orchestrator: task list × LLM client → results.json."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurogolf_2026.synthesis.clients import LLMClient
from neurogolf_2026.synthesis.extractor import ExtractError, extract_weight_fn
from neurogolf_2026.synthesis.runner import RunResult, run_weight_fn

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPLANATIONS_PATH = (
    REPO_ROOT
    / "data"
    / "external"
    / "logic-for-each-arc-task"
    / "arc_explanations.json"
)


def _load_explanations() -> dict[str, str]:
    if not EXPLANATIONS_PATH.exists():
        raise FileNotFoundError(f"arc_explanations.json not found at {EXPLANATIONS_PATH}")
    return json.loads(EXPLANATIONS_PATH.read_text())


@dataclass
class SynthesisResult:
    task_id: str
    status: str  # "accepted" | "extract_failed" | "run_failed" | "missing_explanation"
    raw_output: str | None = None
    extract_error: str | None = None
    run_result: dict | None = None  # asdict(RunResult)
    accepted: bool = False


def run_synthesis(
    client: LLMClient,
    task_ids: list[str],
    out_dir: Path | str,
    *,
    kernel_size: int = 1,
    save_raw: bool = True,
) -> list[SynthesisResult]:
    """task_ids ごとに client.synthesize → extract → runner、結果を out_dir/results.json に書く."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    explanations = _load_explanations()
    results: list[SynthesisResult] = []

    for tk in task_ids:
        explanation = explanations.get(tk)
        if not explanation:
            results.append(SynthesisResult(
                task_id=tk, status="missing_explanation",
            ))
            continue

        # 1) LLM dispatch
        try:
            raw = client.synthesize(tk, explanation, kernel_size=kernel_size)
        except Exception as e:
            results.append(SynthesisResult(
                task_id=tk, status="run_failed",
                extract_error=f"client.synthesize raised: {e!r}",
            ))
            continue

        if save_raw:
            (out_path / f"{tk}_raw.txt").write_text(raw)

        # 2) extractor
        try:
            fn = extract_weight_fn(raw)
        except ExtractError as e:
            results.append(SynthesisResult(
                task_id=tk, status="extract_failed",
                raw_output=raw[:500],
                extract_error=str(e),
            ))
            continue

        # 3) runner
        rr: RunResult = run_weight_fn(tk, fn, kernel_size=kernel_size)
        results.append(SynthesisResult(
            task_id=tk,
            status="accepted" if rr.accepted else "run_failed",
            raw_output=None if save_raw else raw[:500],
            run_result=asdict(rr),
            accepted=rr.accepted,
        ))

    # 集約 results.json
    summary: dict[str, Any] = {
        "_meta": {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "client": type(client).__name__,
            "kernel_size": kernel_size,
            "task_count": len(task_ids),
            "accepted_count": sum(1 for r in results if r.accepted),
        },
        "results": [asdict(r) for r in results],
    }
    (out_path / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    return results


__all__ = ["SynthesisResult", "run_synthesis"]
