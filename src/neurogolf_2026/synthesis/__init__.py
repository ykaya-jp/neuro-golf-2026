"""LLM-driven program synthesis pipeline (= 候補 2 framework, exp002).

エントリ: pipeline.run_synthesis(client, task_ids, out_dir)
client interface: clients.LLMClient (synthesize(task_id, explanation) -> code: str)
"""
from __future__ import annotations

from .clients import DummyClient, LLMClient
from .extractor import ExtractError, extract_weight_fn
from .pipeline import SynthesisResult, run_synthesis
from .runner import RunResult, run_weight_fn

__all__ = [
    "DummyClient",
    "LLMClient",
    "ExtractError",
    "extract_weight_fn",
    "SynthesisResult",
    "run_synthesis",
    "RunResult",
    "run_weight_fn",
]
