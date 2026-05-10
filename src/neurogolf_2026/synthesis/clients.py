"""LLM client abstraction.

interface: synthesize(task_id, explanation, kernel_size) -> raw_output: str
- DummyClient: deterministic、 task276 reference + 全 task 用 trivial template を返す
- AgentDispatchClient: 実装は次 plan (Claude Code の Agent tool は CLI 内のみ呼び出し可)。
  本 plan では NotImplementedError を上げる plug-point として用意。
- 実 Anthropic / OpenAI client は別 plan で plug-in (API key + cost 管理)。
"""
from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def synthesize(self, task_id: str, explanation: str, kernel_size: int = 1) -> str:
        ...


# task276 の reference Python 実装 (= 既知 functional correct な weight_fn のソース)
_TASK276_REFERENCE = '''\
```python
def weight_fn(channel_out, channel_in, kernel_coord):
    if kernel_coord != (0, 0):
        return 0.0
    if channel_out == 6:
        return 0.0
    if channel_out == 2:
        if channel_in == 2 or channel_in == 6:
            return 1.0
        return 0.0
    if channel_out == channel_in:
        return 1.0
    return 0.0
```
'''


# 任意 task 用 trivial template (= zero-conv、 functional incorrect だが 安全 ONNX)
_ZERO_TEMPLATE = '''\
```python
def weight_fn(channel_out, channel_in, kernel_coord):
    return 0.0
```
'''


class DummyClient:
    """deterministic stub。

    task276 のみ reference code を返し、その他 task は zero-template を返す。
    test 用。実 LLM 呼出に依存せず pipeline end-to-end を pin する。
    """

    def synthesize(self, task_id: str, explanation: str, kernel_size: int = 1) -> str:
        del explanation, kernel_size  # unused
        if task_id == "task276":
            return _TASK276_REFERENCE
        return _ZERO_TEMPLATE


class AgentDispatchClient:
    """Claude Code の Agent tool 経由 dispatch。

    本 plan では interface のみ。実装は CLI 経由で Agent tool を直接呼ぶ運用 (= 開発者が
    docs/dev/exp002-llm-synthesis-design.md に記載する手動 dispatch、その出力を pipeline
    に paste-in する形) を想定。次 plan で headless dispatch を実装。
    """

    def synthesize(self, task_id: str, explanation: str, kernel_size: int = 1) -> str:
        raise NotImplementedError(
            "AgentDispatchClient: use Claude Code Agent tool externally and pipe response. "
            "See docs/dev/exp002-llm-synthesis-design.md."
        )


__all__ = ["LLMClient", "DummyClient", "AgentDispatchClient"]
