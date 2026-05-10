"""LLM 用 prompt template.

LLM 出力の format:
- Python ```code block``` のみ
- 単一の純関数 `weight_fn(channel_out, channel_in, kernel_coord) -> float`
- import 不可、外部関数呼出不可、stdlib 型 (int / tuple / float) のみ
"""
from __future__ import annotations

# 主催者 helper の signature を context として明示 (= LLM が ONNX 構造を知らなくても書ける)
HELPER_SIGNATURE = """
neurogolf_utils.single_layer_conv2d_network(weight_fn, kernel_size) は
[1, 10, 30, 30] (= one-hot 10 channel) 入力に対し、
weight_fn が決める conv weight で ONNX (Conv op, padding=kernel_size//2) を生成する。

weight_fn signature:
    def weight_fn(channel_out: int, channel_in: int, kernel_coord: tuple[int, int]) -> float:
        # channel_out, channel_in: 0..9 (= color)
        # kernel_coord: (-(K//2)..(K//2), -(K//2)..(K//2))
        # return: weight value (typically -1.0, 0.0, 1.0)

例 (= "color 6 を color 2 に置換、color 7 不変、他 channel identity"):
    def weight_fn(channel_out, channel_in, kernel_coord):
        if kernel_coord != (0, 0):
            return 0.0
        if channel_out == 6:
            return 0.0
        if channel_out == 2:
            return 1.0 if channel_in in (2, 6) else 0.0
        return 1.0 if channel_out == channel_in else 0.0
"""

PROMPT_TEMPLATE = """\
You are designing a minimal ONNX neural network for the IJCAI-ECAI 2026 NeuroGolf Championship.

Task: {task_id}
Logic (natural language description of the transformation):
{explanation}

Constraints:
- Output ONLY a single Python function `weight_fn` inside a ```python code block```.
- The function signature MUST be exactly:
  `def weight_fn(channel_out: int, channel_in: int, kernel_coord: tuple[int, int]) -> float:`
- Allowed: int / float / tuple literals, comparison, arithmetic, conditional return,
  membership test (`in`), boolean logic.
- Forbidden: `import`, `eval`, `exec`, `open`, function calls (other than int/float/tuple),
  attribute access, subprocess, os, file I/O, randomness, side effects.
- The network is `Conv2D` with kernel_size = {kernel_size}, applied to a [1,10,30,30] tensor.
- Return exactly the conv weight at position (channel_out, channel_in, kernel_coord).
- Aim for sparse weights (mostly 0.0) and small magnitudes (-1.0, 0.0, +1.0).
- The transformation must be functional-correct on every input: do not encode position-dependent
  branching (kernel_coord is the *only* spatial signal you have, and it is local to a {kernel_size}x{kernel_size}
  neighborhood).

Reference (helper signature and example):
{helper}

Your output:
"""


def build_prompt(task_id: str, explanation: str, kernel_size: int = 1) -> str:
    return PROMPT_TEMPLATE.format(
        task_id=task_id,
        explanation=explanation.strip(),
        kernel_size=kernel_size,
        helper=HELPER_SIGNATURE.strip(),
    )
