"""task276 — Color Swapping (Complexity 1).

logic (arc_explanations.json):
  "Replace all occurrences of color 6 with color 2,
   while keeping color 7 unchanged."

10-channel one-hot 入力なので、color 6 channel を color 2 channel に
コピーし、color 6 channel を 0 にする 1x1 Conv2D 1 layer で表現可能。

cost 試算 (W6 first-principles より):
  params = C_out * C_in * K * K = 10 * 10 * 1 * 1 = 100
  memory = params * 4 byte (float32) = 400 byte
  ※ activation tensor の memory は scorer が profiler から計算するので
    実測値は score_network() で確認。
  期待 score = max(1, 25 - ln(500)) ≈ 18.79
"""
from __future__ import annotations

import onnx

from ._helpers import single_layer_conv2d_network


def _weight(channel_out: int, channel_in: int, kernel_coord: tuple[int, int]) -> float:
    """task276 用 1x1 Conv2D weight function.

    Identity を base に以下を上書き:
    - color 6 channel は 0 になる (= 削除)
    - color 2 channel は元の color 2 + color 6 の合算 (= color 6 を吸収)
    - 他 channel (0,1,3,4,5,7,8,9) は identity
    """
    if kernel_coord != (0, 0):
        return 0.0
    if channel_out == 6:
        # color 6 出力は zero (= 全部消す)
        return 0.0
    if channel_out == 2:
        # color 2 出力 = color 2 + color 6
        if channel_in in (2, 6):
            return 1.0
        return 0.0
    # その他 channel は identity
    if channel_out == channel_in:
        return 1.0
    return 0.0


def build() -> onnx.ModelProto:
    """task276 用 ONNX model (1x1 Conv2D, 1 layer)."""
    return single_layer_conv2d_network(_weight, kernel_size=1)
