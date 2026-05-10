"""Fallback ONNX builder for tasks without dedicated builder.

baseline 段階では task276 以外の 399 task は **functional incorrect でよい**
(scope 外、後続 plan で各 task 用 builder を追加)。

ただし以下は **必須**:
- ONNX 制約準拠 (1.44 MB, static shape, no banned op, no kernel_time)
- file が valid な ONNX 形式 (= submission.zip 構造を満たす)
- scorer が crash しない (= score_network が `score >= 0` を返す)

設計: 1x1 Conv2D 1 layer, weight 全 0 (= 全 channel が 0 出力)。
これは大半 task で functional incorrect だが ONNX validator 通過 + 制約準拠。
"""
from __future__ import annotations

import onnx

from ._helpers import single_layer_conv2d_network


def _zero_weight(channel_out: int, channel_in: int, kernel_coord: tuple[int, int]) -> float:
    return 0.0


def build() -> onnx.ModelProto:
    """全 channel zero output の 1x1 Conv2D ONNX model."""
    return single_layer_conv2d_network(_zero_weight, kernel_size=1)
