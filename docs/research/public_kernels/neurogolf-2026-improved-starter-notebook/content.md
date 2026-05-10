# neurogolf-2026-improved-starter-notebook — MACs 込みコスト計算と starter 拡張

## [MD]

(notebook 自体に markdown cell ほぼ無し。コードコメントから読み取る。)

`yash9439` による改良版 starter で、初代 V2 が **MACs を計算に含めていなかった** バグを修正している。改良点はコメント中に明記:

> Cost = params + memory_bytes + MACs.
> Critical: V2 was missing MACs! This changes strategy ranking.

## [CODE]

```python
def estimate_model_cost(model):
    """Cost = params + memory_bytes + MACs."""
    total_params = 0
    total_bytes = 0

    tensors = {}  # name → array
    for init in model.graph.initializer:
        arr = onh.to_array(init)
        tensors[init.name] = arr
        total_params += arr.size
        total_bytes += arr.nbytes

    # Estimate MACs
    total_macs = 0
    for node in model.graph.node:
        if node.op_type == "Conv":
            if len(node.input) >= 2 and node.input[1] in tensors:
                w = tensors[node.input[1]]
                if w.ndim == 4:
                    c_out, c_in, kh, kw = w.shape
                    # Output spatial = input spatial (we use 'same' padding)
                    total_macs += c_out * c_in * kh * kw * H * W
        elif node.op_type == "Gemm":
            ...
```

`grid_to_tensor` (= 30x30 grid を 10ch one-hot に encode) と `tensor_to_grid` (= argmax で復元) のユーティリティを再公開。`shapes_match(p)` で input/output 同サイズの task のみを対象とする hint も提供。

## 要点 (W2 抽出)

- **手法 (technique)**: heuristic_handcrafted_weights + cost-aware utility (= 主催者 starter の cost 計算改良版)
- **score (LB)**: starter 自体は LB 値を主張せず、実装 detail を提供するのみ
- **votes**: 108 (= 全 kernel 中 3 位)
- **核心アルゴリズム**:
  1. ARC grid を `(1, 10, 30, 30)` の one-hot tensor にエンコード
  2. ONNX runtime で推論 → argmax で grid 復元 → 一致判定 (= `check_model_correct`)
  3. cost = params + memory_bytes + MACs を per-model で集計し、最小コストの解を採用 (実装方針は `improved-starter` で正本)
- **特徴的な工夫**:
  - **MACs 計算を Conv / Gemm の両方でカバー** (= V2 starter は MACs を抜いていたので、Conv-heavy 解を不当に高評価していた)
  - `tensor_to_grid` で `slice_.max(0) < 0.5` を「セル空白」として扱う閾値ロジックが neurogolf 流 (= 他 kernel もこれを踏襲)
  - `shapes_match(p)` の inline check で「入出力同 shape タスク」と「reshape タスク」を分離する道筋を示す
- **当該コンペでの応用余地**:
  - 我々が starter として真っ先に取り込むべきユーティリティ。MAC 込みコスト計算は **2026-05-06 の rule 変更前の旧採点ロジック** だが、ローカル評価で「どのアプローチが過去採点でどれくらい cheap だったか」を測る際に使える
  - `check_model_correct` の inference loop は per-task verification の reference として再利用
- **限界 / 弱点**:
  - **2026-05-06 採点 rule 変更後** は `25 - log(max(1, memory + params))` に変わり MACs は寄与しなくなった (出典: konbu17 may-8-updated content.md 参照)。本 cost 関数をそのままにすると過剰最適化
  - shape 不一致タスク (= reshape / tile / crop) への対応は別途必要

## 出典

- Kernel URL: [https://www.kaggle.com/code/yash9439/neurogolf-2026-improved-starter-notebook](https://www.kaggle.com/code/yash9439/neurogolf-2026-improved-starter-notebook)
- このディレクトリの `kernel-metadata.json` 参照
