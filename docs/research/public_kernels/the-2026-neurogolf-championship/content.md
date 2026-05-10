# the-2026-neurogolf-championship — 主催者公式 starter

## [MD]

> This starter notebook for the [2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026) is designed to help contestants verify the functional correctness of their networks. It allows one to load example pairs for a task, visualize them, and test whether a candidate network produces expected results across all public competition benchmarks.

主催者 (Michael D. Moffitt) が公開した最小 starter。task 番号を 1 つ指定し、`single_layer_conv2d_network(weight_fn, kernel_size=3)` で **1 層 Conv2D の重みをハードコード** して `verify_network` で正解判定するワークフローを提示している。

## [CODE]

```python
def weight(channel_out, channel_in, kernel_coord):
  if kernel_coord == ( 0,  0) and channel_in == channel_out: return 1.0
  if kernel_coord == ( 0,  0) and channel_in != 5 and channel_out == 0: return -1.0
  if kernel_coord == (-1, -1) and channel_in != 5 and channel_out == 0: return 1.0
  if kernel_coord == (-1, -1) and channel_in != 5 and channel_out == 5: return -1.0
  return 0.0

network = single_layer_conv2d_network(weight, kernel_size=3)
verify_network(network, task_num, examples)
```

依存ライブラリは `onnx==1.21.0` / `onnxruntime==1.24.4` / `onnx-tool==1.0.1` / `numpy==2.4.4` で固定。`/kaggle/input/competitions/neurogolf-2026/neurogolf_utils` から `single_layer_conv2d_network` / `show_examples` / `verify_network` を import する。

## 要点 (W2 抽出)

- **手法 (technique)**: heuristic_handcrafted_weights (= 主催者推奨の base case)
- **score (LB)**: タスク 0 (illustrative) のみ示しているので LB 値はゼロベース。1 タスクあたりに何点取れるかの感覚を掴ませるためのもの
- **votes**: 185 (= 全 kernel 中 1 位)
- **核心アルゴリズム**:
  1. `(channel_out, channel_in, kernel_coord)` を引数とする純関数 `weight(...)` を書く
  2. `single_layer_conv2d_network(weight, kernel_size=3)` がその関数を walk して 3x3 Conv2D の重みテンソルを構築
  3. `verify_network` で train/test pair すべてに対する一致を確認 → ONNX 形式で submit
- **特徴的な工夫**:
  - 「1 層 Conv2D で解ける」最小タスクの存在を提示し、ARC タスクが per-task tiny model で解ける hint を与えている
  - 重みは `+1.0 / -1.0 / 0.0` の三値 (= ternary) のみ。`channel_in == 5` (= 黒以外を抜く) という neurogolf 流の color slot 利用
  - 重み関数は純関数なので code review しやすく、コピペで他タスクに展開しやすい
- **当該コンペでの応用余地**:
  - 単純色 mapping / 反転 / 平行移動などの幾何変換タスクでは `weight(...)` 関数を 1 つ書くだけで解ける
  - ターゲット数値 cost = 三値重み数 + bias 個数 (極小)。スコア理論最大 (25 点近傍) を狙える
- **限界 / 弱点**:
  - 1 層 Conv2D で表現できないタスク (= 多段 logical 推論、object 検出、動的 reshape) は別アーキが必要
  - `weight(...)` 関数を **400 task 分手書き** するのは現実的でなく、自動探索 or LLM 生成が必須
  - `kernel_size=3` 固定。大域 pattern matching や 30x30 全体のパターン認識は対象外

## 出典

- Kernel URL: [https://www.kaggle.com/code/mmoffitt/the-2026-neurogolf-championship](https://www.kaggle.com/code/mmoffitt/the-2026-neurogolf-championship)
- このディレクトリの `kernel-metadata.json` 参照
