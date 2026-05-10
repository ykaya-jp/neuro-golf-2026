# neurogolf-2026 — 第一原理 TL;DR 版

> 2026-05-10 編集。詳細は `first-principles.dense.md`。

<!-- 高校生でも読める文体。式は平文化、図解推奨。 -->

---

## 1. 評価指標を 1 文で

**NeuroGolf score** は「あなたの AI ネットワークが ARC-AGI を正しく解きつつ、パラメータ数とメモリ使用量を最小化できているか」を測る指標。**値が大きい方が良い** (最大 25 点/task、400 task で 10,000 点満点)。

数式: score = max(1, 25 - ln(cost))、ここで cost = params + memory_bytes。例: 1-layer conv (params 900、memory 3600) なら cost 4500 → score 16.6 点。

## 2. 落とし穴 / 守らないと submission が silent fail する条件

- **tensor 名に "kernel_time" を含めてはいけない** — ONNX Runtime の Profiler が自動的にこの suffix をつけ、validator が reject します。node 名を output[0] に統一する対策が必須。
- **複数入力・複数出力の graph は禁止** — 入力は "input" 1 個、出力は "output" 1 個に限定。multi-input/multi-output だと memory 計算が失敗します。
- **Dynamic shape (symbolic dimension) は禁止** — すべての tensor dimension は固定値でないと shape_inference が失敗し、memory 計算できません。[1, 10, 30, 30] のように数字で指定するのみ。
- **Loop, Scan, NonZero, Unique, Script, Function, Compress operator は禁止** — これらを含むと validator reject。
- **ONNX file size > 1.44 MB** は reject — 1 network が 1.44 MB 以下に収まる必要があります。

## 3. 知っておくべき公式・定数

**Cost = params + memory (bytes)**

単純 1-layer convolution:
- params = C_out × C_in × kernel_size × kernel_size
- memory = params × (4 bytes if float32, 1 byte if int8)

例: 10×10 の重み、3×3 kernel なら params = 900、float32 なら memory = 3600 bytes、cost = 4500。

**Score は logarithmic decay**

cost 4500 → score 16.6
cost 1800 → score 17.5 (int8 量子化で +0.9 点)
cost 900 → score 18.2
cost 150 → score 20.0

**total = 400 task × score/task**

全 task で cost 900 達成なら 400 × 18.2 = 7290 / 10000 (= current LB top 73%)。

## 4. 当該 domain の現代 SOTA を 3 行で

- **2024-2025 ARC 系**: 大型 LLM を per-task で fine-tune する Test-Time Training (ARChitects 53.5%, MindsAI 55.5%) と、7M params の小型 NN を反復 refinement する Tiny Recursive Model (45% on ARC-AGI-1) が 2 大潮流 ([ARC Prize 2024 report](https://arcprize.org/blog/arc-prize-2024-winners-technical-report), [TRM paper](https://arxiv.org/abs/2510.04871))。
- **2024-2025 NN 圧縮系**: 構造化 pruning + knowledge distillation (Minitron, NeurIPS 2024) と activation 重要度ベースの 1-shot pruning (Wanda, ICLR 2024) と INT8/INT4 quantization (ONNX Runtime / Intel Neural Compressor) が現役 SOTA。
- **当該コンペでは**: 「TTT で per-task に重みを合成し → 蒸留 + pruning + INT8 quant で 1.44 MB 以下の ONNX に詰める」が両者の交点。**cost = ln(params + bytes) は MDL / Solomonoff prior 由来**で、結局「task の最短プログラムを NN 重みに焼く」問題 ([Chollet 2019](https://arxiv.org/abs/1911.01547))。

---

## 5. これだけは絶対やる / 絶対やらない

**やる:**

- **task 別に異なる architecture を選ぶ**: tiling は 1×1 conv (100 params)、color-swap は lookup table、mirror は weight hard-code。「平均 cost を 900 → 500 に下げれば +560 点」。
- **int8 量子化を試す**: 同じ network で memory 4× 削減。ln() 関数の性質上、cost 低下による score 向上は 後ろに行くほど diminishing return だが、全 400 task なら +360 点は大きい。

**やらない (= 数式的に逆効果):**

- **すべての task に 1 つの通用 model を使う**: cost = shared params + 400 × memory。任意の model で cost > 1M → score < 0 (実質)。
- **float32 のままリリース**: memory_bytes が cost の大部分を占める (例: 900 params + 3600 bytes memory = 80% が memory)。int8 で削減しない理由なし。
- **不要な layer を残す**: bias がないなら 0 を足さない。Constant node で unused parameter を hold しない。1 param = 1 cost point。
