# neurogolf-2026 — 第一原理に基づく数学的解析 (dense)

> Worker C (W6 first-principles-deriver) + Worker D (W5 domain-knowledge-synthesizer)、2026-05-10 編集。
> 一次情報源: < engine source / metric definition / 公式 docs URL >

<!--
agent comp (rl) の場合:
  - engine source (`.venv/lib/python3.11/site-packages/kaggle_environments/envs/<slug>/<slug>.py`) を W6 が読み下す
  - 物理定数・state/observation 形式・reward 計算・action validation を closed form で整理
tabular / vision / nlp / timeseries の場合:
  - 評価指標の数式 (例: AUC, RMSE, F1, custom metric) を W6 が math 表記で書き下す
  - data spec (column dtype, target dtype, sample size, time span) を整理
  - W5 が domain SOTA / 教科書知識を並走で補う

すべての公式・定数に file:line または公式 docs URL を必ず付ける。
-->

---

## 0. 表記法・規約・特異点

| 記号 | 意味 | 出典 |
|---|---|---|
| | | |

**ストレージ上の癖 / 落とし穴** (= コードを書く前に必ず把握):

- < 例: 「planet record の slot 2 は X 座標、3 は Y 座標 (engine 内で sample → store で transposed)」 >
- < 例: 「target column は train.csv では int 型だが test sample submission は float 型」 >

---

## 1. 評価指標の数式定義

### 1.1 metric の closed form

<!-- public docs / engine source の数式をそのまま math 表記で。
     TeX 表記は `$ ... $` インライン、`$$ ... $$` ブロック。 -->

$$\text{metric}(y, \hat{y}) = \cdots$$

出典: < public docs URL > / `engine.py:NNN-NNN`

### 1.2 metric が依存する column / state

- < dependency 1 >
- < dependency 2 >

### 1.3 worst case / best case

- 理論最小値: < value > (条件: < ... >)
- 理論最大値: < value > (条件: < ... >)
- starter notebook の score: < value >

---

## 2. ゲーム / データの不変条件 (= 違反すると submission が silent fail / disqualify する)

<!-- agent comp:
     - action validation rule (forbidden moves が silent drop されるか)
     - timeout / step limit
     - resource constraint (ship max, RAM, etc.)
     tabular / nlp / vision:
     - submission file format
     - submission row count (= test set 行数と一致)
     - probability の合計が 1 になる必要があるか
     - leak の avoidance ルール -->

| 条件 | 違反時の挙動 | 出典 |
|---|---|---|
| | | |

---

## 3. 主要 quantity の閉形式 / 公式

<!-- agent comp 例:
     - fleet 速度公式: $v(N) = 1 + (v_{max}-1)(\ln N / \ln 1000)^{1.5}$
     - lead-shot 角度: $\theta = \arctan(\cdots)$
     - forbidden cone half-angle: $\alpha = \arcsin(R_\odot / D)$
     tabular / timeseries 例:
     - target encoding の bias correction
     - fold split の予測有効性
     - feature importance の信頼区間 -->

### 3.1 < quantity 1 >

$$\text{...} = \cdots$$

| < input > | < output > |
|---|---|
| | |

出典: < file:line >

### 3.2 < quantity 2 >

(同形式)

---

## 4. Domain 知識 (W5 出力)

> 本節は ARC-AGI (= Abstraction and Reasoning Corpus, AGI benchmark) と「最小 ONNX NN による per-task 解」という neurogolf-2026 固有の交点を整理する。一次情報を全て出典 URL 付きで列挙する。

### 4.1 当該 domain の最新 SOTA (= ARC + NN 圧縮 の交点)

#### 4.1.a ARC / program synthesis 系 (= 「何で解くか」の最新形)

- **Tiny Recursive Model (TRM, Jolicoeur-Martineau et al., 2025, Samsung SAIT Montreal)** ([arXiv:2510.04871](https://arxiv.org/abs/2510.04871)): 2 層 / 7M params の単一小型 NN を recursive な answer + latent state refinement で学習し ARC-AGI-1 で 45%、ARC-AGI-2 で 8% を達成。HRM (27M) を 4 倍小さい params で上回る。**neurogolf-2026 cost = ln(params + bytes)** を考えると 7M でも依然 大きいが、 「2 層 + 反復 refinement」という architecture は本コンペの cost 最小化方針と直接整合する (= 同じ重みを多層に流用すれば params 線形増加を抑制)。
- **ARChitects (ARC Prize 2024 1st place, 53.5% private eval)** ([ARC Prize 2024 Technical Report](https://arxiv.org/abs/2412.04604), [blog](https://arcprize.org/blog/arc-prize-2024-winners-technical-report)): Test-Time Training (TTT) + LLM fine-tune で per-task に重みを更新。2025 版は 2D-aware masked-diffusion + recursive self-refinement に発展 ([ARC Prize 2025 results](https://arcprize.org/blog/arc-prize-2025-results-analysis))。**TTT パラダイムは neurogolf-2026 の「per-task に独立した NN を提出する」設計と本質的に同じ**(= 各 task の入出力 example で「重みそのものをプログラムとして合成する」)。
- **MindsAI (TTT pioneer, 55.5% private eval 2024)** ([ARC Prize 2024 report](https://arcprize.org/media/arc-prize-2024-technical-report.pdf)): TTFT + augmentation ensemble + tokenizer dropout。**augmentation (回転・反転・色置換) で見かけ training data を 8-48 倍に増やす**手法は、ARC が input/output 3 例しか与えない制約下での over-fit 抑制に必須。
- **SOAR (ARC Prize 2025 2nd, 52% on ARC-AGI-1 open-source)** ([ARC Prize 2025 results](https://arcprize.org/blog/arc-prize-2025-results-analysis)): 自己改善型 evolutionary program synthesis。LLM を自分の探索 trace で fine-tune してブートストラップ。 **DSL を人手で書かずに winning ticket を見つける**思想。
- **DreamCoder (Ellis et al., 2021)** ([arXiv:2006.08381](https://arxiv.org/abs/2006.08381)): wake-sleep Bayesian program learning で「DSL ライブラリ + neural search policy」を共進化。ARC を含む 8 ドメインで実証。**neurogolf-2026 で 400 task を別々に解く際、共通 sub-routine を抽出して params を圧縮する考え方**の理論的支柱。
- **Hodel DSL (re-arc)** ([arXiv:2412.04604 §3.2 で言及](https://arxiv.org/html/2412.04604v2)): Michael Hodel が ARC 専用に整備した DSL で program search の効率を大幅向上。**ARC タスクの解は数 op の DSL プログラムで書ける**実証であり、最小 NN がこの DSL 級の表現力を持てば params は劇的に小さくできる。
- **Chollet ARC-AGI-2 (2025)** ([arXiv:2505.11831](https://arxiv.org/abs/2505.11831)): ARC-AGI-1 が LLM brute-force でほぼ飽和したため新版 (ARC-AGI-2) 投入。 「test-time compute による program-in-weights 合成」が共通トレンドと総括。

#### 4.1.b NN 圧縮 / sparsity 系 (= 「どう小さくするか」の最新形)

- **Lottery Ticket Hypothesis (Frankle & Carbin, ICLR 2019)** ([arXiv:1803.03635](https://arxiv.org/abs/1803.03635)): dense network 内には初期化を保ったまま 90%+ の重みを刈っても元精度に到達できる sparse subnet が存在する。**neurogolf-2026 では「per-task に当たり券を引き当てる」プロセスが直接 cost 削減に寄与**。
- **Wanda (Sun et al., ICLR 2024)** ([blog refs](https://www.meta-intelligence.tech/en/insight-pruning)): weight magnitude × input activation magnitude で重要度を評価。SparseGPT より 300x 高速、50% sparsity で perplexity が magnitude pruning baseline の半分以下。**1-shot で sparse subnet を得る**手法は per-task NN を量産する本コンペで再学習コストを劇的に下げる。
- **Minitron (Muralidharan et al., NeurIPS 2024, NVIDIA)** ([arXiv:2407.14679](https://arxiv.org/abs/2407.14679)): 構造化 pruning + knowledge distillation で 15B → 8B / 4B を 1/40 トークンで再学習。**structured pruning + distillation は ONNX export 時に dense 演算で残せる** (= unstructured sparsity と違い ONNX op 制約に優しい) ため neurogolf-2026 と相性が良い。
- **Deep Compression (Han et al., ICLR 2016)** ([arXiv:1510.00149](https://arxiv.org/abs/1510.00149)): pruning + quantization + Huffman coding で 35-49x 圧縮を無損失達成した古典。**neurogolf cost = params + memory bytes** という定義は Deep Compression の枠組みそのもの。
- **INT8 / INT4 quantization for ONNX** ([ONNX Runtime quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), [Intel Neural Compressor](https://github.com/intel/neural-compressor)): float32 → int8 で重み bytes が 4 倍削減。**neurogolf-2026 の cost が memory bytes を直接含むため、quantization は最も直接的な勝ち筋**。
- **Knowledge Distillation (Hinton et al., 2015)** ([arXiv:1503.02531](https://arxiv.org/abs/1503.02531)): 大教師の soft target を小生徒に転写。**TTT で得た重い per-task model を ONNX で提出可能な小型 student に蒸留する**フローが neurogolf 提出パイプラインの中核になる。

### 4.2 教科書 / standard reference

- **Chollet, "On the Measure of Intelligence" (2019)** ([arXiv:1911.01547](https://arxiv.org/abs/1911.01547)): 知能を「skill-acquisition efficiency on unknown tasks」と定義し、ARC を「人間 prior に近い最小限の prior だけで」解く benchmark として提案。**neurogolf-2026 の「task ごとに独立した NN を最小コストで」という設計は、 「task 解は単独でなくその task を解く能力を獲得する効率」を測るという ARC 本来の哲学と一致**。
- **Russell & Norvig, "Artificial Intelligence: A Modern Approach" (4th ed., 2020), §19 (Inductive Learning)**: ARC は inductive program synthesis (= 入出力例から program を induce) の典型例。Occam's razor (= 最短説明仮説) は cost = ln(params+bytes) と数式的に同型 (= MDL 原理)。
- **Murphy, "Probabilistic Machine Learning: An Introduction" (2022), §5 (MDL / Bayesian Occam) & §13 (Sparsity)**: 最小記述長原理と sparse priors の関係を数式で展開。**neurogolf-2026 score = max(1, 25 - ln(cost)) は Solomonoff prior の対数項そのもの**であり、 「task に対する最短プログラム」を NN 重みで近似する問題と読み替えられる。

### 4.3 該当 lib / framework の docs から拾った非自明な仕様

- **ONNX `Initializer` vs `Constant` op の cost 影響** ([ONNX IR spec](https://onnx.ai/onnx/repo-docs/IR.html), [Constant op spec](https://onnx.ai/onnx/operators/onnx__Constant.html)): graph の重み tensor は (a) `Initializer` フィールドに置く、(b) `Constant` op の `value` 属性に置く、の 2 通り。両者ともファイルサイズに同等寄与するが、**`Initializer` は外部ファイル (`external_data`) に逃せる**ため 1.44 MB / file 制約をすり抜ける誘惑が生じる。コンペ側 `neurogolf_utils.py` がこの抜け道を許容するかは要検証 (= W6 領域)。
- **ONNX shape inference は static shape 必須** ([ONNX Shape Inference docs](https://onnx.ai/onnx/repo-docs/ShapeInference.html), [docs/ShapeInference.md](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)): `TensorShapeProto` は compile-time に解決可能でなければならず、**動的 dim を含むモデルは shape 推論が落ちて runtime エラー**。 30×30 grid という固定形状で input/output が決まる neurogolf-2026 ではこれは制約でなく利点 (= 全 op を static にできる)。
- **禁止 op (Loop / Scan / NonZero / Unique / Script / Function) の代替** ([ONNX Operators 一覧](https://onnx.ai/onnx/operators/)): これらは「動的 trip count」「動的 output shape」「sub-graph 実行」を持つ op で、shape inference が不能になるか実質 program synthesis 的な制御フローを許す。**代替として `Where` / `MatMul` / `Conv` / `Gather` / `Reshape` の組み合わせで dataflow を表現**。 特に「条件分岐」は `Where` (= mask × A + (1-mask) × B) で書く。
- **PyTorch `torch.onnx.export(..., dynamo=True)` が 2026 推奨** ([PyTorch ONNX export tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html), [torch.onnx docs](https://docs.pytorch.org/docs/stable/onnx.html)): PyTorch 2.5 以降は `torch.export` + Torch FX ベースの dynamo exporter が default。 `model.eval()` を必ず呼ぶ。 **legacy `torch.onnx.export` (TorchScript ベース) の方がエッジケースで動く op 範囲は広い**ため、dynamo で禁止 op が混入したら legacy にフォールバックして比較する 2 段構えが安全。
- **ONNX Runtime profiler が memory footprint を測る source** ([ONNX Runtime quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), [Profiler API](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)): comp の `neurogolf_utils.py` が profiler 経由で peak working memory bytes を取る前提だとすれば、**activation tensor の最大 footprint を「層をまたぐ tensor の最大瞬間 bytes」で見積もる**必要があり、weight bytes だけ削っても activation が大きい設計 (= wide-shallow) は cost に効かない。深く狭い (deep-narrow) かつ residual で activation を捨てる設計が有利 (= TRM の 2-layer recursive がここでも合理的)。
- **ONNX Simplifier による redundant op 除去** ([onnx-simplifier README](https://github.com/daquexian/onnx-simplifier)): export 直後の graph には constant folding 余地が多い。**提出前に simplify を必ず通すこと**で params + ops を 10-30% 削れる事例多数。

---

## 5. submission / agent template の制約

<!-- file size limit, memory limit, time limit per row/turn, internet 可否 -->

| 項目 | 制約 | 出典 |
|---|---|---|
| File size | | |
| Memory | | |
| Wall time per row/turn | | |
| Internet during inference | | |
| External data 可否 | | |

---

## 6. これらの第一原理から導かれる「やってはいけないこと」

- < anti-pattern 1 + 数式根拠 >
- < anti-pattern 2 >

---

## 7. これらの第一原理から導かれる「優位性の source」

- < advantage 1 + 数式根拠 >
- < advantage 2 >
