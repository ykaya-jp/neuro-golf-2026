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
| INPUT_SHAPE = [1, 10, 30, 30] | 入力 tensor 形状: batch=1, channels=10 (one-hot), height=30, width=30 | neurogolf_utils.py:86, 105 |
| OUTPUT_SHAPE = [1, 10, 30, 30] | 出力 tensor 形状 (入力と同じ) | neurogolf_utils.py:105, verify_network:444 |
| PARAM_COUNT | 重み initializer / Constant node の要素数合計 + scalar parameter 数 | neurogolf_utils.py:291-314 (calculate_params) |
| MEMORY_BYTES | 各 tensor の最大サイズ総和 (bytes、ONNX Runtime Profiler より算出) | neurogolf_utils.py:189-249 (calculate_memory) |
| COST = PARAM_COUNT + MEMORY_BYTES | 評価対象の総コスト | neurogolf_utils.py:454 |
| SCORE = max(1, 25 - ln(COST)) | ゲーム score 1 task あたり | neurogolf_utils.py:454 |

**ストレージ上の癖 / 落とし穴**:

- **tensor / initializer 名衝突禁止** (2026-05-06): ONNX graph.input / graph.output の名前と initializer の名前が被ると reject。例: input 名に "input"、output 名に "output" を避ける (neurogolf_utils.py:192-196)。
- **"kernel_time" 文字列禁止** (2026-05-06): node output 名に "kernel_time" を含むと validator が reject。ONNX Runtime Profiler が自動的に "_kernel_time" suffix をつけるため、node name を output[0] に統一する際に注意 (neurogolf_utils.py:240, 430)。
- **One-hot encoding**: 各 grid cell [r, c] は色インデックス (0-9) で表現。tensor では [batch=0, color_index, r, c] = 1.0、その他は 0.0。border 外は全て 0 (zero-hot) (neurogolf_utils.py:262-272)。
- **Grid pad**: 入力 grid が 30x30 未満でも BATCH=1, CHANNELS=10, HEIGHT=30, WIDTH=30 に zero-pad される。test 実行時に grid > 30x30 は ignored される (neurogolf_utils.py:268)。
- **Shape inference strict mode**: ONNX shape inference は strict_mode=True で実行。すべての tensor dimension が静的 (no symbolic dim、no sequence_type) でないと shape 推論失敗 → memory calculation None → error (neurogolf_utils.py:191)。

## 1. 評価指標の数式定義

### 1.1 metric の closed form

$$\text{score}_t = \max(1, 25 - \ln(\text{cost}_t))$$

ここで、

$$\text{cost}_t = \text{params}(\text{NN}_t) + \text{memory\_bytes}(\text{NN}_t)$$

$t \in \{1, 2, \ldots, 400\}$ = task ID。

出典: neurogolf_utils.py:454、`verify_network()` 内の点数計算式。

### 1.2 metric が依存する量

- **params(NN)**: ONNX model.graph.initializer (weight tensor) と graph.node[op_type='Constant'] の attribute で定義される定数の要素数合計 (neurogolf_utils.py:291-314)。2026-05-06 より scalar parameter (Constant node の value_ints, value_floats 等) も 1 param ずつ count される (neurogolf_utils.py:308-313)。
- **memory_bytes(NN)**: ONNX Runtime Profiler (enable_profiling=True) が出力する JSON trace から、各 tensor の最大メモリ占有量を抽出し合計 (neurogolf_utils.py:189-249)。float32 では 4 bytes/param、int8 では 1 bytes/param。
- **ln()** = 自然対数。
- **max(1, ...)** = 最低スコア 1 点 (cost が exp(24) ≈ 26B 以上でも必ず 1 点獲得)。

### 1.3 worst case / best case

- **理論最小値**: 1 点 (= cost が大きい limit)。例: 全 400 task で score 1 なら total 400。
- **理論最大値**: 25 点/task (= cost → 0、ただし params ≥ 1 なので不可能に近い)。例: cost = 1 なら score = 25 - ln(1) = 25。
- **実務的最大値**: cost ≈ 1 程度が限界 (bias 1 + 最小限の weight)。
- **current LB top**: 7290 / 10000 ≈ 73%。これは平均 18.23 points/task ⟺ 平均 cost ≈ 876 (neurogolf_utils.py の計算から逆算)。
- **starter benchmark**: 1-layer 3x3 conv (weight only、900 params + float32 memory 3600 bytes = cost 4500) なら score ≈ 16.59/task → 6635/10000 = 66%。

## 2. ゲーム / データの不変条件

| 条件 | 違反時の挙動 | 出典 |
|---|---|---|
| ONNX file size ≤ 1.44 MB | submission 全体 reject | neurogolf_utils.py:104, check_network:256 |
| 禁止 operator: LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, FUNCTION, COMPRESS | validator 内で check (Kaggle engine が reject) | neurogolf_utils.py:103 (note: COMPRESS は 2026-04-30 追加) |
| 単一入力・単一出力 graph のみ | multi-input / multi-output は None return (2026-05-06) | neurogolf_utils.py:192 |
| Static shape 必須 (dynamic dim 禁止) | shape_inference fail → memory calc None → error | neurogolf_utils.py:222-228 (dim_param は rejected) |
| Sequence type / nonpositive tensor dim 禁止 | (2026-05-04) Sequence は rejected、dim ≤ 0 は rejected | neurogolf_utils.py:219, 227 |
| tensor 名に "kernel_time" を含まない | validator reject (2026-05-06) | neurogolf_utils.py:430 |
| initializer ∩ (input ∪ output) = ∅ (名衝突禁止) | (2026-05-06) memory calc None | neurogolf_utils.py:195-196 |
| Custom domain / functions / subgraph 禁止 | (2026-04-30) None return | neurogolf_utils.py:197-206 |
| Node negative params / memory 禁止 | (2026-04-28) error flag set | neurogolf_utils.py:447-448 |
| Grid ≤ 30×30 (test data) | > 30×30 は ignored in benchmark | neurogolf_utils.py:268 |
| Output: 30×30 one-hot encoding | border 外は zero-hot (color 10) | neurogolf_utils.py:275-288, convert_from_numpy() |

## 3. 主要 quantity の閉形式

### 3.1 1-layer convolution の params 計算

$$\text{params}_{\text{conv}} = C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w + (\text{bias=optional})$$

例: $C_{\text{out}} = 10, C_{\text{in}} = 10, K = 3 \times 3$ なら

$$\text{params}_{\text{conv}} = 10 \times 10 \times 3 \times 3 = 900 \text{ (weight のみ)}$$

bias 含む場合は +10、計 910。

出典: neurogolf_utils.py:401-418 (single_layer_conv2d_network)。kernel_size=3 で kernel_offsets = [-1, 0, 1]、weights shape = [10, 10, 3, 3]。

### 3.2 Memory footprint (float32)

$$\text{memory\_bytes}_{\text{float32}} = \text{params} \times 4 \text{ bytes/element}$$

例: params = 900 なら memory = 3600 bytes。

Quantized int8:

$$\text{memory\_bytes}_{\text{int8}} = \text{params} \times 1 \text{ byte/element}$$

例: 同じ 900 params で memory = 900 bytes (4× 削減)。

### 3.3 Full cost (params + memory) と score

例: 1-layer 3×3 conv, C_out=C_in=10, float32:
- params = 900
- memory = 3600
- cost = 4500
- score = max(1, 25 - ln(4500)) = max(1, 25 - 8.412) = **16.588**

400 task 全て cost 4500 なら: $400 \times 16.588 = 6635 / 10000 = 66.35\%$

int8 量子化で cost = 1800 なら score = max(1, 25 - 7.495) = **17.505** → 400 × 17.505 = 7002 / 10000 (+367, +5.5%)。

出典: neurogolf_utils.py:454、verify_network()。

### 3.4 Score range と cost の関数関係

$$\ln(\text{cost}) = 25 - \text{score}$$

score = 20 → cost ≈ 148
score = 18 → cost ≈ 876 (current LB top)
score = 16 → cost ≈ 4500 (1-layer baseline)

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

## 5. submission / network 制約

| 項目 | 制約 | 出典 |
|---|---|---|
| ONNX file size | ≤ 1.44 MB / file | neurogolf_utils.py:104 (_FILESIZE_LIMIT_IN_BYTES) |
| Submission archive | task001.onnx ~ task400.onnx を zip して提出 | neurogolf_utils.py:verify_network() docstring (implied) |
| ONNX opset | 10 固定 (ONNX IR version 10) | neurogolf_utils.py:106 (_IR_VERSION, _OPSET_IMPORTS) |
| Input tensor 名 | "input" (固定) | neurogolf_utils.py:331, 410 |
| Output tensor 名 | "output" (固定) | neurogolf_utils.py:331, 411 |
| Input shape | [1, 10, 30, 30] (static) | neurogolf_utils.py:86, 105, 264 |
| Output shape | [1, 10, 30, 30] (static) | neurogolf_utils.py:105 |
| Inference time limit | 記載なし (kaggle notebook は ~ 30 min timeout) | (TBD: discussion 確認) |
| External data | Kaggle 標準規約 (不可) | (standard Kaggle rules) |
| Internet during inference | 不可 | (standard Kaggle rules) |
| Constant folding | 有効 (2026-04-28 より params に count される) | neurogolf_utils.py:291-307 (calculate_params, Constant node handling) |

## 6. これらの第一原理から導かれる「やってはいけないこと」

- **すべての task に同じ汎用 NN を使う**: cost = params (shared across 400 task) + memory_bytes × 400 になり、1 network で 400 copy のメモリコストを払う。任意の 1 network で cost ≥ 400M → score < 1 (ln(400M) > 25)。task 別最適化必須。
- **大型 dense layer (> 1000 output channels)**: params > 1M → ln(1M) = 13.8 → score < 12。grid 30×30 で spatial structure を失う dense は非効率。
- **dynamic shape / symbolic dimension**: shape_inference strict_mode=True で fail → memory = None → error。static shape 必須 (ONNX standard ではなく当コンペ制約)。
- **tensor 名に "kernel_time" を含める**: Profiler が自動的に "_kernel_time" suffix を追加し、validator がこれを reject。node.name = node.output[0] の pattern で対応 (neurogolf_utils.py:429)。
- **initializer と input/output の名衝突**: validator check (neurogolf_utils.py:195-196) → None return → rejected。"W", "bias" など無難な命名が必須。
- **unnecessary bias**: bias = C_out params、数式上 NN 性能に寄与しなければ cost waste。task 別に必要判定する。
- **float32 で保存し続ける**: int8 量子化で memory 4× 削減可能。cost が ln() に logarithmic なため、削減効果は +0.9 point/task × 400 = +360 点。
- **Loop / Scan operator 使用**: 禁止リスト (neurogolf_utils.py:103)。dynamic loop や sequential processing は不可能。

## 7. これらの第一原理から導かれる「優位性の source」

- **task 別最適 architecture 選択**: 例えば tiling task は 1-layer 3×3 conv で sufficient (params = 900)、pure color-swap なら lookup table 風 1×1 conv (params = 100)。architecture を task category に応じて分岐させれば avg cost を 900 → 500 に低下させ、avg score を 18.2 → 19.6 に向上 (+560 点)。
- **重み hand-craft / training skip**: supervised training を skip、ARC-AGI task の logic を手 analysis / LLM で reverse-engineering し、weight を directly construct。overfit zero + cost 最小化。例: reflect / rotate task なら weight matrix を hard-code。
- **int8 / sparse quantization**: float32 → int8 で memory 4× 削減。sparse weight (90% zero) さらに活用すれば memory_bytes / params → 0。cost reduction log-sensitive → +5-10% total score gain。
- **task category clustering + template library**: ARC-explanations.json, arc-primitives.json から task category を抽出 (reflect, rotate, color-map, etc.)。category ごとに fixed NN template を設計し、task parameter tune only。design reuse + cost amortization。
- **arc_explanations.json の LLM 自動化**: task 自然言語説明 → ONNX graph 自動生成。rule-based pattern matching で logic → graph conversion。manual design overhead reduction。
- **Initializer vs. Constant node trade-off**: initializer (initializer count toward params) vs. Constant node (attribute count で params)。sparse pattern によって最適化分岐。例: sparse tensor なら sparse_initializer (memory_bytes 削減)。
- **float16 fallback**: float32 より float16 (2 bytes/param) を試験。accuracy impact verify 後、memory cost 50% 削減可能。ONNX Cast operator で type conversion (cost: 0 params)。
