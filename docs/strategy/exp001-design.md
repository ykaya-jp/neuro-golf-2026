# exp001 Design — Phase 1 Synthesis from 6 Parallel Research Tracks (NeuroGolf 2026)

> **Status**: Phase 2-3 synthesis (= research summary + 構造原理が異なる 3+ 案).
> **判断は開発者**。中央 (= main agent) は推奨案を出さない (~/.claude/CLAUDE.md 中立指示原則 / 去好去悪).
> **実装は次プラン** (`/plan kaggle-neurogolf-2026-baseline`) で。
> **Date**: 2026-05-10

---

## 1. Context

**NeuroGolf 2026** (IJCAI-ECAI 2026 Competitions Track) — ARC-AGI v1 公開 train 400 task を **解きつつ最小 ONNX neural network** を構築するコンペ。

- 賞金: $50,000 (1st $12K + 2nd $10K + 3rd $10K + Top Student $8K + **Longest Leader $10K**)
- Deadline: 2026-07-15 23:59 UTC
- 評価: per task で `score_t = max(1, 25 - ln(cost_t))`, `cost_t = params + memory_bytes`、合計 400 task で理論最大 10,000

我々の現状: 未提出。LB top: yiheng "5.5 or 4.7?" 7290.56 (avg 18.2 / task = ln(cost) ≈ 6.8 = cost ≈ 900)。Top 10 まで 6500+。

**critical 制約** (= W6 + W1 一次資料調査済):

- ONNX 1.44 MB / file、static shape 必須
- 禁止 operator: Loop / Scan / NonZero / Unique / Script / Function (+ 2026-04-30 で Compress 追加)
- 2026-05-04 で MAC は metric から除外、cost = `params + cumulative_memory_bytes`
- 2026-05-06 で scalar parameter unit cost 化、`kernel_time` 文字列 ban、Multi-input/output graph ban
- **scorer-poison op 回避リスト** (W1 finding): `ArgMin / TopK / Where+uint8 / Compress` 等は spec-legal でも scorer crash → 全 task 0 点
- **5/15 前後の metric 再改訂予想** (W1) → exploit-free baseline と実験の 2 系統 submission 運用

---

## 2. Phase 1〜2 統合所見

### 2.1 W3 (past-comps) — 戦略的 prior

- **構造類似度 maximum 2 件**:
  - **ARC Prize 2024** (軸 A: ARC 解法) — ARChitects 53.5%、TTT + LoRA + transduction/induction hybrid ([source](https://arxiv.org/html/2412.04604v2))
  - **MicroNet Challenge 2019** (軸 B: NN 圧縮) — MIT-Han Lab 1.8M params、pruning + 8-bit quant + KD で 35-49× 圧縮 ([source](https://github.com/mit-han-lab/neurips-micronet))
- **構造類似度 high 5 件**: Abstraction & Reasoning Challenge 2019 (DSL 142 prim), CompressARC 2025 (76K params で 20-34%), TRM 2025 (7M params で 45%), SOAR 2025 (LLM evolutionary 52%), ARC Prize 2025 NVARC (synthetic + TTT + TRM)
- **歴史的 winning paradigm**:
  - 2019 ARC: pure heuristic DSL (icecuber)
  - 2024 ARC Prize: TTT + LLM hybrid (ARChitects 53.5%, MindsAI 55.5%)
  - 2025: refinement loop が universal、tiny networks (76K-7M params) が大型 LLM を outperform、synthetic augmentation 必須
- **Hybrid 推奨 recipe**: program synthesis → NN compile → 8-bit quant + pruning ([source](docs/research/past-comps.dense.md:N.3))

### 2.2 W2 (公開 kernels) — 戦術ヒント

15 kernel pull + INDEX.md 完了 (44 unique のうち、優先 15 件詳細化 / 残り 29 件 metadata 列挙):

- **重要発見**: **2026-05-06 metric 改訂後、LB top は「他人の解 zip を集めて per-task 最安を選ぶ blending」が独占** (出典: `docs/research/public_kernels/neurogolf-2026-may-8-updated/content.md`)
- **公式 starter (mmoffitt, 185 vote)** = `single_layer_conv2d_network(weight_fn, kernel_size=3)` で 1 層 Conv2D 重みハードコード style
- **真の per-task solver** (= Tiny-ONNX, Nano-Engine 系) は少数派 (4-5 件)
- **LLM-driven program synthesis** (= logic-decoder 系) は準備段階に留まる
- **横断的に見える技法 3 つ**:
  1. **1-2 layer Conv2D で 30x30 grid を直接処理** (= 主催者 starter pattern、複数 kernel で踏襲)
  2. **公開 source zip の per-task argmin blending** (= Top LB 戦略の現状、5/15 改訂で死ぬ可能性)
  3. **arc_primitives.json の category 別 template 化** (= karnakbaevarthur 著者、host dataset 提供者の同人 kernel)

### 2.3 W6 (first-principles) — 数理的 invariant

- **scoring 公式**: `score_t = max(1, 25 - ln(cost_t))`, `cost_t = params + memory_bytes` (出典: `docs/strategy/first-principles.dense.md`, `data/raw/neurogolf_utils/neurogolf_utils.py`)
- **baseline (1-layer 3×3 conv, float32, 900 params + 3600 byte = cost 4500)** → **score 16.59** per task = 6,635 / 10,000 total
- **INT8 量子化** (cost 1800) → score 17.50 per task = **+367 点 (= +5.5%)**
- **task 別 architecture 選択** (cost 500-1500) → +160-560 点 (= 上位陣 LB の avg 18.2 への接続)
- **Constant scalar も 1 param ずつ count** (2026-05-06)、`kernel_time` 名禁止、Multi-input/output 禁止、static shape 必須
- **anti-pattern**: shared model / dense > 1000 ch / dynamic shape / float32 保持 / scorer-poison op
- **advantage**: task 別 1-2 layer conv hand-tune、INT8 quant、template library、LLM 自動化

### 2.4 W4 (host dataset) — 実測値

- comp data: **400 task × 平均 254 pair = 101,718 pair** の正解必須 (出典: `docs/research/host_datasets.md` §3.2)
- input/output grid: median 10×10、p90 20×20、max 70×75 / 42×38
- color 0 (clear/black) が **80% を占める** → channel sparse、conv weight も大半 zero
- arc_primitives.json: **400 task すべてに category + Estimated_Complexity (1-5) 付与**、Grid_Size_Changed が過半
- arc_explanations.json: **400 task すべてに自然言語 logic 説明** 付き (例: task001 = "9x9 output by tiling 3x3 input in 3x3 pattern" → ONNX 1 layer で書ける)
- arc-gen-100k: 各 task 262 pair の synthetic data、validation 用

### 2.5 W1 (discussion) — fresh insight

- **5/4 metric 改訂が現行**: cost = params + cumulative_memory_bytes (MACs 削除)、static shape 強制、Constant 計上、`onnx-tool 1.0.1` 固定
- **scorer-poison op 回避リスト**: ArgMin / TopK / Where+uint8 / Compress 等 → spec-legal だが scorer crash で全 task 0 点
- **5/15 前後の metric 再改訂予想** (現在 4 件の未公開 exploit が報告されている、host が継続 fix 明言)
- **Longest Leader $10K 賞は 5/6 00:00 UTC 起算リセット** (= deadline 2026-07-15 まで首位継続 で勝ち)
- **Kaggle Agent チームが top 5 にいる** (cdeotte 等)、LLM agent 戦略の prototype として注目

### 2.6 W5 (domain 知識) — SOTA & 教科書

- **TTT (Test-Time Training) パラダイムが NeuroGolf と本質的に同型** (= 「per-task に独立 NN ファイル提出」 = 「per-task に重みを fine-tune して提出」) ([source](https://arxiv.org/abs/2412.04604))
- **TRM (7M params, 45%) の 2-layer + iterative refinement architecture が deep-narrow で activation footprint も小さい** ([source](https://arxiv.org/abs/2510.04871))
- **両者の交点 = TTT で per-task 重み合成 → distillation + structured pruning + INT8 quant で 1.44 MB 以下に圧縮**
- ONNX `Initializer` external_data で 1.44 MB 制約をすり抜け可能性 (要 W6 検証)
- 禁止 op の代替は `Where` operator (mask × A + (1-mask) × B) で条件分岐表現
- `torch.onnx.export(..., dynamo=True)` が 2026 推奨だが legacy exporter の op 範囲が広い、2 段構え + ONNX Simplifier 必須

---

## 3. exp001 候補 — 構造原理が異なる 3+ 案

各候補は **異なるパラダイム**。同じ paradigm 内の param tuning は「バリエーション」なので除外。

---

### 候補 1: Per-task hardcoded 1-2 layer conv + INT8 quantize (= 主催者 starter ベース、symbolic minimalism)

**Provenance** (= 着想の出典):

- **公式 starter** (`mmoffitt/the-2026-neurogolf-championship`, 185 votes) — `single_layer_conv2d_network(weight_fn, kernel_size=3)` で 1 層 Conv2D 重みハードコード
- **MicroNet Challenge 2019** ([source](https://github.com/mit-han-lab/neurips-micronet)) — pruning + 8-bit quantization pipeline
- **icecuber DSL 2019** ([source](https://github.com/top-quarks/ARC-solution)) — handcrafted 142 primitive、symbolic baseline
- **W4 EDA 知見**: median grid 10×10、color 0 が 80%、Estimated_Complexity 1-3 が支配的 → 1-2 layer conv で十分な task が支配的

**Mechanism** (= 構造原理):

1. **arc_primitives.json + arc_explanations.json で 400 task を category 別 (Spatial / Object / Color / Pattern) に分類**
2. category × complexity 別に **template ONNX architecture を ~10 種** 用意 (例: identity / 3x3 conv / 5x5 conv / channel permute / spatial tile / crop / mirror)
3. 各 task の自然言語 logic を読み取り、template に合わせて **重みを手書き** (Python helper で `weight_fn` を書く → `single_layer_conv2d_network` 経由で ONNX export)
4. **INT8 quantization** (`QuantizeLinear` + `DequantizeLinear` ペア) で memory_bytes を 4× 削減
5. ONNX Simplifier (constant folding) で 10-30% 追加削減
6. local validator (`neurogolf_utils.score_network()`) で全 train+test+arc-gen 正解を確認してから submit

**Phase 1 知見の活用**:

- W2 公式 starter pattern を全 400 task に拡張、INDEX で見つけた 1-2 layer conv kernel (Tiny-ONNX, Nano-Engine) を参考
- W3 MicroNet 由来の量子化パイプライン
- W6 cost 公式: 1-layer 3×3 conv float32 = 4500 cost、INT8 = 1800 cost、target task-specific = 500-1500
- W4 EDA: 80% sparse の color 0 channel を活用、weight も大半 zero でさらに圧縮
- W1 scorer-poison list 回避

**Failure modes** (= 失敗シナリオ — Phase 4 critic に独立検証させる):

- **Hand-craft が 400 task は時間ボトルネック** (likelihood: high, impact: high) — 1 task あたり 30 分でも 400 × 30 = 200 時間
- **Estimated_Complexity 4-5 task で 1-2 layer 不足** (likelihood: medium, impact: medium) — 50-100 task 程度で fail 推定
- **private benchmark で過学習** (likelihood: medium, impact: high) — 公開 train で動いても private で fail、score 0
- **手書き重みの bug 検出が困難** (likelihood: high, impact: medium) — 400 task 各々 unit test 必要

**Disqualifying conditions** (= この案を「やめる」べき条件):

- 1 月で 100 task しか hand-craft 完了せず、自動化なしでは追いつかない
- private で 100 task 以上 fail (= overfitting 過大)
- 主催者が更に厳しい operator ban を入れて template が崩壊

**コスト & 期待**:

- 開発: **2-4 週** (template 設計 1 週 + 400 task 重み手書き 2-3 週)
- compute: CPU only、ローカル開発
- 期待 score: **5500-6500** (= 公式 starter 系の延長、avg 14-16 / task)
- リスク: top tier 7000+ には届かない、メダル圏外の可能性

---

### 候補 2: LLM-driven program synthesis → ONNX template compile (= SOAR + CompressARC + arc_explanations.json 駆動)

**Provenance**:

- **SOAR 2025** ([source](https://arxiv.org/abs/2507.14172)) — LLM evolutionary program synthesis + hindsight learning、ARC-AGI-1 で 52%
- **CompressARC 2025** ([source](https://github.com/iliao2345/CompressARC)) — MDL-based neural code golf、76K params で 20-34%
- **host_datasets**: `arc_explanations.json` (400 task 自然言語 logic 完備) + `arc_primitives.json` (category + complexity) — **LLM の primary input**
- **W2 logic-decoder 系列** (karnakbaevarthur, 73 votes) — LLM 駆動 ONNX 生成の prototype
- **Kaggle Agent team が LB top 5** (cdeotte 等、W1) — LLM agent 戦略が prototype 段階で実用へ

**Mechanism**:

1. **arc_explanations.json から 400 task の自然言語 logic を取り出し**、LLM (Claude Opus / GPT-5 / Codex) に input
2. LLM が **per-task に Python コード生成** (= `weight_fn(channel_out, channel_in, kernel_coord)` の if-then ロジック)
3. 生成コードを `neurogolf_utils.single_layer_conv2d_network(weight_fn, ...)` で ONNX export
4. local validator で正解判定、fail なら **error feedback を LLM に戻して refine** (= SOAR pattern)
5. 成功 task は INT8 quantize、ONNX Simplifier
6. fail task は category 別 template で fallback (候補 1 の手書き)

**Phase 1 知見の活用**:

- W3 SOAR pattern (52%) を ONNX 化に application
- W4 host_datasets 完備 (= 400 task 全 logic 自然言語済) で **LLM の前処理コスト ゼロ**
- W2 logic-decoder の prior art を model architecture の参考に
- W6 cost 公式: LLM が生成する weight が sparse なら CompressARC pattern (76K params) と同等
- W5 TTT パラダイム適用: per-task に独立 NN を offline 生成 = TTT の inference-only 化

**Failure modes**:

- **LLM が間違った logic を generate** (likelihood: high for complex tasks, impact: high) — Estimated_Complexity 4-5 で hallucination 多発予想
- **ONNX 制約違反** (likelihood: medium, impact: high) — LLM が scorer-poison op (Compress, TopK 等) を選ぶ
- **template 多様性 vs compile 困難** (likelihood: medium, impact: medium) — 400 task で 400 architecture 全部違うと管理困難
- **API cost 爆発** (likelihood: medium, impact: medium) — 400 task × refinement 5 round × ~50K token = 約 $200-500

**Disqualifying conditions**:

- arc_explanations.json の logic が ambiguous で LLM が解析不能 (= category Pattern_Recognition 系で発生予想)
- LLM が scorer-poison op を選び続け 50 task 以上で fail
- API cost が予算超過

**コスト & 期待**:

- 開発: **2-4 週** (LLM pipeline 1 週 + 400 task 自動生成 + refine 2-3 週)
- compute: API only、CPU 検証
- 期待 score: **6000-7000** (= LLM 駆動の 60-70% カバー × INT8 quant)
- リスク: medium — LLM の generalization に依存、private で degradation

---

### 候補 3: Public submission blending + per-task argmin (= 5/6 後 LB top 戦略の踏襲、metaplay)

**Provenance**:

- **W2 INDEX.md finding**: 「2026-05-06 metric 改訂後、LB top は『他人の解 zip を集めて per-task 最安を選ぶ blending』が独占」
- **公開 dataset 多数**: `konbu17/neurogolf-2026-blended-401-v117` (LB 5331+), `karnakbaevarthur/neurogolf-2026-task-transformation-library`, `magmacot/5700-neurogolf-logic-driven-ensemble`, `jonathanchan/ngc26-constraint-smart-logic-mix-blending` (115 votes), `needless090/neurogolf-onnx-v31` 等
- **W1 finding**: scorer の 5/15 改訂前なら blending は機能、5/15 後は exploit-free baseline 優先

**Mechanism**:

1. **公開 dataset (LB 5000+ 帯) を 5-10 件 DL** (= 既に 1 件は acquired)
2. 各 zip の `task001.onnx` ~ `task400.onnx` を unpack、per-task で **cost 計算**
3. 各 task で **min(cost) の onnx を選択** (= per-task argmin blending)
4. 選択した onnx を 1 つの submission.zip に統合
5. local validator で全 task 正解を確認
6. submission

**Phase 1 知見の活用**:

- W2 INDEX で「blending dominant」を確認、jonathanchan 等の中身を参考
- W6 cost 計算公式で per-task argmin
- W1 metric 履歴: 5/4 改訂後の現行 metric 上で blending は valid

**Failure modes**:

- **5/15 の metric 再改訂で全 blending が無効化** (likelihood: high (= W1 報告), impact: catastrophic) — 公開 zip の cost が改訂後計算で激増
- **private benchmark で他人解の overfitting** (likelihood: high, impact: high) — 他人の per-task 最安解は public で min だが private で fail 多発
- **independent solution 不在で IJCAI talk invitation 不利** (likelihood: medium, impact: medium) — 賞金は取れるが学術的 contribution ゼロ
- **主催者が blending submission を ban** (likelihood: low, impact: catastrophic) — host が制度的に禁止する可能性

**Disqualifying conditions**:

- 主催者 announcement で blending 禁止が明示される
- 5/15 metric 改訂で score が下落 (= 7000 → 4000 級になる可能性)
- private LB shake-up で ranking 崩壊

**コスト & 期待**:

- 開発: **1-2 週** (DL + argmin 自動化 + integration testing)
- compute: CPU only
- 期待 score: **5800-7000** (= 現状 LB の現行 metric 上の値、ただし 5/15 後で大幅減の risk)
- リスク: **catastrophic risk あり** (5/15 metric 改訂、private overfitting)

---

## 4. 評価軸 (中立、開発者判断用)

| 軸 | 候補 1 (Per-task hardcoded conv) | 候補 2 (LLM program synthesis) | 候補 3 (Public blending) |
|---|---|---|---|
| **実現可能性** | 高 (CPU only、確実) | 中 (LLM API 信頼性に依存) | 高 (DL → argmin だけ) |
| **開発期間** | 2-4 週 | 2-4 週 | 1-2 週 |
| **必要 compute** | CPU only | LLM API (cost ~$200-500) | CPU only |
| **期待 score** | 5500-6500 | 6000-7000 | 5800-7000 (現行 metric) |
| **メダル / Top 10 確率** | 中 | 中-高 | 中 (5/15 で激減 risk) |
| **Phase 1 トラック活用度** | W4 全 + W6 全 + W3 一部 + W2 一部 | W3 全 + W4 全 + W5 全 + W2 一部 | W1 一部 + W2 一部 |
| **failure 時の learning** | 確実な lift、損失なし | LLM pipeline の prior art | metric 改訂で 0 になる |
| **次段階への接続** | 候補 2 の baseline / 教師に再利用 | 候補 1 の fallback 統合可 | 単独運用、次段なし |
| **学術的 contribution (IJCAI talk 候補)** | 中 (template engineering) | 高 (LLM-driven NN synthesis) | ゼロ (他人解の re-arrange) |
| **5/15 metric 改訂 robustness** | 高 (一次原理ベース) | 高 (LLM が再生成可) | **catastrophic** (= 改訂で無効化) |
| **Longest Leader $10K (= 7/15 まで首位継続) との親和** | 低 | 中 | 高 (即 push、ただし 5/15 で死) |

---

## 5. 段階的アプローチ案

```
Week 1-2: 候補 1 (Per-task hardcoded conv)
   → category × complexity 別 template 設計、簡単 100-200 task で baseline 5500 帯
   ↓ baseline 確立 (= 候補 2 の training data + 候補 3 の min-source)
Week 3-4: 候補 2 (LLM program synthesis)
   → 候補 1 で解けなかった残り 200 task を LLM 駆動で攻める
   → LLM の output を候補 1 template にも feedback (= refinement loop)
   → score 6500-7000 帯
   ↓ 5/15 前後の metric 改訂で再採点
Week 5-6: 候補 1 + 2 を融合した final ensemble
   → INT8 + Simplifier で全 400 task 圧縮、LLM hindsight learning で残り task refine
   → score 7000+ 目標 (= 現 LB top 帯)
Week 7-9: Longest Leader 賞狙いで首位維持戦
   → 5/15 metric 改訂対応、private overfitting 防止、submission strategy
```

(候補 3 は 5/15 metric 改訂 risk が catastrophic なので、**main submission 候補から除外推奨**だが、補助的に「他人解の logic を解析する」用途では使える。)

---

## 6. 中立判断ガイド (= 中央は推奨しない、開発者判断)

| もし開発者が ... | 候補 |
|---|---|
| **時間限られる、確実な順位上昇が欲しい (3-4 週)** | 候補 1 (Per-task hardcoded conv) |
| **LLM 自動化 + IJCAI talk 狙い、API cost OK** | 候補 2 (LLM program synthesis) |
| **メダル賞金狙い 100%、IJCAI talk は不要、5/15 risk OK** | 候補 3 (Public blending) — 但し 5/15 で死亡 risk |
| **学習目的、新手法を試したい** | 候補 2 |
| **判断保留、まず 1 だけ試してから決める** | 候補 1 → 結果見て候補 2 を選ぶ (段階的アプローチ) |
| **Longest Leader $10K 狙い** | 候補 3 で即 push → 7/15 まで首位維持 (但し 5/15 で激減) |

---

## 7. Out of scope (このプランではやらない)

- 候補のうちどれを実装するかの **選択** (= 開発者の判断、次プランで `/plan kaggle-neurogolf-2026-baseline` 起動時に決定)
- 候補 2 の場合の LLM prompt engineering 詳細 (実装プランで)
- 候補 3 の場合の特定公開 dataset の license 確認 (= use 前に individual に確認必須)
- LB top 1 (yiheng 7290.56) を抜く戦略 (= top 10 圏 6500+ を目標とする)
- private benchmark の特性推測 (= 主催者発表待ち、5/15 改訂と合わせて再評価)

---

## 8. 参照ファイル一覧 (Phase 1 deliverable map)

### Research (Track A) — W3
- `docs/research/past-comps.dense.md` (350 行、8 コンペ)
- `docs/research/past-comps.kids.md` (60 行)
- `docs/research/past-comps.references.json` (8 records, structural_relevance maximum: 2 件)

### Public kernels (Track B) — W2
- `docs/research/public_kernels/README.md` (= INDEX.md, 155 行, 44 kernel 分類)
- `docs/research/public_kernels/<slug>/content.md` × 15 件 (vote 上位 + score 上位 + 主催者 starter)
- `docs/research/kernels_manifest_2026-05-10.csv` (44 unique)

### LB observation (Track C)
- `docs/research/lb_snapshot_2026-05-10.csv` (top 200)

### Strategy (Track D) — W6 + W5
- `docs/strategy/first-principles.dense.md` (W6: §0/§1/§2/§3/§5/§6/§7、W5: §4)
- `docs/strategy/first-principles.kids.md` (W6 + W5)

### Host Data (Track E) — W4 (main agent 代行)
- `docs/research/host_datasets.md` (CLI 出力 + EDA + 戦略含意)
- `data/raw/task001-400.json` + `data/raw/neurogolf_utils/neurogolf_utils.py`
- `data/external/{neurogolf-2026-task-transformation-library, logic-for-each-arc-task, the-arc-gen-100k-dataset}/`

### Discussion (Track F) — W1
- `docs/discussion/insights.md` (155 行)
- `docs/discussion/2026-05-10.md` (4786 行 / 222 KB, 41 topic raw dump)
- `docs/discussion/topics.json` (41 topic index)

### Synthesis (中央, Phase 3)
- `docs/strategy/exp001-design.md` (= this file)

---

## 9. 次プランへの引き継ぎ事項

開発者は次のセッションで **候補 1/2/3 のいずれか (or その組み合わせ) を選択** し、`/plan kaggle-neurogolf-2026-baseline` で実装プランを立てる。

その際:

- このファイル (`exp001-design.md`) を Read で全文読み直す
- 該当候補の "Phase 1 知見の活用" 列に書かれた dense.md セクションを Read
- `docs/research/host_datasets.md` の §3 EDA 結果を Read で再確認
- `docs/strategy/first-principles.dense.md` §6 anti-pattern と §7 advantage を必ず再読 (= 候補ごとの cost 計算前提)
- `docs/discussion/insights.md` の **scorer-poison op 回避リスト** と **5/15 metric 改訂予想** を必ず読み込み、**exploit-free baseline と実験の 2 系統 submission** を計画

実装中に新しく見えてくる事 (公開 notebook の追加情報、新規 LB 観察、5/15 後の metric 改訂結果等) は **新しい research dense.md** として追記し、本ファイルは synthesis スナップショットとして固定する。

---

<!-- 以下、Phase 4 で critic agent 出力が append される -->

## critic 反論

> Phase 4 critic gate 出力。2026-05-10 編集。
> 役割: exp001-design.md (= 中央 / main agent が Phase 3 で書いた構造原理 3 案 + 評価軸) を **independent adversarial reviewer** として批判する。
> 出力規約: 最低 6 objection、各 objection に **lens / likelihood / impact / detectability / scenario / mitigation / applicable to** を付与する。本文は日本語、コード識別子・URL・file:line は英数字のまま。
> CRITICAL: critic は **修正案を「決定」しない**。failure scenario を提示し、開発者の判断を仰ぐだけ (= 主道 中立指示原則)。

---

### 前提として critic が見抜いた「artifact の暗黙仮定」

本 design.md は次の仮定を疑わずに前提化している。critic はこれら全てを攻撃対象とする。

1. **「per-task 30 分で hand-craft できる」 (候補 1)** — `exp001-design.md:130` "1 task あたり 30 分でも 400 × 30 = 200 時間"。30 分平均が成立する根拠は提示されていない。
2. **「arc_explanations.json の logic を LLM が正しく読める」 (候補 2)** — `exp001-design.md:162-168` で前提化。logic 文の質と LLM の reading comprehension fault rate は実測されていない。
3. **「5/15 metric 改訂後も候補 1, 2 は robust」 (評価軸)** — `exp001-design.md:257` "高 (一次原理ベース)" と評定。具体的にどの metric 変更パターンに耐えるかの考察なし。
4. **「段階的アプローチで Week 1-2 候補 1 → Week 3-4 候補 2」が線形に進む** — `exp001-design.md:264-278`。実際は依存と相互フィードバックがあり、bottleneck が serial に積み上がる。
5. **「private benchmark の特性は『主催者発表待ち』で扱える」** — `exp001-design.md:303`。public 400 task で 100% 解いても private で 0% になる risk を out-of-scope にしている。
6. **「Longest Leader 賞 7/15 まで首位継続が 1 戦略軸として独立」** — `exp001-design.md:258, 293`。5/10 時点で既に 4 日経過しており、5/6 起算で 5/10 までに後発が submit pipeline を完成させた可能性を計算していない。

---

### Objection 1: 候補 1 の「30 分 / task」は category 4-5 task で破綻、200 時間は楽観値の floor

- **lens**: operational
- **likelihood**: high
- **impact**: high
- **detectability**: pre-deploy testable (= category 5 task 5 件で実測すれば即わかる)
- **scenario**: `host_datasets.md:166-175` で Estimated_Complexity 分布が 1-8 (= 主催者は 1-5 と公称、実 data には 6-8 が出現、合計 4-8 が 331/400 task = **82.75%**) と判明している。design.md 自身が `exp001-design.md:131` で "Estimated_Complexity 4-5 task で 1-2 layer 不足" と認めているにもかかわらず、200 時間見積もりはこれら 331 task でも 30 分で完了する前提で計算されている。実際は (a) `arc_explanations.json` の logic を読み解き、(b) 30×30 静的 shape 制約下の dataflow を設計し、(c) 重み手書き、(d) 400 task のうち arc-gen 262 pair 全てで `score_network()` 100% pass を確認する、という 4 ステップを 30 分で完了することは Estimated_Complexity 5+ では不可能。実際の単価は 2-4 時間/task で、331 task × 2.5h = **827 時間 ≈ 21 週 (full-time)**。Week 2-4 で完走は不可能、deadline 7/15 (= 残り 65 日) も full-time でしか間に合わない。
- **mitigation**: critic は修正案を出さない。開発者は (a) 候補 1 を simple task (Complexity 1-3 = 69 task) のみに限定するか、(b) 30 分前提を撤回して再計画するか、(c) 自動化を Week 1 から並行投入するか、を判断する。
- **applicable to**: 候補 1

### Objection 2: 候補 2 の LLM hallucination 率は arc_explanations.json の質に強く依存、定量化されていない

- **lens**: correctness
- **likelihood**: high
- **impact**: high
- **detectability**: pre-deploy testable (= 50 task サンプルで LLM 生成 → local validate で fault rate 実測)
- **scenario**: `host_datasets.md:163-164` の `arc_explanations.json` は karnakbaevarthur が 2026-04-19 に upload した **第三者作成**の natural language description であり、主催者公式ではない。Pattern_Recognition (= 160 task / 40%) に対し「3-7 文の自然言語 explanation」が `host_datasets.md:92` で記載されているが、Pattern_Recognition 系 task で「3-7 文で transformation logic が一意に決まる」保証はない。例: ARC-AGI 公式 task の中には "find the largest object that has a specific shape pattern" のような文で記述されたが、実際には 4 通りの解釈が train pair から ambiguous なまま残る task が存在する (= `discussion/2026-05-10.md:3434-3469` で host が認知)。LLM が間違った解釈を picked up し ONNX を生成すると、**train pair で 100% pass しても arc-gen 262 pair / private で 0% になる**。この hallucination は ONNX 制約違反 (= scorer-poison op を選ぶ) と区別できず、「ONNX export 成功 + train pass + arc-gen fail」の 3 条件で初めて検出できる。Estimated_Complexity 4-8 (= 331 task / 82.75%) で hallucination 率 30-50% を想定すると、最終 successful task は 200/400 程度に止まる。
- **mitigation**: critic は修正案を出さない。開発者は (a) sample 50 task で LLM fault rate を実測してから候補 2 採否を判断、(b) arc-gen 262 pair の 100% pass を gate とするか、(c) Pattern_Recognition 系を候補 2 から除外し別アプローチに振るか、を判断する。
- **applicable to**: 候補 2

### Objection 3: 「5/15 metric 改訂で候補 3 のみ catastrophic」判定は楽観的、候補 1, 2 も少なくとも moderate damage を受ける

- **lens**: correctness
- **likelihood**: medium-high
- **impact**: high
- **detectability**: post-mortem only (= 5/15 改訂内容次第。事前検出不能)
- **scenario**: `exp001-design.md:257` で「5/15 metric 改訂 robustness: 候補 1 高 / 候補 2 高 / 候補 3 catastrophic」と評定されているが、`insights.md:84-89` で報告されている **未公開 exploit 4 件** (#697048 / #697059 / #697063 / #696365) が何の op 群を狙っているかは不明。`first-principles.dense.md:165` で W5 が "ONNX `Initializer` external_data で 1.44 MB 制約をすり抜け可能性 (要 W6 検証)" と指摘しており、これが exploit に当たるなら 5/15 で `Initializer` の memory 計上方法が変わる可能性がある。改訂が `MatMul` の memory 計算方法 / `Constant` の重複参照 / `QuantizeLinear+DequantizeLinear` の chain handling 等に及ぶと、INT8 quantize した候補 1 baseline も **再 score で cost +20-50% になる task が発生**し、avg score 16.59 → 14-15 に落ちる。design.md で「一次原理ベースだから robust」と書かれているが、現行 metric 自体が `onnx-tool 1.0.1` の実装に依存している (= `insights.md:87` の "scorer-poison list" と同根) ため、一次原理は「params + memory_bytes が下がれば score 上がる」という大方向しか保証しない。**特定の op 構成は依然 metric 改訂で再評価対象**。
- **mitigation**: critic は修正案を出さない。開発者は (a) 5/15 までに submission しないか、(b) submit するなら Initializer / Constant / MatMul / QDQ chain のいずれにも依存しない最小構成を別途用意するか、(c) 5/15 後に rescore 結果を見てから next step を決めるか、を判断する。
- **applicable to**: 全候補 (特に候補 1, 2 の "robust" 評定への異議)

### Objection 4: 段階的アプローチ (候補 1 → 2 → 融合) は serial 依存で deadline に間に合わない可能性

- **lens**: lifecycle
- **likelihood**: high
- **impact**: high
- **detectability**: pre-deploy testable (= Week 2 終了時点で候補 1 完成度を測れば判定可能)
- **scenario**: `exp001-design.md:264-278` の段階フローは Week 1-2 候補 1 → Week 3-4 候補 2 → Week 5-6 融合 → Week 7-9 首位維持戦、と serial 9 週間 (= 63 日) を仮定している。**しかし 2026-05-10 起点で deadline 2026-07-15 までは 65 日**。Objection 1 で示した通り候補 1 の Week 2 完成は楽観値であり、Week 2 終了時点で 100-150 task しか hand-craft できていなければ、候補 2 の "残り 200 task" は実際 250-300 task になる。さらに候補 2 は候補 1 の "教師 / fallback template" を前提とした設計 (`exp001-design.md:166-168` の "fail task は category 別 template で fallback") なので、候補 1 の遅延が候補 2 にそのまま積み上がる。Week 3 開始時点で候補 1 が 80% 未満なら、Week 3-4 で候補 2 を始めても候補 1 完成と並行作業になり、両方が deadline に間に合わない risk。さらに `Longest Leader` 賞の 5/6 起算 (= 既に 5/10 時点で 4 日経過) を考えると、この段階フローは Longest Leader 賞戦略と完全に矛盾する (= 7/15 まで首位継続するには 5/10-5/30 のうちに submit pipeline を立ち上げる必要があるが、この plan では Week 2 まで何も submit しない)。
- **mitigation**: critic は修正案を出さない。開発者は (a) parallel に候補 1, 2 を Week 1 から並走するか、(b) Week 1 終了時点で submit MVP (= 公式 starter ベース) を出して LB に存在させるか、(c) 段階を 9 週から 6 週に圧縮するか、を判断する。
- **applicable to**: 段階フロー全体

### Objection 5: scorer-poison op の発見リスクは public 400 task で動いても private で発火する可能性がある

- **lens**: correctness
- **likelihood**: medium
- **impact**: catastrophic (= 全 400 task 0 点)
- **detectability**: runtime only (= local `score_network()` で raise しなければ submit してから出る)
- **scenario**: `insights.md:83` で報告されている scorer-poison op (`ArgMin / TopK / Where+uint8 / Compress`) は **既知** リストだが、`insights.md:84-89` で「5/5 時点で 4 件の未公開 exploit 報告」、`insights.md:88` で「未報告 poison op が残る可能性」と明示されている。候補 1 hand-craft でも候補 2 LLM 生成でも、 **`Where + uint8` のような既知 poison combination** は注意深く避けるが、未知の combination (例: `Cast(uint8) → MatMul`、`Pad + negative dim`、`Reshape + symbolic dim`) は local では通っても submit すると "Error processing one or more onnx networks" で task 単位の診断なしに 0 点になる (`insights.md:94`)。さらに「**local validator が成功しても private benchmark の grid_size 分布で profiler が違う path を踏む**」と poison op に **後から** 当たる可能性がある (= input shape 依存の crash)。これは特に Pattern_Recognition の 160 task で発火率が高い。public で動いた 400 task のうち 50 task でも private で 0 点になれば、avg score 17.5 → 15.3 に落ち、メダル圏外。
- **mitigation**: critic は修正案を出さない。開発者は (a) `score_network()` を arc-gen 262 pair で全て pass するまで gate するか、(b) op 許容リストを whitelist 化 (= blacklist でなく) するか、(c) 未公開 exploit 4 件の影響を 5/15 改訂後に reassess するか、を判断する。
- **applicable to**: 全候補 (特に候補 2 LLM 生成で発火率高)

### Objection 6: private benchmark で arc_primitives.json category overfitting が発火する可能性が high

- **lens**: correctness
- **likelihood**: high
- **impact**: high
- **detectability**: post-mortem only (= 主催者が private LB を公開するまで不可視)
- **scenario**: `host_datasets.md:155-180` で arc_primitives.json の Primary_Category (Pattern_Recognition 40%, Object_Based 39.5%, Spatial_and_Geometric 11.8%, Color_and_Logical 8.8%) を category 別 template に使う設計 (`exp001-design.md:113-116`) が両候補で前提となっている。この arc_primitives.json は **karnakbaevarthur が 2026-04-19 に upload した第三者注釈**であり、`host_datasets.md:76` で "(推定) CC BY 4.0" と書かれている通り**主催者公式ではない**。private benchmark の task 分類はこの categorization と一致する保証がない。さらに critic 仮説: 主催者は private benchmark を意図的に **public 400 task の category 分布と異なる** ように設計している可能性が高い (= ARC-AGI の本質的目的が "general intelligence" であり、categorization で解ける = inductive bias で解けるのは ARC の本来意図と矛盾する)。例えば private で Pattern_Recognition が 5% / Object_Based が 70% という分布になれば、Pattern_Recognition に最適化した template が稼働しない 60% の task が生じる。avg score が public 17.5 → private 13-14 になる risk。`insights.md:114` の `#696569` で「locally against freshly generated synthetic pairs ... I can see it score less than 100% pass rate」が既に観測されているのと同根の現象が拡大する可能性。
- **mitigation**: critic は修正案を出さない。開発者は (a) arc-gen 262 pair で 100% pass を gate にするか、(b) category 別 template を使わず task 単独で arc-gen pair から rule を induce するか、(c) private LB に shake-up 想定を組むか、を判断する。
- **applicable to**: 全候補 (特に候補 1 で fatal、候補 2 でも依然 high)

### Objection 7: Longest Leader $10K の 5/6 起算 - 7/15 終了は既に 4 日後発、現実性が再検討必要

- **lens**: scalability
- **likelihood**: medium-high
- **impact**: medium
- **detectability**: pre-deploy testable (= 5/6-5/10 の LB top の score 推移を見れば判定可能)
- **scenario**: `exp001-design.md:258` で「Longest Leader $10K (= 7/15 まで首位継続) との親和: 候補 1 低 / 候補 2 中 / 候補 3 高 (即 push、ただし 5/15 で死)」と評定されているが、**現実には 5/6 00:00 UTC 起算なので 2026-05-10 時点で既に 4 日経過**。現 LB top yiheng "5.5 or 4.7?" 7290.56 (`exp001-design.md:18`) は 5/6 起算後にも首位を維持中の可能性が高い (要 LB 履歴確認、`insights.md:17` で 5/4 改訂後 LB top 戦略は blending 独占)。仮に yiheng が 5/6 起算で既に 4 日連続首位を維持していて 5/15 metric 改訂後も blending を維持できれば、後発の我々が "首位を奪う + 60 日連続維持" は **2 段階の高い壁** を乗り越える必要がある。さらに段階的アプローチ Week 1-2 (= 5/10-5/24) で submit MVP を出すと言う計画では、**最も早くて 5/24 に首位を奪う = 残り 52 日連続維持**となるが、52 日間の中に 5/15 metric 改訂と未公開 exploit fix が含まれるので、score の不安定性が高い。Longest Leader 戦略を採用するなら、**5/10-5/14 に何かを submit して LB に存在させる即応 plan が必要**だが、本 design.md にはその記述がない。
- **mitigation**: critic は修正案を出さない。開発者は (a) Longest Leader 戦略を断念し他賞 (1st-3rd) に集中するか、(b) Week 1 を 1 day に圧縮し公式 starter で 5/11 に submit するか、(c) Longest Leader を主目的にせず他賞達成の副次効果として狙うか、を判断する。
- **applicable to**: Longest Leader 賞戦略全般、特に段階的アプローチとの整合性

### Objection 8: broken-task list (W2 finding `t045/t067/t111/t159/t176/t192/t210/t256/t309/t320` 等) の扱いが design.md に記述されていない

- **lens**: operational
- **likelihood**: medium
- **impact**: medium
- **detectability**: pre-deploy testable (= 該当 task を arc-gen で実行すれば確認可能)
- **scenario**: critic 質問プロンプトで「`t045/t067/t111/t159/t176/t192/t210/t256/t309/t320` および `{21, 55, 80, 184, 202, 366}` で除外推奨されている task をどう扱うか」と指定されているにもかかわらず、`exp001-design.md` 全文 (357 行) に **これら broken-task の記述が一切ない**。design.md は 400 task 全てを uniform に処理する前提で書かれているが、実際は 16 task が "broken" (= train pair から rule が一意に決まらない、もしくは scorer の特殊挙動を踏む) と W2 で報告されている。これらを default として全 400 task の hand-craft / LLM 生成に含めると (a) hand-craft 工数の 4% が捨てられる、(b) LLM が無理に rule を fit しようとして hallucination が増える、(c) submit 時にこれら task で 0 点を取る、という triple loss が発生する。avg score への直接影響は 16 task / 400 = 4% (= avg 17.5 × 0.04 = 0.7 pt loss / task = 280 pt loss) で medium impact だが、**broken-task が 16 件しかない保証もない** (= 主催者と W2 の認知が一致していない可能性)。
- **mitigation**: critic は修正案を出さない。開発者は (a) broken-task list を W2 から正式に取得して design.md に追記するか、(b) これら task のみ public blending zip から流用するか (= 候補 3 の補助利用)、(c) 17.5 pt 上限を 17.0 に下げて期待値を補正するか、を判断する。
- **applicable to**: 全候補 (broken-task の扱いを明示する責任は design 段階)

### Objection 9: ONNX `Initializer` external_data exploit (W5 が `first-principles.dense.md:165` で指摘) を design.md が「要検証」のまま放置している

- **lens**: security
- **likelihood**: medium
- **impact**: medium-high
- **detectability**: pre-deploy testable (= local で external_data ファイル付き ONNX を `score_network()` に通せば即判定)
- **scenario**: `first-principles.dense.md:165` で W5 が "ONNX `Initializer` external_data で 1.44 MB 制約をすり抜け可能性 (要 W6 検証)" と明記している。これは **重要な失敗 / 悪用リスク** だが、`exp001-design.md:90` で簡単に言及されているのみで、**3 候補のいずれにも external_data の扱い** が明示されていない。critic 観点での 2 つの failure scenario: (a) 我々の baseline (候補 1) が知らずに external_data で書き出すと、**現行 scorer は通す** が **5/15 改訂で塞がれて全 task 0 点になる**。(b) 我々が意図的に external_data exploit を狙うと、5/15 改訂で塞がれた瞬間 catastrophic loss + 5/15 fix の trigger 元として host が IP/account level でフラグ立てる risk (= 競技倫理違反扱い)。(a) は accident、(b) は deliberate、いずれも design 段階で明示しないと avoidable risk を取り損ねる。さらに `torch.onnx.export(..., dynamo=True)` の default では initializer のサイズが threshold (~1024 bytes) を超えると **自動的に external_data 化する場合がある** ので、開発者が意図せず exploit ルートに入る可能性がある (要 PyTorch docs 検証)。
- **mitigation**: critic は修正案を出さない。開発者は (a) export 時に `save_as_external_data=False` を強制するか、(b) external_data の current scorer での scoring を local で実測し whitelist/blacklist 判断するか、(c) 5/15 改訂の含む可能性が高い項目として monitoring するか、を判断する。
- **applicable to**: 全候補 (特に candidate 1 の INT8 quantize、candidate 2 の LLM 生成 ONNX で意図せぬ external_data 化が起きやすい)

### Objection 10: 候補 3 (public blending) の license / 競技規約 risk が "use 前に individual に確認必須" だけで済まされている

- **lens**: team
- **likelihood**: medium
- **impact**: high
- **detectability**: post-mortem only (= 主催者が disqualify するまで不可視)
- **scenario**: `exp001-design.md:301` で "候補 3 の場合の特定公開 dataset の license 確認 (= use 前に individual に確認必須)" と out-of-scope 化されている。しかし候補 3 を本格採用すると次の team / process risk が走る: (a) **`konbu17/neurogolf-2026-blended-401-v117` 等の Kaggle dataset が CC BY 4.0 等の public license** だとしても、**Kaggle 競技規約は「他参加者の submission をそのまま転載することを禁止する場合がある」** (= Kaggle 公式 rule で "You may not submit any solution that is not your own original work" が default。`exp001-design.md:227` で "主催者が blending submission を ban" を likelihood: low と評定しているが、low の根拠が示されていない)。(b) IJCAI-ECAI 2026 talk invitation の選考時に host が submission の独立性を verify する場合、blending 主体の解答は **disqualify 対象になる**。(c) Top Student $8K 賞の elig 判定時に同じく学生 author が独立解答を提出していることが要件になる可能性。(d) 仮に賞金は獲得できても **学術的 contribution ゼロ** が `exp001-design.md:256` で明示されており、professional reputation cost が発生する。
- **mitigation**: critic は修正案を出さない。開発者は (a) 候補 3 を main submission から完全に外すか、(b) 補助 (= 他人解の logic を解析する用途) のみに限定するか、(c) license + Kaggle 規約を 5/10-5/15 までに verify するか、を判断する。
- **applicable to**: 候補 3 (および補助利用の出典記録の team-process の責務)

---

### 集計と最も致命的な 2 件

| # | objection | lens | likelihood | impact | detectability |
|---|---|---|---|---|---|
| 1 | 候補 1 の 30 分 / task は破綻 | operational | high | high | pre-deploy testable |
| 2 | 候補 2 の LLM hallucination 率定量化されていない | correctness | high | high | pre-deploy testable |
| 3 | 5/15 改訂の damage 範囲が候補 3 単独でない | correctness | medium-high | high | post-mortem only |
| 4 | 段階アプローチが serial 依存で deadline 危険 | lifecycle | high | high | pre-deploy testable |
| 5 | scorer-poison op の private 発火 | correctness | medium | catastrophic | runtime only |
| 6 | private benchmark で category overfit が発火 | correctness | high | high | post-mortem only |
| 7 | Longest Leader 既に 4 日後発 | scalability | medium-high | medium | pre-deploy testable |
| 8 | broken-task の扱いが design.md に未記述 | operational | medium | medium | pre-deploy testable |
| 9 | external_data exploit が "要検証" 放置 | security | medium | medium-high | pre-deploy testable |
| 10 | 候補 3 license / Kaggle 規約 risk | team | medium | high | post-mortem only |

**lens 分布**: correctness 4, operational 2, security 1, scalability 1, lifecycle 1, team 1 = **合計 10 (≥ 6 satisfied)**

**最も致命的な 2 件 (= 開発者が最優先で考慮すべき)**:

1. **Objection 5 (scorer-poison op の private 発火)** — catastrophic impact + runtime only detectable で、submit してから 0 点を引く typical failure。**全候補に共通**で、特に candidate 2 LLM 生成では発火率が高い。pre-deploy gate (= arc-gen 262 pair で `score_network()` 100% pass) を作らない限り回避不能。
2. **Objection 1 (候補 1 の 30 分 / task は category 4-8 で破綻)** — high likelihood × high impact で、Week 1-2 完走前提が崩れる。Estimated_Complexity 4-8 が 331/400 task = 82.75% を占める実測 (`host_datasets.md:166-175`) を踏まえれば、200 時間見積もりは **4 倍楽観値**で実態は 800+ 時間。段階的アプローチ全体が崩壊する起点。

### 候補別 viability 判定

- **候補 1 (Per-task hardcoded conv)**: **moderate-severe risk** — 30 分/task 前提が破綻、Estimated_Complexity 4-8 で工数 4 倍。ただし simple task (Complexity 1-3 = 69 task) のみに限定すれば achievable で、その範囲では確実な lift。
- **候補 2 (LLM program synthesis)**: **moderate risk** — LLM hallucination 率が arc_explanations.json の質に依存、未実測。category Pattern_Recognition (40%) で fault rate 30-50% 想定。ただし 50 task サンプルで実測すれば fault rate を pre-deploy で gate できる。
- **候補 3 (Public blending)**: **severe risk** — 5/15 metric 改訂 catastrophic + private overfit + license / Kaggle 規約 disqualify risk。design.md の "main submission 候補から除外推奨" 判断は妥当だが、補助利用 (= 他人解の logic 解析) も license risk を残すので慎重に。

### what would change my mind (= 開発者が以下を提示すれば critic は対応する objection を撤回する)

1. **Objection 1 撤回条件**: Estimated_Complexity 4-5 の task 5 件で実 hand-craft を行い、arc-gen 262 pair で 100% pass する完成品を平均 30 分以内で得られた実測ログ。
2. **Objection 2 撤回条件**: arc_explanations.json の logic を LLM に与えて 50 task の ONNX 生成、arc-gen で 100% pass の rate を実測 (期待: ≥ 70%)。
3. **Objection 3 撤回条件**: 5/15 改訂後の metric 仕様を host が事前公開し、Initializer/Constant/MatMul の cost 計算が現行と等価であることを host が明言。
4. **Objection 4 撤回条件**: Week 1 (= 5/10-5/17) で MVP submission (= 公式 starter ベース) を LB に存在させ、Week 2 開始時点で hand-craft 200+ task を完了している実績。
5. **Objection 5 撤回条件**: scorer-poison op の網羅 whitelist を作成、`score_network()` を arc-gen 262 pair で 100% pass する gate が CI に組み込まれている。
6. **Objection 6 撤回条件**: arc-gen 262 pair で 100% pass + 主催者から private benchmark の category 分布が public と一致する保証。
7. **Objection 7 撤回条件**: 5/10-5/14 に submit MVP が LB に出ており、5/15 改訂耐性を持つ最小構成が backup として用意されている実績。
8. **Objection 8 撤回条件**: W2 broken-task list (16 件) が公式 exclude として design.md に追記され、各 task の代替戦略が明示。
9. **Objection 9 撤回条件**: external_data 機構が現行 scorer で reject される実測ログ、または 5/15 改訂後も保持される旨の host 公式回答。
10. **Objection 10 撤回条件**: Kaggle 競技規約 + 候補 dataset license を verify、blending submission が 1st-3rd / Top Student / Longest Leader 賞 elig に影響しないことを host コメントで確認。

---

### 完了スタンプ

- critic agent: 2026-05-10
- objection 数: 10 (≥ 6)
- lens: correctness 4 / operational 2 / security 1 / scalability 1 / lifecycle 1 / team 1
- artifact 末尾に append 完了 (上書きなし)
