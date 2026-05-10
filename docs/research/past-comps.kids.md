# 過去の類似コンペ — TL;DR 版

> neurogolf-2026 向け、2026-05-10 編集。
> 詳細は `docs/research/past-comps.dense.md`、構造化 DB は `past-comps.references.json`。

---

## どんなコンペが「似てる」か?

**構造類似度 maximum** (= 2 軸別々に):
- **ARC Prize 2024** (Kaggle 2024) — ARC を解く + NN を効率的に作る、両立必須だから (軸 A)
- **MicroNet Challenge 2019** (NeurIPS 2019) — NN 圧縮が主目的、cost = params + memory が直接同型 (軸 B)

**構造類似度 high** (5 件):
- **Abstraction & Reasoning Challenge** (Kaggle 2019) — DSL + program synthesis の初代 SOTA
- **CompressARC** (2025) — 76 K params で ARC 解く、NeuroGolf の目標に最接近
- **Tiny Recursive Model** (2025) — 7 M params + recursive loop、minimal architecture design
- **SOAR** (2025) — LLM + evolutionary search で program synthesis
- **ARC Prize 2025** (2025) — 最新 SOTA、test-time training + TRM fusion

---

## 過去の優勝者は何をしたか? (3 件)

### ARC Prize 2024「ARChitects」(1 位, 53.5% score)

- **手法**: Test-Time Training (TTT) + LoRA adapter per-task
- **すごい工夫**:
  - Geometric augmentation (rotation/flip) で training data 拡張
  - Transduction (grid → grid direct) + induction (program search) を hybrid
- **当該コンペでも使える?**: 部分的に。ONNX inference-only なら TTT は不可だが、program synthesis + NN compile は可。

### MicroNet Challenge 2019「MIT-Han Lab」(1 位, perplexity 34.1)

- **手法**: Pruning + 8-bit quantization + knowledge distillation
- **すごい工夫**:
  - Adaptive embedding (weight sharing で param 削減)
  - Automatic Gradient-based Pruning (AGP) で 9× 削減、quantization で 32 bit → 8 bit
- **当該コンペでも使える?**: ほぼそのまま。1.8 M params → 360 K params へ scale-down するだけ。

### CompressARC (Isaac Liao, 2025)

- **手法**: MDL-based neural code golf (VAE + decoder regularization)
- **すごい工夫**:
  - 76 K params のみで 20-34% 達成 (ARC-AGI-1)
  - Per-task gradient descent で program search (combinatorial explosion 回避)
- **当該コンペでも使える?**: はい、ほぼそのまま。ONNX 1.44 MB 制約に最適化済み。

---

## 「これは外せない」共通テクニック

過去 top 5 が全員やっていること:

1. **Refinement loop (per-task iteration)** — Test-time training か program search か parameter update で per-puzzle の最適化
2. **Synthetic task generation + augmentation** — Leave-one-out + geometric transforms で training data 拡張
3. **Pruning + quantization pipeline** — 精度保ちつつ 30-50× 圧縮 (MicroNet, CompressARC の precedent)
4. **Ensemble or modality fusion** — Symbolic + neural、あるいは TTT + recursive (単一 paradigm 上限の壁突破)

---

## 当該コンペでの推奨実装順

```
Week 1-2: Baseline (公式 starter または simple DSL で 400 task カバー)
Week 3-4: Program synthesis (LLM-driven or symbolic search)
Week 5-6: NN compilation + 8-bit quantization (weight tying, pruning)
Week 7-8: Ensemble + per-task refinement + leaderboard tuning
```

---

## 注意すべき過去事例

- **Symbolic vs Neural の false dilemma**: ARC Prize 2024 では両方を hybrid したチームが winning。どちらか単体は 40% 程度
- **Over-compression の pitfall**: CompressARC は 76 K params でも 20% 達成だが、問題の difficulty 分布で「簡単な 100 task に特化」では public eval overfitting。private eval で 4% に落ちる例多数
