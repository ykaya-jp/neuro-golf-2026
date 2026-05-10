# 公開 kernel カタログ (NeuroGolf 2026)

> 2026-05-10 編集 (W2: kernel-taxonomist)。
> Pull スクリプト: `~/.claude/skills/kaggle-onboard/tools/pull_kernels.sh`
> 元 manifest: `docs/research/kernels_manifest_2026-05-10.csv` (= vote 上位 30 ∪ score 上位 30 で 44 unique kernel)
> 主出力: 上位 15 件は `<slug>/content.md` で詳細要点化、残り 29 件は本 INDEX に metadata のみ列挙

<!--
W2 (kernel-taxonomist) の主出力。
- 各 kernel ごとに `<kernel-slug>/content.md` (= 要点化) + `<kernel-slug>/kernel-metadata.json` (= raw)
- このファイル (= INDEX.md) は全 44 kernel の「technique 別索引」
- 上位選手は public kernel を出さないことが多い (= LB top 1 の解法が含まれるとは限らない)
-->

## このコンペの technique 構造

NeuroGolf 2026 は ARC-AGI 画像変換タスクを **最小 ONNX network** (= params + memory が小さいほど高得点) で解く golf 系。skeleton の章構成 (= IL / RL / GBDT / Transformer 等) は本コンペには合わないため、neurogolf 固有の technique 軸で再分類する。

LB は **2026-05-06 採点 rule 変更** (= MACs を採点から除外、`25 - log(max(1, memory + params))` に変更) を境に大きく動いており、変更後は **「他人の解 zip を集めて per-task 最安を選ぶ blending」** が public kernel 上位を独占している (出典: `neurogolf-2026-may-8-updated/content.md`)。本格的な per-task solver (= Tiny-ONNX / Nano-Engine 系) は少数派で、LLM-driven program synthesis は logic-decoder 系統の準備段階に留まる。

---

## INDEX (technique 別)

### A. Handcrafted ONNX weights — rule-based, no training

主催者公式 starter が示した「1 層 Conv2D の重みを純関数で書き下す」アプローチ。三値 {-1, 0, +1} 重みでサイズ最小化。

| kernel | author | vote | score (LB) | 1 行特徴 |
|---|---|---|---|---|
| [`the-2026-neurogolf-championship`](./the-2026-neurogolf-championship/content.md) | mmoffitt | 185 | (例示) | 主催者公式 starter、1 層 Conv2D の三値重み手書き |
| [`neurogolf-2026-improved-starter-notebook`](./neurogolf-2026-improved-starter-notebook/content.md) | yash9439 | 108 | (starter) | V2 の MAC 計算バグを修正、Cost = params + bytes + MACs |
| `neurogolf-championship-2026-starter-notebook` | parthenos | 86 | (starter) | starter の別実装、metadata only |
| `neurogolf-2026-starter` | sigmaborov | 87 | (starter) | starter の別実装、metadata only |
| `neurogolf-2026-onnx` | mpwolke | 32 | (starter) | ONNX 入門 starter |
| `the-2026-neurogolf-championship` (foysalemonshanto fork) | foysalemonshanto | 33 | (starter fork) | mmoffitt fork |
| `the-2026-neurogolf-championship` (ldausl fork) | ldausl | 31 | (starter fork) | mmoffitt fork |
| `neurogolf-2026-public-data` | sigmaborov | 56 | - | public data 整理用 |

### B. Logic-driven ensemble blending — submission zip aggregator

複数の他人 submission zip を per-task で best-pick して merge。**現行 LB 上位帯 (= 5500-6200+) の主流。**

| kernel | author | vote | score (LB) | 1 行特徴 |
|---|---|---|---|---|
| [`ngc26-constraint-smart-logic-mix-blending`](./ngc26-constraint-smart-logic-mix-blending/content.md) | jonathanchan | 115 | ~5546 (v16) | v1-v17+ で版ごとに `+取り込み source / +N 点 LB` を完全公開、credit graph 形成 |
| [`neurogolf-5480-41-current-rules-score`](./neurogolf-5480-41-current-rules-score/content.md) | afr1ste | 82 | **5480.41** | manifest SHA256 固定で score 再現性を保証、negative probes の除外明示 |
| [`neurogolf-2026-may-8-updated`](./neurogolf-2026-may-8-updated/content.md) | konbu17 | 48 | **5571.69** | 採点 rule 変更追従 + broadcast Mul/Add の `Tile` 修復 + 138 task source swap |
| [`neurogolf-multi-source-onnx-solver`](./neurogolf-multi-source-onnx-solver/content.md) | vyankteshdwivedi | 64 | ~6000+ (源 source) | 公式 `score_network()` の dynamic import + `EXCLUDED_TASKS={21,55,80,184,202,366}` |
| [`cross-source-ensemble`](./cross-source-ensemble/content.md) | karnakbaevarthur | 64 | ~5000 帯 | `onnx-tool==1.0.0` 強制 + priority tie-break、3 source の小規模 blend |
| [`neurogolf-2026-rule-based-onnx-solver`](./neurogolf-2026-rule-based-onnx-solver/content.md) | imaadmahmood | 75 | ~4979 (源 source) | `compute_cost` で shape inference + Conv/Gemm の MAC 計算を厳密実装 |
| [`arc-nano-engine`](./arc-nano-engine/content.md) | svanikkolli | 63 | ~5500-6225 (源) | 4 notebook + 31 dataset = 35 source の auto-load aggregator |
| [`neurogolf-new-blending`](./neurogolf-new-blending/content.md) | magmacot | 27 | (集約のみ) | 「rename 不要なら raw bytes を一切触らない」原則の安全 blend |
| [`5700-neurogolf-logic-driven-ensemble`](./5700-neurogolf-logic-driven-ensemble/content.md) | magmacot | 5 | ~5700 主張 | new-blending 派生 + IMMUTABLE_TASKS lock 機構 |
| [`neurogolf-4250`](./neurogolf-4250/content.md) | needless090 | 53 | **4250** | 14 行の極小 extract+rezip、後続 kernel が version pin 取り込み |
| `neurogolf-emsembling` | yash9439 | 63 | - | starter 派生の ensembling、metadata only |
| `kaggle-agent-ensemble-with-yash9439` | jiweiliu | 64 | - | Kaggle agent + yash9439 系の組合せ、metadata only |
| `neurogolf-2026-blended-till-4-27` | konbu17 | 60 | (4/27 までの blend) | konbu17 シリーズ前段、metadata only |
| `neurogolf-2026-till-4-27` | konbu17 | 59 | (4/27 までの blend) | 同上、metadata only |
| `8-may-update` | hanifnoerrofiq | 46 | (5/8 update) | 5/8 採点変更後の sub、metadata only |
| `neurogolf-4808-21-post-apr-28-update` | thisray | 34 | **4808.21** | 4/28 update、metadata only |
| `magmacot/4200-v5-neurogolf-fix-for-new-system-soon` | magmacot | 73 | ~4200 | magmacot シリーズ baseline、metadata only |
| `magmacot/4289-submission` | magmacot | 11 | ~4289 | metadata only |
| `magmacot/4213-neurogolf-blending-v1-new-metric` | magmacot | 3 | ~4213 | metadata only |
| `magmacot/5600-neurogolf-logic-driven-ensemble` | magmacot | 14 | ~5600 主張 | metadata only |
| `magmacot/neurogolf-logic-driven-ensembe` | magmacot | 4 | - | metadata only |
| `magmacot/neurogolf-ensembling` | magmacot | 4 | - | metadata only |
| `rauffauzanrambe/neurogolf-constraint-smart-logic-ensemble-4k` | rauffauzanrambe | 12 | **4000+** | jonathanchan v1 の派生元、metadata only |
| `amanatar/neurogolf-2026-optimal-blending-ensemble` | amanatar | 2 | - | metadata only |
| `rockerritesh/neurogolf-2026-5353-lb-public-submission` | rockerritesh | 3 | **5353** | metadata only |
| `yutodennou/neurogolf-championship-ensemble-full-code` | waticson | 1 | - | metadata only |
| `davebogd/neurogolf-2026-submission` | davebogd | 1 | - | metadata only |
| `ashok205/neurogolf-locally-solved` | NNMax | 31 | - | locally solved 系、metadata only |

### C. Compressed network search — small NN train + ONNX export

per-task で small NN を学習し ONNX export。重みを ternary {-1, 0, +1} に snap してサイズ削減。

| kernel | author | vote | score (LB) | 1 行特徴 |
|---|---|---|---|---|
| [`neurogolf-2026-tiny-onnx-solver`](./neurogolf-2026-tiny-onnx-solver/content.md) | aliafzal9323 | 64 | (詳細不明) | detector + multi-seed learned-conv + ternary snap fallback |
| `hanifnoerrofiq/neurogolf-sparse-builder` | hanifnoerrofiq | 34 | - | sparse network builder、metadata only |
| `dimokshalashov/task001-optimized` | dimokshalashov | 6 | (1 task のみ) | task001 だけ最適化、metadata only |

### D. LLM-driven program synthesis (logic decoder + taxonomy)

LLM (DeepSeek 等) で task の logic を自然言語/構造化 JSON 化し、ONNX template selector の入力にする。

| kernel | author | vote | score (LB) | 1 行特徴 |
|---|---|---|---|---|
| [`logic-decoder`](./logic-decoder/content.md) | karnakbaevarthur | 73 | (補助 dataset) | DeepSeek で各 task の 2-3 sentence rule を生成、JSON 永続化 |
| [`neurogolf-all-task-logic-complexity-map`](./neurogolf-all-task-logic-complexity-map/content.md) | karnakbaevarthur | 71 | (補助 dataset) | 24 primitive の closed vocabulary + difficulty 1-10 |

### E. Per-task tiny conv (1-2 layer hand-tuned)

(handcrafted weights と connect が深いため A に統合。独立 kernel は無し。)

### F. Mixed source aggregator at scale (公開 dataset の submission を統合)

(B に統合。`arc-nano-engine` / `neurogolf-multi-source-onnx-solver` がここに該当。)

### G. EDA / starter (no real solution)

(A の starter group + `sigmaborov/neurogolf-2026-public-data` 等。スコア生成しない。)

### H. Cost optimization / scoring research

| kernel | author | vote | score (LB) | 1 行特徴 |
|---|---|---|---|---|
| `yash9439/neurogolf-2026-cost-optimization` | yash9439 | 29 | - | cost 最小化観点の研究 kernel、metadata only |
| `poonszesen/strategies-explained-openonnxscience` | theredbluepill | 6 | (4723.80 source) | 戦略解説 kernel、metadata only |
| `jazivxt/infinitesimals` | jazivxt | 76 | - | 高 vote 数だが詳細未確認、metadata only |

---

## 横断的な技法発見 (= 上位 5+ kernel に共通する technique)

W2 が 15 kernel 詳細解析 + 残り 29 metadata 観察から抽出した、複数 kernel に再現する pattern。

1. **per-task best-pick blending** (= ensemble_blending の核)
   - 出典: [ngc26-constraint-smart-logic-mix-blending/content.md](./ngc26-constraint-smart-logic-mix-blending/content.md), [neurogolf-5480-41-current-rules-score/content.md](./neurogolf-5480-41-current-rules-score/content.md), [neurogolf-multi-source-onnx-solver/content.md](./neurogolf-multi-source-onnx-solver/content.md), [cross-source-ensemble/content.md](./cross-source-ensemble/content.md), [arc-nano-engine/content.md](./arc-nano-engine/content.md), [neurogolf-new-blending/content.md](./neurogolf-new-blending/content.md), [neurogolf-2026-may-8-updated/content.md](./neurogolf-2026-may-8-updated/content.md)
   - 全 7 kernel が「複数 source の `task001.onnx` ～ `task400.onnx` を集めて per-task で最小コスト解を採用」する pipeline。これが現行 LB の主流戦略

2. **submission.zip の `task<NNN>.onnx` 命名規約と 1.44 MB / file 制約への厳格遵守**
   - 出典: [the-2026-neurogolf-championship/content.md](./the-2026-neurogolf-championship/content.md), [ngc26-constraint-smart-logic-mix-blending/content.md](./ngc26-constraint-smart-logic-mix-blending/content.md), [neurogolf-2026-rule-based-onnx-solver/content.md](./neurogolf-2026-rule-based-onnx-solver/content.md), [neurogolf-2026-tiny-onnx-solver/content.md](./neurogolf-2026-tiny-onnx-solver/content.md), [cross-source-ensemble/content.md](./cross-source-ensemble/content.md)
   - `TASK_RE = re.compile(r'^task\d{3}\.onnx$')` で正規化。`MAX_BYTES = int(1.44 * 1024 * 1024)` で per-file チェック (= submission.zip 全体ではなく per-file)。注意: ngc26 v10 で誤認 fix 済み

3. **banned-op gate** (`Loop, Scan, NonZero, Unique, If, Function, Script`)
   - 出典: [ngc26-constraint-smart-logic-mix-blending/content.md](./ngc26-constraint-smart-logic-mix-blending/content.md), [neurogolf-2026-rule-based-onnx-solver/content.md](./neurogolf-2026-rule-based-onnx-solver/content.md), [neurogolf-2026-tiny-onnx-solver/content.md](./neurogolf-2026-tiny-onnx-solver/content.md), [cross-source-ensemble/content.md](./cross-source-ensemble/content.md)
   - 各 kernel で BANNED_OPS の中身に微差あり (例: `cross-source-ensemble` は `Script/Function` を含むが ngc26 / rule-based-onnx-solver は含まない)。**最も厳格な集合は `cross-source-ensemble` の `{Loop, Scan, NonZero, Unique, Script, Function}`**

4. **ONNX input/output 名の `'input'`/`'output'` 強制 + IR 8 / opset 13 normalization**
   - 出典: [neurogolf-new-blending/content.md](./neurogolf-new-blending/content.md), [5700-neurogolf-logic-driven-ensemble/content.md](./5700-neurogolf-logic-driven-ensemble/content.md), [neurogolf-2026-may-8-updated/content.md](./neurogolf-2026-may-8-updated/content.md)
   - grader が `input` / `output` 名を期待し、IR version 8 / opset 13 以外を弾く。re-serialization で他 op が壊れない rename 順序 (= 先に old name capture → 後で rename) が確立されている

5. **採点 rule 変更 (2026-05-06) への対応**
   - 出典: [neurogolf-2026-may-8-updated/content.md](./neurogolf-2026-may-8-updated/content.md), [neurogolf-2026-improved-starter-notebook/content.md](./neurogolf-2026-improved-starter-notebook/content.md)
   - May 6 を境に MACs 不要 → `25 - log(max(1, memory + params))`。MACs を含む旧 cost 関数は再評価必須。Conv-heavy 解が新採点で逆に有利化

6. **broken-task list (= grader が壊す既知 task)**
   - 出典: [neurogolf-2026-may-8-updated/content.md](./neurogolf-2026-may-8-updated/content.md) (= `t045/t067/t111/t159/t176/t192/t210/t256/t309/t320` の broadcast Mul/Add bug + `Tile` 修復), [neurogolf-multi-source-onnx-solver/content.md](./neurogolf-multi-source-onnx-solver/content.md) (= `EXCLUDED_TASKS = {21, 55, 80, 184, 202, 366}`)
   - 我々の baseline でも同等の blacklist + 修復ロジックを最初から組み込むべき

7. **public score を信じ、local cost を最終判断にしない**
   - 出典: [neurogolf-5480-41-current-rules-score/content.md](./neurogolf-5480-41-current-rules-score/content.md) (= "Public score is the promotion gate; local scoring can be misleading on parser-sensitive graph rewrites")
   - manifest sha256 + 公開 LB で確定 score を artifact 化する運用が複数 kernel で見られる

---

## 注意事項

- 公開 kernel の score は LB の「公開 baseline 帯」を反映するが、private LB top の解法は通常 public 化されない
- 著作権は各 kernel author に帰属。当該コンペでの利用は Kaggle Terms of Service に従う
- pull した raw notebook は `<kernel-slug>/<title>.ipynb` または `.py` として残す (再現性のため)
- 上記 INDEX に「metadata only」と表示した 29 kernel は Phase 1 では content.md を作成せず、kernel-metadata.json のみ pull 済み (= Phase 2 で必要に応じて pull)
