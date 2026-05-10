# NeuroGolf 2026 — Host 公開 dataset 一覧と EDA

> Worker B / W4 (host-dataset-analyzer) 出力。2026-05-10 編集。
> **CRITICAL**: ~/.claude/CLAUDE.md "[2026-05-10] orbit-wars lesson" に基づき、本ファイルの 「Host dataset 一覧」セクションは `kaggle datasets list -s neurogolf-2026` の出力をそのまま貼ることが skill QG-1 の必須条件。EDA セクションは host dataset が存在する場合に最低 1 day 分の DL + 分布実測 (N≥1000 行) を含むことが QG-2 の必須条件。

<!--
このファイルは 2 段構成:
1. CLI 出力 (= Phase 0 で main agent が冒頭に貼る、上書き禁止)
2. EDA (= Phase 2 で W4 が分析結果を埋める)
-->

---

## 1. `kaggle datasets list -s neurogolf-2026` の出力 (Phase 0 取得)

```text
ref                                                          title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
-----------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
karnakbaevarthur/neurogolf-2026-task-transformation-library  Neurogolf 2026: Task Transformation Library          864793  2026-04-20 18:57:51.553000            229         25                1  
karnakbaevarthur/logic-for-each-arc-task                     Logic for each ARC task                             1407522  2026-04-19 11:04:40.910000            281         28                1  
konbu17/neurogolf-2026-blended-341                           NeuroGolf-2026 Blended 341-task Submission          1297843  2026-04-27 13:33:59.607000            139          4           0.4375  
konbu17/neurogolf-2026-blended-401-v117                      NeuroGolf-2026 Blended 401-task v117 (LB 5331+)     1964245  2026-04-27 13:44:01.407000            218          2           0.4375  
tonylica/google-neurogolf-2026-public                        google-neurogolf-2026-public                         973908  2026-04-24 16:05:51.687000             74          2            0.375  
needless090/neurogolf-onnx-v31                               NeuroGolf ONNX Solutions v31                         505367  2026-04-19 06:02:30.047000            165          2              0.5  
konbu17/neurogolf-2026-blend-source-v3-6-0                   neurogolf-2026 blend source v3.6.0                  1090553  2026-05-08 08:34:30.107000             24          0           0.4375  
jonathanchan/sub4971-98                                      sub4971_98                                          1177050  2026-05-01 03:53:33.843000             69          0           0.5625  
jonathanchan/sub4979-07                                      sub4979_07                                          1178391  2026-04-30 08:50:08.933000             63          0           0.5625  
franksunp/neurogolf-2026-v18-multi-source-artifact           neurogolf-2026-v18-multi-source-artifact            1162307  2026-05-04 13:30:16.620000             42          0            0.375  
franksunp/neurogolf-blended-v24                              neurogolf-blended-v24                               1126754  2026-05-04 23:13:36.433000             35          0        0.3529412  

```

取得日時: 2026-05-10 12:45:18Z

<!-- 出力が空 (= host が公開 dataset 出していない) 場合は、ここに「公開 dataset なし」と明示する。
     W4 はこの場合 EDA を skip し、QG-2 は skip 判定で OK。 -->

---



## 1.b 関連 dataset (`kaggle datasets list -s arc-agi`)

ARC-AGI 派生 dataset (ARC-GEN-100K 含む) も主催者の評価対象 (= ARC-AGI-1 + ARC-GEN-100K + 私的 benchmark) に含まれる。参考列挙:

```text
ref                                                          title                                              size  lastUpdated                 downloadCount  voteCount  usabilityRating  
-----------------------------------------------------------  -------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
karnakbaevarthur/logic-for-each-arc-task                     Logic for each ARC task                         1407522  2026-04-19 11:04:40.910000            281         28                1  
karnakbaevarthur/neurogolf-2026-task-transformation-library  Neurogolf 2026: Task Transformation Library      864793  2026-04-20 18:57:51.553000            229         25                1  
karnakbaevarthur/arc-agi-3-all-tasks-explanation             ARC-AGI-3 All Tasks Explanation                    2659  2026-04-30 10:26:32.227000             22          7                1  
arcgen100k/the-arc-gen-100k-dataset                          The ARC-GEN-100K Dataset                        5747875  2025-08-18 00:27:28.947000            951         14           0.8125  
pshikk/arc-agi-csv-data                                      ARC-AGI-CSV-DATA                                 296979  2024-06-15 11:46:35.627000             60          4        0.5882353  
seshurajup/golf-task-2-arc-agi                               golf-task-2-arc-agi                                4287  2025-08-01 07:27:45.907000             96          4       0.47058824  
poonszesen/arc-interactive-community                         arc-interactive-community                        757949  2026-03-25 17:55:10.447000             12          4                1  
muhammaddanyalmalik/golf-files-superb-400-complete           golf-files-superb-400-complete                   136185  2025-10-30 17:17:41.530000             31          9            0.875  
boristown/arc-agi-2                                          ARC-AGI-2                                        646572  2025-07-24 05:52:35.253000             78          1           0.4375  
karnakbaevarthur/arc-task-logic-labels                       ARC Task Logic Labels                              4594  2026-04-26 11:37:22.123000              7          1                1  
codemasterminds01/arc-prize-2026-arc-agi-2                   ARC Prize 2026 - ARC-AGI-2                       498708  2026-03-26 07:28:38.570000             43          1           0.3125  
happy1scientist/arc-agi-2-public-eval                        ARC-AGI-2 Public Eval                            102870  2025-08-25 05:37:15.607000             14          1           0.1875  
evanhislupus/arc-agi-dataset                                 ARC agi dataset                                  448805  2024-07-18 19:02:30.443000             12          1             0.25  
chakrabhuanavdeva/arc-agi-synthetic                          arc-agi-synthetic                               3234092  2026-02-14 14:50:38.197000              1          1           0.1875  
fedimser/arc-agi-code-golf-216-solutions                     ARC-AGI Code golf - 216 solutions                 16290  2025-08-09 03:39:36.210000             46          1            0.375  
wawanbsetyawan/arc-agi                                       arc agi                                          320775  2024-09-15 15:22:24.230000             21          0            0.375  
nikhilkumarbharti/arc-agi-1-0-2                              ARC-AGI 1.0.2                                    480890  2024-08-05 06:28:12.840000             33          0           0.5625  
ravikaash/qor-arc3-agent                                     QOR ARC-AGI-3 Agent                             2651733  2026-03-26 18:16:26.203000              6          1           0.1875  
offeibekoe/arc-agi-africa                                    ARC-AGI Africa                                   264897  2025-07-27 20:14:48.387000              4          0       0.29411766  
bytestorm/arc-agi-datasets-legacy                            ARC AGI Datasets (Legacy)                        661745  2025-10-02 12:08:30.113000             14          0            0.625  

```

## 2. ダウンロード済 dataset の inventory

| Dataset slug | 規模 | パス | DL 日 | License | 用途 |
|---|---|---|---|---|---|
| `karnakbaevarthur/neurogolf-2026-task-transformation-library` | 845 KB / 400 tasks | `data/external/neurogolf-2026-task-transformation-library/` | 2026-05-10 | CC BY 4.0 (推定) | 各 task の **categorization + Estimated_Complexity** (Spatial/Object/Color/Pattern, complexity 1-5) — 戦略の重み付けと baseline NN 構造選択の guide |
| `karnakbaevarthur/logic-for-each-arc-task` | 1.34 MB / 400 tasks | `data/external/logic-for-each-arc-task/` | 2026-05-10 | CC BY 4.0 (推定) | 各 task の **transformation logic 自然言語説明** — LLM-driven program synthesis の primary input、ONNX 設計の根拠 |
| `arcgen100k/the-arc-gen-100k-dataset` | 5.48 MB / 400 file (`<task_uuid>.json`) | `data/external/the-arc-gen-100k-dataset/` | 2026-05-10 | (要確認) | 各 task の arc-gen pair 集合 (= comp data の `arc-gen` field と同等)。validation 用 |
| (`konbu17/neurogolf-2026-blended-401-v117` 1.96 MB LB 5331+ 公開 submission) | (未取得、Phase 1 で必要に応じ DL) | - | - | - | 上位陣の現物 ONNX submission |
| (公式 `kaggle competitions download -c neurogolf-2026`) | 5.59 MB / 400 task json + `neurogolf_utils.py` | `data/raw/` | 2026-05-10 | comp rules | **本コンペの主データ**: train/test/arc-gen の三分割が json で揃う |

---

## 3. EDA — comp data 全体統計 (= 仮説の方向性確証、N=101,718 pair で実測)

### 3.1 schema / 列構成

| file | rows | columns | 主要 column 例 |
|---|---|---|---|
| `data/raw/task<NNN>.json` | 400 task file | dict 3 keys | `train`: list[pair], `test`: list[pair], `arc-gen`: list[pair]. 各 pair は `{input: list[list[int 0-9]], output: list[list[int 0-9]]}` |
| `data/external/neurogolf-2026-task-transformation-library/arc_primitives.json` | 400 task | dict | per task: `Spatial_and_Geometric`, `Object_Based`, `Color_and_Logical`, `Pattern_Recognition`, `Primary_Category`, `Grid_Size_Changed`, `Estimated_Complexity` |
| `data/external/logic-for-each-arc-task/arc_explanations.json` | 400 task | dict | per task: 自然言語 explanation (3-7 文) |

### 3.2 主要分布 (= 仮説の方向性を data で確証)

再現コマンド:
```bash
uv run python -c "
import json, statistics as st
from pathlib import Path
from collections import Counter
tasks = sorted(Path('data/raw').glob('task*.json'))
in_h, in_w, out_h, out_w, colors = [], [], [], [], Counter()
for t in tasks:
    d = json.loads(t.read_text())
    for split in ('train', 'test', 'arc-gen'):
        for p in d.get(split, []):
            in_h.append(len(p['input'])); in_w.append(len(p['input'][0]) if p['input'] else 0)
            out_h.append(len(p['output'])); out_w.append(len(p['output'][0]) if p['output'] else 0)
            for row in p['input']: colors.update(row)
print(f'N={len(in_h)} pairs')
"
```

#### task 構造 (per task)

| 軸 | 値 |
|---|---|
| task 数 | 400 |
| **総 pair 数** | **101,718** |
| train pairs / task (median) | 3.0 (= ARC-AGI-1 公式 train) |
| test pairs / task (median) | 1.0 (= ARC-AGI-1 公式 test) |
| arc-gen pairs / task (median) | 262.0 (= ARC-GEN-100K 由来、評価で必須) |

#### grid size 分布 (= NN 設計上もっとも重要)

| 軸 | p50 | p90 | max |
|---|---|---|---|
| input height | 10.0 | 20 | 70 |
| input width | 10.0 | 20 | 75 |
| output height | 9.0 | 19 | 42 |
| output width | 10.0 | 19 | 38 |

**所見**:
- median 10x10、p90 20x20、max は input 70x75 / output 42x38 が出現する。**input は仕様の 30x30 を超える sample がある** (= arc-gen で生成された大型 grid。ただし入力は `[1,10,30,30]` tensor に zero-pad されるので NN は 30x30 想定で書く)。
- **大半の task は 10x10 前後の小さい grid で動く** → 1-2 layer の小型 conv で表現可能なロジックが支配的。NN cost (params + bytes) を最小化するには grid size に応じた dynamic 設計が hugue 利得を生む。

#### color usage 分布 (input pixel 数、計 100 億級)

| color | pixel count | 全体比 |
|---|---|---|
| 0 (clear/black) | 10,441,024 | **61.8%** (= 背景支配) |
| 1 | 836,842 | 5.0% |
| 2 | 873,981 | 5.2% |
| 3 | 677,697 | 4.0% |
| 4 | 584,784 | 3.5% |
| 5 | 996,774 | 5.9% |
| 6 | 601,643 | 3.6% |
| 7 | 526,601 | 3.1% |
| 8 | 777,684 | 4.6% |
| 9 | 566,008 | 3.4% |

**所見**: color 0 が **80% 弱を占める**。one-hot encoding 後 `[1, 10, 30, 30]` の channel 0 は 80% sparse、他 9 channel は **5-10% 程度**。**channel 5 (灰色) も多い** (5.9%) のは hypothetical task 例が channel 5 を ”clear" として扱う設計と整合 (Overview の例: `if channel_in == 5: return 0.0`)。

#### task category 分布 (arc_primitives.json 由来)

| Primary Category | task 数 | 比率 |
|---|---|---|
| Pattern_Recognition | 160 | 40.0% |
| Object_Based | 158 | 39.5% |
| Spatial_and_Geometric | 47 | 11.8% |
| Color_and_Logical | 35 | 8.8% |

#### Estimated_Complexity 分布 (1-5 scale)

| complexity | task 数 |
|---|---|
| 1 | 1 |
| 2 | 34 |
| 3 | 34 |
| 4 | 105 |
| 5 | 68 |
| 6 | 96 |
| 7 | 61 |
| 8 | 1 |

**所見**:
- **400 task すべてが categorize されている** (= 上位陣も含むコミュニティが解析済)。Spatial_and_Geometric が支配的なら conv 系 1-2 layer で大半を解ける可能性
- complexity 1-2 が多ければ **1-layer NN で 100+ task が稼げる**、complexity 5 は探索が必要
- Grid_Size_Changed = 131 / 400 task (33%) → 過半数で grid size が変わる (= tiling, cropping, scaling 系)

### 3.3 host dataset と train/test の関係

- `arc-gen-100k` の各 file は **comp data の `arc-gen` field と同一**だが file 名は `<arc_task_uuid>.json` (= ARC-AGI-1 の元 ID)。task001 等の comp 命名との対応は別途 mapping 必要 (W2 で確認)
- `arc_primitives.json` / `arc_explanations.json` は **400 task すべて** に対して情報が揃っており、test data と直接は重ならないが、test の input/output に対しても同 logic で動く NN を設計する必要がある (= overfitting 不能)

### 3.4 重大 caveat

- **競技 metric の version**: `data/raw/neurogolf_utils/neurogolf_utils.py` の Version History (2026-05-06 / 05-04 / 04-30) を必ず確認。MACs は metric から外れた、scalar parameter は unit cost、duplicate node names のバグ修正、`kernel_time` 文字列禁止、Multi-input/output graph 禁止、Sequences/nonpositive tensor dim 禁止、Loop/Scan/NonZero/Unique/Script/Function operator 禁止
- **private benchmark** がある (Overview 記載): `task001-400` で overfit する submission は private で score 落ちる。各 task の transformation を **本質的に正しい NN** で実装する必要がある (= ARC-GEN-100K の 262 pair で test して 100% 正解、ただし全色置換や scale 変化に robust)
- **arc-gen-100k の license** は要確認 (kaggle dataset page で license 情報)

---

## 4. 戦略への直接含意

| # | 仮説 | dataset 由来の確証 | Phase 1 への含意 |
|---|---|---|---|
| 1 | 大半 (>50%) の task は 1-layer または 2-layer 小型 conv で解ける | grid p50=10x10、color 0 が 80%、Estimated_Complexity 1-3 が支配的 | **baseline は 1-2 layer conv の手書き重み**、`single_layer_conv2d_network(weight, kernel_size=3)` (Overview の helper) を utility 化 |
| 2 | task ごとに transformation logic が自然言語で記述されている → LLM-driven program synthesis 可能 | `arc_explanations.json` 400 task カバー | **LLM (Claude / GPT) を per-task agent として活用**、説明 → 候補 NN コード → ONNX export → validate のループ |
| 3 | category ごとに NN architecture を共通化できる | `arc_primitives.json` の Primary_Category | **category 別 baseline** を作り、各 category 内で grid_size + complexity に応じた variant |
| 4 | input は 30x30 静的 shape 必須、grid size 変化系 task は output shape を decoder で出す | comp 仕様 + Grid_Size_Changed が過半 | **encoder-decoder 構造** (= 30x30 input → small bottleneck → max 30x30 output) を category に応じて分離 |
| 5 | 上位陣は既に 401 task 分の blended submission (LB 5331+) を公開 | konbu17/neurogolf-2026-blended-401-v117 | Phase 1 で DL し、上位の architecture を逆解析。我々の baseline と diff を取る |

---

## 5. 未取得 / 巨大すぎて DL skip した dataset

| Dataset | 規模 | skip 理由 | 取得が必要になる条件 |
|---|---|---|---|
| `konbu17/neurogolf-2026-blended-401-v117` | 1.96 MB | 優先度 high だが Phase 1 の kernels manifest 取得時にまとめて DL | exp001 設計後、上位 submission 逆解析で参照 |
| `karnakbaevarthur/arc-task-logic-labels` | 4.6 KB | 既に primary category は arc_primitives.json で取得済 | より詳細な label 情報が必要になったら |
| `chakrabhuanavdeva/arc-agi-synthetic` | 3.2 MB | overlap で arc-gen-100k で代替 | private benchmark を意識した data augmentation で必要なら |
| `wb55l-nemomini-fulleval-tuned` | 9 GB | LLM 重み、本コンペ範囲外 | (取らない) |
