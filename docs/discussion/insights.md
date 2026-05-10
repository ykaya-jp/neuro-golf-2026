# neurogolf-2026 戦略 insights

> 作成日: 2026-05-10
> 情報源: discussion (`docs/discussion/2026-05-10.md` の 41 topic / 82 中) + host dataset (`docs/research/host_datasets.md` の 400-task / 101,718-pair 自家分析)
> 取得方法は `2026-05-10.md` 冒頭参照 (Python API のみ動作)。`topics.json` に取得済 topic_id とタグ index あり。

<!-- W1 出力。W4 の host dataset 実測と統合。
     全 claim には URL 出典 (Kaggle URL) または `discussion/2026-05-10.md:line`、もしくは
     `docs/research/host_datasets.md:<line>` を必ず添える。 -->

---

## 1. データ駆動で判明した「真の主因」

### 1.1 `cost = MACs + memory + params` のうち、**LB 上位は cost を 1 まで下げて score 25 を稼いでいる** が、その大半は scorer (= `onnx-tool` 静的解析) のバグ exploit であり、true minimal NN ではない

| 軸 | 一般 baseline (= `[1,10,30,30]` をそのまま渡す empty model) | 上位 (exploit-driven) | **我々が目指すべき (clean static)** |
|---|---|---|---|
| cost (= MACs + memory + params) | ≥ 36,000 | **1** | 数百〜数千 |
| 1 task score | ~14.5 | **25.00** (上限) | 14-21 |
| カバー率 | 全 task (但し dynamic shape NG) | 任意 task に汎用適用可 | 真に小さい変換のみ |
| 全体 LB | < 5800 | **9538-10000** (一時的、5/4 fix で消滅) | (実測必要) |

**所見**: LB 9000+ は **ほぼ全て scorer exploit による artificial inflation**。代表例は (a) `Pad` の負次元 → memory < 0 (`discussion/2026-05-10.md:794-820`)、(b) `Compress + Resize/Gather/GridSample` で出力 shape `[0,10,30,30]` を fabricate して MACs=0 (`discussion/2026-05-10.md:336-360`)、(c) 未使用 `value_info` に `dim_value=-139737` を埋め込み memory を負にする (`discussion/2026-05-10.md:1056-1078`)、(d) dynamic shape (`dim_param`) を残して memory 計算を欺く (`discussion/2026-05-10.md:707-720`、Kaggle Agent audit で 392 task 中 **373 task** が `value_info` に symbolic dim を含んでいた)。

**真の主因は「最小の NN を設計する純粋なアルゴリズム競争」ではなく「scorer = `onnx-tool` 1.0.1 の静的解析バグを発見・回避する逆解析競争」** が現状。host も認知 (`discussion/2026-05-10.md:148-155`):
> "Several contestants have found serious, unresolvable issues in the third-party profiler ... we've resigned to **eliminating MACs** from the objective criterion altogether. Going forward, only *cumulative memory footprint* and *parameter count* will determine the cost of your networks." (5/4 update)

出典: `docs/research/host_datasets.md:80-90` (公式 metric 仕様) + `discussion/2026-05-10.md:148-202` (5/4 host 投稿) + `discussion/2026-05-10.md:686-740` (Kaggle Agent 9538 自己 audit)

### 1.2 Metric は **5/4 で大改訂**、MACs 削除 + 静的 shape 強制 + Constant のパラメータ計上 + 負次元拒否

| 日付 | 主な変更 | 出典 |
|---|---|---|
| 4/21 | submission/day=100、>30x30 task は除外、scorer が failing network を報告、`Pad` 負 memory 拒否、版固定 (`numpy=2.4.4 / onnx=1.21.0 / onnxruntime=1.24.4 / onnx-tool=1.0.1`) | [#693711](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693711) / `discussion/2026-05-10.md:24-35` |
| 4/30 | Constant ノードのパラメータ計上、static shape 強制 (symbolic dim → 0 点)、profiling idiosyncrasies 緩和、**前日に batch rescore** | [#695230](https://www.kaggle.com/competitions/neurogolf-2026/discussion/695230) / `discussion/2026-05-10.md:483-511` |
| 5/4 | **MACs を objective から削除**、`calculate_memory()` を ORT 検証 shape ベースに、positive tensor dim assert、`Longest Leader` 賞の起算を **5/6 00:00 UTC** にリセット | [#696953](https://www.kaggle.com/competitions/neurogolf-2026/discussion/696953) / `discussion/2026-05-10.md:148-202` |
| 5/4 以降 | まだ exploit 報告継続 (5/5 だけで `#697048 / #697059 / #697063 / #696365` の 4 件、いずれも email 通報、未公開) | `discussion/2026-05-10.md:1504-1517, 4722-4738, 3334-3364, 1033-1090` |

**所見**: 現行 cost = `params + cumulative_memory_bytes` のみ。exp001 の baseline 設計時は **MACs を最適化する誘惑を排除**、**memory bytes と param count を直接削る** 方向に専念。MACs 系 helper (`single_layer_conv2d_network` の MAC 計算等) は意味を失った。

### 1.3 上位陣の戦略多様性 (= 単一最適戦略 vs マルチ戦略)

| 戦略 | 代表例 | 公開度 |
|---|---|---|
| **A. AI agent 全自動** (codex / Claude / Kaggle Agent) | jiweiliu の 11-subagent / 1729-tool / 173-compact run、LB 794 (`discussion/2026-05-10.md:1371-1416`) | high (notebook 公開) |
| **B. AI 1 task ずつ手動指導 + Spox** | thisray (LB 4743、394/400 task) (`discussion/2026-05-10.md:1416-1481`)、kawingkelvin の Spox 紹介 (`discussion/2026-05-10.md:4184-4512`) | high |
| **C. 公開 ONNX pack を blend** | konbu17 の `neurogolf-2026-blended-401-v117` (LB 5331+、`docs/research/host_datasets.md:79`) / mpwolke / sigmaborov 等 | high (kaggle dataset 公開) |
| **D. scorer 逆解析 exploit** (= 一時的に LB 9538-10000 を達成、5/4 で消滅) | Kaggle Agent (cdeotte 等) `discussion/2026-05-10.md:686-740` | semi-public (audit) |
| **E. 無料 LLM + 人間 rule infer** | `#694287` (Gemini/Deepseek/Qwen 無料を組合せ) `discussion/2026-05-10.md:3199-3242` | high |

**戦略の収束は無し**。我々は **A (full agent) ではなく B+C のハイブリッド** が現実的: 公開 pack を seed として開始 → category/grid_size 別に AI が ONNX を再生成 → score 14-21 を狙う。D (exploit) は 5/4 fix 後 LB が**正常化済**だが、host 自身が「exploit 報告継続」と認知しており、**5/15-5/30 にもう一度大きな fix が走る可能性高い** (`discussion/2026-05-10.md:148-176`)。

---

## 2. ディスカッション横断で繰り返し言及されている事項

### 2.1 主催者 / Host が明示的に開示した制約・データ

- **禁止 operator**: `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`、加えて scorer 上 **`Sequence` 型と nonpositive tensor dim も実質禁止** (`discussion/2026-05-10.md:32-35, 491-495`、出典 [#693711](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693711))
- **環境固定** (5/4 時点): `numpy=2.4.4`, `onnx=1.21.0`, `onnxruntime=1.24.4`, `onnx-tool=1.0.1` (`discussion/2026-05-10.md:32-35`、[#693711](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693711))
- **30x30 超え task の扱い**: scorer 側で skip。raw json には大きい grid が残るが評価対象外 (`discussion/2026-05-10.md:1203-1248`、[#692621](https://www.kaggle.com/competitions/neurogolf-2026/discussion/692621))
- **Multi-input/output graph 拒否**、**Sequences/nonpositive dim 拒否** (5/4 update、`discussion/2026-05-10.md:148-202`、[#696953](https://www.kaggle.com/competitions/neurogolf-2026/discussion/696953))
- **`Longest Leader` 賞の起算リセット**: 5/6 00:00 UTC (5/4 大改訂による) (`discussion/2026-05-10.md:163-167`、[#696953](https://www.kaggle.com/competitions/neurogolf-2026/discussion/696953))
- **submission limit**: 100/day (4/21 から、`discussion/2026-05-10.md:24-25`、[#693711](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693711))
- **agent ジャイル賞**: 別枠で `IJCAI-ECAI 2026` 表彰、`#691461` welcome message (`discussion/2026-05-10.md:373-471`)
- **private benchmark 存在**: overview 記載、`#696569` で具体的に懸念表明あり ("a very few of my tasks scored legit scores but when validated locally against freshly generated synthetic pairs ... I can see it score less than 100% pass rate" — 出典 `discussion/2026-05-10.md:1620-1641`、[#696569](https://www.kaggle.com/competitions/neurogolf-2026/discussion/696569))

### 2.2 上位プレイヤー / 公開 notebook 著者が明かしている技法

- **Constant 化 + パラメータ削減**: 4/30 update 前は Constant のパラメータが計上されず、LLM が rule logic を Constant にどんどん畳み込んで score を稼いでいた。fix 後は **Constant も計上**、過去の最適化は逆効果になる場合あり (`discussion/2026-05-10.md:1642-1712`、[#693589](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693589))
- **Spox** (Quantco 製 ONNX builder) を使うと dtype/opset/graph 修正が容易。`onnx.helper` より agent も人間も読みやすい (`discussion/2026-05-10.md:4184-4512`、[#694845](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694845))
- **AI に rule を発見させるな、人間が rule を発見して AI に ONNX 化させる**: 例 task007 で reshape `(-1,3)` 一発で diagonal pattern が垂直線になる、これは LLM 単独だと辿り着かない。"Human + agent" が支配的 strategy (`discussion/2026-05-10.md:2332-2400`、[#694628](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694628))
- **memory profiling の挙動 (4/29 確認)**: input/output tensor は memory 計上から除外、各ノードは output buffer のみカウント。**中間結果 R を最小化、padding は背景 channel の 1-2 strip で十分** (`discussion/2026-05-10.md:2400-2440`、同 [#694628](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694628))
- **Compactness の floor**: empty `[1,10,30,30] → [1,10,30,30]` で memory=36,000、score ≈ **14.5** が「真面目に解いた」上限 (`discussion/2026-05-10.md:1713-1820`、[#694051](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694051))
- **task001 baseline 13.6 → 14.5+ への iter**: chagpt + Spox で「specification.md → make_onnx() → verify_network() の score を feedback → 改良」のループを 5 周まわすと安定する (`discussion/2026-05-10.md:1820-2235`、[#693280](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693280))
- **Expand op を活用**: pytorch の `repeat_interleave` 同様、メモリ allocation 不要で繰り返しを表現 (`discussion/2026-05-10.md:1786-1789`、[#694051](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694051))
- **Kaggle Agent の現状**: jiweiliu の Codex agent が 11 subagents / 1729 tools / 12-173 auto-compact triggers で 12 時間で LB 794 達成、3 日連続非停止運転 (`discussion/2026-05-10.md:1371-1416`、[#692571](https://www.kaggle.com/competitions/neurogolf-2026/discussion/692571))
- **個別 task の頂点スコア共有**: `#693022` で task001 16.39 → 21+ → 15.8 (4/22 fix 後の真の floor) の遷移が記録されている (`discussion/2026-05-10.md:3243-3333`、[#693022](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693022))

### 2.3 重大 caveat (engine bug / metric 変更 / late shake-up 警告 / data leak)

- **scorer = `onnx_tool` 1.0.1 は spec より strict**。`shape_infer` / `profile` で「ONNX spec 上は legal だが scorer がクラッシュする」op が複数。代表例: **`ArgMin` (実装漏れ)**, **`Where` + uint8**, **`TopK`**, 一部 `Compress` パターン (`discussion/2026-05-10.md:153-158, 4561-4685`、[#697079 message #3 / #695454 / #693204](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693204)). **submit 前に local で `score_network()` を必ず通せ**
- **dynamic shape は今や 0 点**。4/30 以降、`value_info` に `dim_param` か空 `dim_value` があると即座に 0 点。Kaggle Agent の audit (`discussion/2026-05-10.md:707-720`) では 392 中 **373 task が affected**。我々の baseline は **必ず `onnx.shape_inference.infer_shapes` → `is_statically_defined` を local で通す**
- **Constant 化の罠**: 4/30 fix で Constant のパラメータが計上された結果、過去 LLM が「Constant に畳み込んで score を稼ぐ」最適化をしていた task は **逆に score が下がる** (`discussion/2026-05-10.md:1656-1679`、[#693589](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693589))
- **batch rescore が複数回**: 4/22 / 4/30 / 5/4 の 3 回。submit_history.csv の score は最新 metric で再計算済 (`discussion/2026-05-10.md:34-35, 502-511, 197-202`)
- **Longest Leader 賞は metric rescore でリセット候補**: ユーザーから「rescore = new start にしてくれ」要望、host が **5/6 00:00 UTC を新起算点に設定** (`discussion/2026-05-10.md:163-167, 1635-1641`)
- **task ambiguity**: 一部 task は train pair から rule が一意に決まらない (`#698249`、3 reply のみだが host 認知済) (`discussion/2026-05-10.md:3434-3469`、[#698249](https://www.kaggle.com/competitions/neurogolf-2026/discussion/698249))
- **未公開 exploit が 5/5 時点で 4 件報告中**: `#697048 / #697059 / #697063 / #696365` いずれも email 通報、details 非公開。次の大規模 fix が直近に走る可能性あり (`discussion/2026-05-10.md:1033-1090, 1504-1517, 3334-3364, 4722-4738`)
- **`onnx-tool` 撤廃要望**: `#696953` message #3 で「`neurogolf_utils` は `onnx_tool.loadmodel/model_profile` を使うのを止めるべき。`ArgMax` は OK / `ArgMin` は NG など、scorer 都合で op が制限されてる」(`discussion/2026-05-10.md:189-202`)。host が将来 scorer を書き換える可能性

### 2.4 重要な観測可能性 / 副次情報の指摘

- **scorer は `valid_profile==False` で `(None,None,None)` を返し、Kaggle UI には "Error processing one or more onnx networks." とだけ出る。task 単位の診断は出ない**。`discussion/2026-05-10.md:4561-4685` ([#693204](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693204)) に **local validation recipe** あり: `onnx.shape_inference.infer_shapes` → `score_network()` を必ず通せ
- **AI コスト戦術**: `#694287` で「Codex/Claude pro なしでも、Gemini 無料 + Deepseek 無料 + Qwen 無料 + GitHub Copilot Student で十分回せる」(`discussion/2026-05-10.md:3199-3242`)。Pro plan 無くても勝負はできる
- **Kaggle Agent (= LLM agent) チームが LB top に複数**: cdeotte / pavelsavchenkov / takuyainoue / yeoyunsianggeremie 等が host に exploit を email 通報した名前リストに登場 (`discussion/2026-05-10.md:172-176`)。彼らは **scorer 逆解析 + 自動 ONNX 生成** を agent 内ループで回している
- **Notebook sharing cooldown 提案** (deadline 前の終盤に notebook 公開を凍結する提案、まだ host 採否未定) `discussion/2026-05-10.md:1572-1597` ([#691904](https://www.kaggle.com/competitions/neurogolf-2026/discussion/691904))

---

## 3. 戦略への直接含意

| # | 改善案 | 解決する gap | 出典 |
|---|---|---|---|
| 1 | **5/4 metric (= MACs 削除、static shape 強制、Constant 計上、`onnx-tool 1.0.1`) の neurogolf_utils を local でセットアップし、全 baseline を local で `score_network()` を通してから submit する** | 4/21-4/30 の旧 metric で稼いだ score が rescore で消える、scorer エラーで全 task が 0 点になる事故 | `discussion/2026-05-10.md:148-202` ([#696953](https://www.kaggle.com/competitions/neurogolf-2026/discussion/696953)) + `discussion/2026-05-10.md:4561-4685` ([#693204](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693204)) |
| 2 | **Spox (Quantco) を ONNX 構築 layer に固定**。`onnx.helper` より読み書きが楽で agent loop に乗る | LLM の ONNX コード生成が `onnx.helper` の冗長さでバグる | `discussion/2026-05-10.md:4184-4512` ([#694845](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694845)) |
| 3 | **公開 pack を seed (`konbu17/neurogolf-2026-blended-401-v117` LB 5331+ / `thisray` LB 4743 / `needless090/neurogolf-onnx-v31`) として task 別にハイブリッド化**。我々の clean baseline で勝てない task は外部 pack を採用 | 400 task をゼロから作る ROI が悪い | `docs/research/host_datasets.md:79-80` + `discussion/2026-05-10.md:1416-1481` ([#694370](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694370)) |
| 4 | **memory floor 14.5 は出力 `[1,10,30,30]` を維持する限り超えられない。grid_size 変化系 task (= 132/400 task、`docs/research/host_datasets.md:179`) で `Slice + Pad` の **小型出力 + memory bytes 削減**を採用すれば 18-21 帯に乗る** | empty model 14.5 floor、20+ 帯は exploit ばかりで真面目に到達した報告が乏しい | `discussion/2026-05-10.md:2400-2440` ([#694628](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694628)) + `docs/research/host_datasets.md:177-180` |
| 5 | **AI agent 単独で submit せず、人間が rule を発見し ONNX 化を AI に任せる**。AI 単独は task 構造の reshape trick (`(-1,3)` で diagonal → 垂直線等) を見落とす | full-auto agent は score 9-12 帯が天井、Kaggle Agent も exploit 抜きでは LB top に達しない | `discussion/2026-05-10.md:2332-2400` ([#694628](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694628)) + `discussion/2026-05-10.md:686-740` ([#694772](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694772)) |
| 6 | **`ArgMin / TopK / Where+uint8 / Compress` 等 scorer-poison op を avoid list 化、Spec-legal だが poison な op の許容リストを `local validation recipe` で管理する** | submit すると "Error processing" で全 task 0 点 | `discussion/2026-05-10.md:4561-4685` ([#693204](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693204)) + `discussion/2026-05-10.md:189-202` ([#696953](https://www.kaggle.com/competitions/neurogolf-2026/discussion/696953)) |
| 7 | **task category × grid_size_changed × Estimated_Complexity の 3 軸で task をクラスタリング、cluster 別の baseline NN テンプレ (1-layer conv / Slice+Pad / Gather+Transpose / etc) を持つ** | task 1 つずつ手作業すると ROI が悪く、400 task 完走に到達できない | `docs/research/host_datasets.md:155-180` (category 分布) + `discussion/2026-05-10.md:3199-3242` ([#694287](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694287)) |
| 8 | **5/15 前後にまた大規模 fix が走る前提で、submission を「exploit-free な baseline」と「exploit を含む可能性ある実験」の 2 系統に分け、前者を `Longest Leader` 賞 (= 5/6 起算) ターゲットの安全 submission として日々更新する** | 次の rescore で全部消える | `discussion/2026-05-10.md:148-202` ([#696953](https://www.kaggle.com/competitions/neurogolf-2026/discussion/696953)) + 5/5 時点で未公開 exploit 報告 4 件 (`discussion/2026-05-10.md:1033-1090, 1504-1517, 3334-3364, 4722-4738`) |
| 9 | **`Constant` のパラメータ計上 fix (4/30) を踏まえ、rule logic を Constant initializer に畳み込む手は使わない。代わりに `Slice / Gather / Expand / Tile` を組合わせて zero-param transformation を作る** | 旧 LLM workflow が Constant 畳み込みに最適化されてた gap | `discussion/2026-05-10.md:1656-1679` ([#693589](https://www.kaggle.com/competitions/neurogolf-2026/discussion/693589)) + `discussion/2026-05-10.md:1786-1789` ([#694051](https://www.kaggle.com/competitions/neurogolf-2026/discussion/694051)) |
| 10 | **private benchmark で 100% pass しない懸念に対応**: arc-gen-100k (262 pair / task) で **100% 正解** を local 検証しないと、private で score 落ちる risk あり (`#696569` で同様の経験談) | overfit 系 submission が private で死ぬ | `discussion/2026-05-10.md:1620-1641` ([#696569](https://www.kaggle.com/competitions/neurogolf-2026/discussion/696569)) + `docs/research/host_datasets.md:184-186` |

---

## 4. 重要な未検証事項

- [ ] **5/4 update 後 (= MACs 削除以降) の真の score floor は何か**: `[1,10,30,30] → [1,10,30,30]` empty model の score 14.5 は 4/30 metric ベース。5/4 metric ではどう変わるか実機確認していない。**Phase 1 で empty model を submit して baseline score を実測**
- [ ] **5/15-5/30 に host の次回 fix が来るか、来た場合に新 metric で何が変わるか**: 5/5 時点で 4 件の未公開 exploit 報告、host も「occasional bug fixes/rescores」と明言 (`discussion/2026-05-10.md:163-167`)。**毎日 forum を check** する体制
- [ ] **konbu17 / thisray / mpwolke の公開 ONNX pack を local `score_network()` で再評価したときの cost 分布**: 各 task で「真の clean static」/「dynamic shape exploit」/「Constant 系 exploit」のいずれに依存するかを task ごと判定。Phase 1 で必須
- [ ] **`onnx-tool` の op 互換マトリクス (= scorer-poison op リスト) の網羅**: 既知 (`ArgMin / TopK / Where+uint8 / Compress` 等) 以外にも未報告の poison op が残る可能性。**`#693204` の local validation recipe を実装**
- [ ] **task ambiguity (#698249) は private で問題になるか**: train pair から一意に rule が決まらない task は agent + arc-gen で over-fit しても private で外す可能性
- [ ] **`Longest Leader` 賞は本当に 5/6 起算でリセット済か**: host コメントは「effective start time」だが Kaggle 内部の集計は別途要確認

これらは Phase 1 統合所見 → exp001 設計の前に追加調査する。

---

## 5. データソース管理

| ソース | パス | 規模 | 用途 |
|---|---|---|---|
| host dataset (categorize) | `data/external/neurogolf-2026-task-transformation-library/` | 400 task | Primary_Category × Estimated_Complexity (戦略 #7) |
| host dataset (logic) | `data/external/logic-for-each-arc-task/` | 400 task | LLM-driven program synthesis の primary input |
| host dataset (arc-gen) | `data/external/the-arc-gen-100k-dataset/` | 400 file × ~262 pair | private benchmark proxy (戦略 #10) |
| 公式 comp data | `data/raw/task<NNN>.json` (400) + `data/raw/neurogolf_utils/neurogolf_utils.py` | 101,718 pair | submission 評価 + local validation |
| 公開 ONNX pack (高 LB) | `konbu17/neurogolf-2026-blended-401-v117` (1.96 MB / LB 5331+, 未取得) | 401 task | seed (戦略 #3) |
| 公開 ONNX pack (中 LB) | `thisray/neurogolf-4743-93-submission-task-table` (`#694370`) | 394 task | seed (戦略 #3) |
| discussion dump | `docs/discussion/2026-05-10.md` | 41 topic / 222 KB | 主情報源 |
| topic index | `docs/discussion/topics.json` | 41 topic | tag 別検索 |

---

## 6. discussion 自動巡回の限界

> orbit-wars `insights.md:152-158` で確立された fallback chain を継承、本コンペで再検証した結果を以下に記録。

- **Kaggle discussion ページは React SPA で WebFetch (200 OK だが本文空 HTML)**: WebFetch / curl の両方で content 取れない (確認済 2026-05-10、URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion 及び個別 topic URL https://www.kaggle.com/competitions/neurogolf-2026/discussion/692827)
- **CLI `kaggle competitions topics neurogolf-2026 -v` は 403 Forbidden** (= API endpoint `discussions.DiscussionApiService/ListTopics` に CLI 経由で叩くと拒否) — 2026-05-10 確認、`Kaggle CLI 2.1.2` で再現
- **Python API (`kaggle.api.competition_list_topics` + `competition_list_topic_messages`) は動作**: 本コンペで 82 topic 全件 listing + 41 topic message 取得に成功。**現在のところこれが唯一動く取得手段**
- 注: Python API 経由でも `author_name` フィールドは空文字で返る (= Kaggle 側の masking 仕様、解決手段なし)。発言者は **本文中の `@username` mention** から推定するしかない
- 取得 topic id がわかれば `kaggle.api.competition_list_topic_messages('neurogolf-2026', <topic_id>)` で `raw_markdown` を取得可能 (HTML content も `content` field で利用可)
- Meta-Kaggle ForumTopics は本コンペには未確認 (snapshot が古い可能性)、Phase 1 で必要なら検証
