# baseline (exp001) 完了後の follow-up TODO

> Phase 4 reviewer (AC-6 manual_review) 出力 + 自己検証で得た改善点。
> baseline submit (= AC-4) と LB 反映 (= AC-7) 後に着手する次 plan の入力。

## 即修正済 (本 commit に含む)

- [x] `validate.py:80-82`: `OrtException` 握り潰しを `runtime_errors` リストで context 保持
- [x] `validate.py`: `validate_task` の dead arg `check_scorer_poison` を削除 (= 常に scorer 走行)
- [x] `validate.py` の `_verify_pairs` 戻り値を `(right, wrong, runtime_errors)` に拡張、`result["errors"]` に注入

## 次 plan で着手

### r-1: Exception capture に traceback 同梱

- 場所: `src/neurogolf_2026/validate.py:118-122, 156-157`
- 問題: `repr(e)` のみで stack trace 欠落、post-mortem に必要な情報が消える
- 直し方: `import traceback; result["traceback"] = traceback.format_exc()` を併記
- 優先度: medium (= debug 効率化、機能性に影響なし)

### r-2: `_helpers.py` の lazy load 化

- 場所: `src/neurogolf_2026/networks/_helpers.py:34`
- 問題: module top-level で `_load_neurogolf_utils()` が走り、`data/raw/neurogolf_utils/` 不在の環境で import error
- 直し方: PEP 562 `__getattr__(name)` で lazy load、または `def _build()` 内で初回呼び出し
- 優先度: medium (= CI / fresh clone での usability 向上)

### r-3: `_validate_onnx` negative-path unit test

- 場所: `tests/test_submission_pipeline.py`
- 問題: positive 系 (= 全 task pass) のみで validator 自体の bug を検知できない
- 直し方: synthetic ONNX (banned op `Loop` 含む / `kernel_time` 文字列含む / multi-input graph) を作って `_validate_onnx` の return が期待通りに violation を返すことを assert
- 優先度: low (= validator は単純実装で bug risk 低、post-baseline で OK)

## 設計上の懸念 (= 次 plan の構造原理を考えるときに思い出す)

### d-1: `score_network` 戻り値の Version History 依存

- 場所: `validate.py:145` で `(memory, params) = score_network(...)` と仮定
- 問題: neurogolf_utils.py の Version History が変わると tuple 順序 / 数 が変わる可能性
- 対応: 5/15 metric 改訂後、neurogolf_utils.py を再 fetch して signature 確認、test を追加

### d-2: `score_network` の memory が常に 0 を返す観測

- 観測: task276 (= 1x1 conv 100 params) で `memory_bytes: 0`
- 原因 (推測): `score_network` は profiler trace 経由で **activation tensor の peak memory** を計算するが、1x1 conv は activation を保持しない (= in-place で完結)
- 含意: per-task NN を **activation 軽量化** (= 1x1 conv で済ませる、grouped conv 不使用) で memory_bytes を 0 に抑えられる可能性
- 次 plan の構造原理に取り込む価値あり

### d-3: Top 5 by score 出力が誤解を招く

- 場所: `validate.py` の `--all` モードで Top 5 by score を表示
- 問題: fallback (= zero conv) も score 20.39 を計算するが、実際の LB では functional incorrect → 0 点
- 対応: Top 5 を `functional_correct: true` の task に絞って表示

## やることリスト (=「優勝してきて」への次の一手)

1. **AC-4**: `kaggle competitions submit -c neurogolf-2026 -f submissions/submission.zip -m "exp001 baseline: task276 functional correct"` → LB 反映確認 (= AC-7)
2. submission_id と public_score を `submissions/.history.csv` に追記
3. exp001 verify (`/verify kaggle-neurogolf-2026-baseline`) で形名参同 Verify 段を通す
4. 次 plan: **`exp002 = simple-task batch (Complexity 1-2 の 22 task hand-craft + INT8 quantize)`** で基本盤を作る (= 候補 1 の段階拡張)
5. その後: 候補 2 (LLM-driven program synthesis) を Complexity 4-8 task (= 全 400 の 82.75%) に向けて投入
