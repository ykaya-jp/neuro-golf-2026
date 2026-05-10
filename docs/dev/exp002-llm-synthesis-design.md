# exp002 — LLM-driven Program Synthesis Pipeline 設計 note

> 2026-05-10 編集 (= exp002 implement 同時)。
> spec: `.criteria/kaggle-neurogolf-2026-exp002-llm-synthesis.yaml`
> 候補 2 (= `docs/strategy/exp001-design.md` 候補 2) の段階 1 実装。

---

## 設計原理

ARC-AGI を最小 ONNX で解く問題は **「自然言語 logic → Python `weight_fn` → ONNX Conv weight」 の三段階圧縮** と見做せる。 SOAR 2025 ([source](https://arxiv.org/abs/2507.14172)) の sampling → refinement → hindsight learning loop を **Python DSL でなく ONNX weight_fn** に application。

主催者 helper `single_layer_conv2d_network(weight_fn, kernel_size)` が ONNX 構造を担保するので、LLM が出力すべきは weight_fn 1 関数のみ。 ONNX 知識ゼロでも解ける。

---

## 安全性: AST whitelist sandbox

LLM 出力は untrusted。 `extractor.py` で以下を強制:

- 文字列 level: `eval / exec / __import__ / open / subprocess / os / __builtins__` 等の forbidden token を `\b` regex で reject
- AST level: 許可 node の whitelist (= Module / FunctionDef / If / IfExp / Compare / BoolOp / BinOp / UnaryOp / Name / Constant / Tuple / List / Return / Subscript)
- `ast.Call` (= 関数呼出) と `ast.Attribute` (= attr access) は **明示的に reject** (= sandbox 破りの一般経路を塞ぐ)
- 引数 signature 厳格 check (= `(channel_out, channel_in, kernel_coord)` 順序 + 名前)
- compile + exec は `__builtins__: {}` の restricted scope で実行

**詳細**: `src/neurogolf_2026/synthesis/extractor.py:_validate_ast`

これで LLM が import / file I/O / network 等の副作用を一切起こせない。test (`tests/test_synthesis.py::TestExtractorSafety`) で 8 種の forbidden パターンが reject されることを確認済。

---

## Pipeline 構成

```
arc_explanations.json[task_id]
        ↓ prompts.build_prompt()
   LLM client.synthesize()
        ↓ raw text (LLM 出力)
   extractor.extract_weight_fn()
        ↓ Callable (sandbox 通過済)
   runner.run_weight_fn()
        ↓ ONNX 化 + score_network() + banned op check
   RunResult (accepted: bool)
        ↓ pipeline.run_synthesis()
   outputs/synthesis/<run_id>/results.json
```

各段階の責務は disjoint。 client は LLM 呼出のみ、 extractor は安全性、 runner は ONNX 制約 + functional correct、 pipeline は集約。

---

## Client 抽象化

| client | 状態 | 用途 |
|---|---|---|
| `DummyClient` | 実装済 | deterministic stub (= task276 reference + zero template)。 test で end-to-end pin |
| `AgentDispatchClient` | interface のみ | Claude Code Agent tool 経由。本 plan では NotImplementedError を上げて開発者が手動で relay する運用。 詳細は §「実 LLM 試行ログ」 |
| `AnthropicClient` | 未実装 | 次 plan: API key 管理 + cost 制御 |
| `OpenAIClient` (= Codex) | 未実装 | 次 plan: 多 model diversity 用 |

実 API client を plug-in する際は `synthesize(task_id, explanation, kernel_size) -> str` を満たすだけで pipeline は変更不要。

---

## 実 LLM 試行ログ (= AC-7)

### 2026-05-10 試行 1: task276 — Claude (general-purpose subagent via Agent tool)

prompt (= `prompts.build_prompt('task276', explanations['task276'])`) を Claude Code の `Agent(subagent_type='general-purpose')` に渡し、 `weight_fn` を 1 回 dispatch。

**LLM 出力** (= `outputs/synthesis/agent-task276-trial/task276_raw.txt`):

```python
def weight_fn(channel_out, channel_in, kernel_coord):
    if kernel_coord != (0, 0):
        return 0.0
    if channel_out == 6:
        return 0.0
    if channel_out == 2:
        if channel_in == 2 or channel_in == 6:
            return 1.0
        return 0.0
    if channel_out == channel_in:
        return 1.0
    return 0.0
```

**Pipeline 実行結果** (= `outputs/synthesis/agent-task276-trial/results.json`):

| 軸 | 値 |
|---|---|
| extractor | ✓ AST sandbox 通過 |
| runner | ✓ ONNX 化成功 (572 byte) |
| constraint_violations | `[]` |
| arc_agi | 4 / 4 pass |
| arc_gen | 262 / 262 pass |
| functional_correct | **true** |
| scorer_ok | **true** |
| cost | 100 (params) + 0 (memory) = 100 |
| **score** | **20.395** |
| **accepted** | **true** |

**所見**: 実 LLM が dummy reference と一致する weight_fn を 1-shot で生成。 prompt のうち「kernel_size = 1 / position-dependent branching 不可 / sparse weight 推奨」が effective に効いた。 channel 6 を 0 に消す + channel 2 が color 6 を吸収する 3 段論法 を LLM が正しく演繹。 これは **後続 task でも同 prompt template で動く可能性が高い** ことを示唆。

---

## 次 plan の入口

### r-exp003: Anthropic API client + 全 400 task 走破

- API key 設定 (`ANTHROPIC_API_KEY` env)
- `AnthropicClient.synthesize` 実装 (= `anthropic.messages.create` 呼出 + retry + rate-limit + token cost log)
- `outputs/synthesis/anthropic-<run_id>/` に全 task の raw + parse 結果を保存
- 期待 success rate: 30-50% (= 簡単な color-mapping / spatial primitive で 100-200 task)
- API cost 見積: 400 task × 1500 input token + 200 output token × $0.015/1K (= Sonnet) ≈ $25/iter

### r-exp004: Refinement loop (= SOAR pattern)

- failed task に対し error feedback (= "score_network reported wrong = 142 / 262") を LLM に return
- 1-2 iteration で 30% → 50% に押し上げ期待
- iteration ごとの cost を追跡

### r-exp005: 多 client ensemble (Claude + Codex)

- Codex (`openai-codex-plugin`) を 2nd client に
- task ごとに両 client が解いた weight_fn を併走、 cost が小さい方 (= functional correct 前提) を採用
- diversity で coverage 拡大

### 設計決定が pending な事項

- **kernel_size > 1 の prompt template** — 現状 1×1 のみ、 3×3 / 5×5 用 prompt は別途必要 (= rotation / reflection / pattern detection を要する task 用)
- **multi-layer ONNX support** — `single_layer_conv2d_network` だけでは表現できない task は別 helper を実装する必要 (= 例: 1x1 conv → 3x3 conv → 1x1 conv の 3 stage architecture)
- **scorer-poison op の発火 risk** — runner の `_validate_onnx` で banned op を reject するが、 `single_layer_conv2d_network` は Conv のみ生成するので現実には発火しない。 multi-layer に拡張時に再評価必要

---

## 参照

- spec: `.criteria/kaggle-neurogolf-2026-exp002-llm-synthesis.yaml`
- 実装: `src/neurogolf_2026/synthesis/`
- test: `tests/test_synthesis.py` (15 件)
- CLI: `scripts/run_synthesis.py`
- exp002 baseline follow-up: `docs/dev/baseline-followup-todo.md`
- 戦略 design: `docs/strategy/exp001-design.md` (候補 2)
