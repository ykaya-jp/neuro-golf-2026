# exp004 Design — Blending Bootstrap + 中期 Cost-aware Refinement Loop

> 2026-05-11 編集。 spec: `.criteria/kaggle-neurogolf-2026-exp004-blending-bootstrap.yaml`
> 短期 (本 plan = blending) + 中期 (= multi-layer + Cost-aware Refinement) を統合した設計 note。
> IJCAI 2026 talk invitation 候補となる新規性を文書化。

---

## 短期: Blending Bootstrap (実施済、 LB 20.39 → 4917.67)

### 実装

`src/neurogolf_2026/blending/{source_pool, argmin, build_blended}.py` の三段。

- **source_pool**: 7 公開 source の inventory (= konbu17-may8, afr1ste-5480-41, jonathanchan-ngc26, magmacot-new-blending, konbu17-blended-401-v117, karnakbaevarthur-task-library, karnakbaevarthur-logic) を path template で管理
- **argmin**: per-task で **ONNX file size を cost proxy** に最小 source を選ぶ + `_validate_onnx` で banned op gate
- **build_blended**: 自前 registry と argmin 結果を統合、 zip 構築。 **同 cost なら自前優先** (= 5/15 metric 改訂時の hedge)

### 実績

submission 52520345 = `public_score 4917.67` (LB rank ~320 / 1095 teams = top 30%)。
source 別 採用数:
- konbu17-may8: 219 task
- karnakbaevarthur-task-library: 82
- afr1ste-5480-41: 58
- magmacot-new-blending: 40
- karnakbaevarthur-logic: 1
- 自前 (task276): 0 件 (= konbu17-may8 が 217 byte で自前 572 byte より小)
- fallback: 0

LB top 1 yiheng 7312 まで gap 2394、 8000 まで 3083。

### 限界 / risk

- **5/15 metric 改訂で公開 zip が崩壊する risk** (W1 finding `docs/discussion/insights.md`)、 これが起きたら 自前 task276 component (= 1 task の hedge) しか残らない
- **Kaggle 規約上 license 監査未完了** (= critic Objection 10)、 主催者 ban announce で即撤回必要
- **学術的 contribution ゼロ** = IJCAI talk invitation 候補にならない

→ 中期 plan で 自前 拡張する必要。

---

## 中期: Multi-layer ONNX Helper + Cost-aware Refinement Loop (= 新規性、 IJCAI talk 候補)

### 課題

公式 helper `single_layer_conv2d_network(weight_fn, kernel_size)` は **1 layer Conv2D 1 種類** しか生成できない。 しかし ARC-AGI の 400 task のうち、 真に 1 layer conv で解けるのは **context-free color substitution** のみ (= 数件、 task276 含む)。 大半の task は object-aware / structure-aware 変換が必要、 multi-layer NN が要る。

しかし naive な multi-layer は **cost が爆発** (= layer 数 × params)。 NeuroGolf は **「正しく解く」 + 「最小 NN」** の二重最適化なので、 layer 数を **task ごとに必要最小限** に抑える search が必要。

### 提案手法 1: Multi-layer ONNX Helper (`multi_layer_conv2d_network(layers)`)

```python
@dataclass
class LayerSpec:
    weight_fn: Callable
    kernel_size: int
    activation: Literal["none", "relu", "sigmoid"] = "none"

def multi_layer_conv2d_network(layers: list[LayerSpec]) -> onnx.ModelProto:
    """N 層 Conv → optional Activation → Conv → ... の ONNX 生成."""
```

- 1 layer ⊂ 2 layer ⊂ N layer の入れ子構造 (= LayerSpec list の長さで制御)
- activation の場所も指定可能 (= linear のみ vs ReLU 挿入)
- 出力は static shape `[1, 10, 30, 30]` 固定

これで **edge detect (3×3 conv) → object isolate (ReLU) → recolor (1×1 conv)** のような典型 ARC pattern が表現可能に。

### 提案手法 2: Cost-aware Refinement Loop (= 新規性 main)

LLM-driven program synthesis の sample → refine → hindsight learning loop ([SOAR 2025](https://arxiv.org/abs/2507.14172)) を **MDL prior + cost feedback** で拡張:

```
for task in tasks:
    architecture = INITIAL_GUESS  # 例: 1 layer 1×1 conv
    while True:
        weight_fn = LLM.synthesize(task, architecture, prior_attempts)
        result = run_weight_fn(task, weight_fn)
        if result.functional_correct and result.cost < best_cost:
            best = (architecture, weight_fn, result.cost)
        # cost 制約を hint としてフィードバック
        if result.cost > THRESHOLD:
            architecture = LLM.suggest_smaller(architecture, "cost too high")
        elif not result.functional_correct:
            architecture = LLM.suggest_richer(architecture, "incorrect")
        else:
            break
```

- **MDL prior**: cost = params + memory_bytes は **Solomonoff 確率の log** に対応 (= shortest program ⇔ Occam's razor)。 Refinement loop は **MDL gradient descent** と等価
- **`prior_attempts`**: 失敗 weight_fn の error log を LLM に return (= hindsight learning)、 N iteration で functional correct + minimal cost に収束
- **`THRESHOLD`**: log scale で減らす (= 最初は 10000、 次 5000、 次 2500、 ... 各 iteration で halve)
- **Pareto front**: 早期 stop で functional correct な最初の解 (= cost 大) を保管、 refinement で 漸進改善、 改善余地ない epoch 数で打ち止め

### 提案手法 3: Multi-architecture Voting (= 失敗 task への safety net)

1 layer / 2 layer / 3 layer / ConvTranspose 等の **複数 architecture を同時試行**:

```
for task in tasks:
    candidates = []
    for arch in [LAYER1_K1, LAYER1_K3, LAYER2, LAYER3, CONVT_K3]:
        weight_fn = LLM.synthesize(task, arch)
        result = run_weight_fn(task, weight_fn)
        if result.functional_correct:
            candidates.append((result.cost, weight_fn, arch))
    if candidates:
        return min(candidates, key=lambda x: x[0])
    return None
```

- 同 task に対し **複数 architecture を競合**、 最小 cost の解が勝つ
- 失敗時は次の architecture へ自動 escalate
- LLM の創意 (= 各 architecture で異なる weight_fn を提案) で coverage 拡大

### 期待 lift

|  source | task per-task avg cost | total score |
|---|---|---|
| 公開 blending (現状) | ~5000 (= 4917.67 / 400 ≈ 12.3 / task = ln(cost) ≈ 13.7 = cost ≈ 880,000、 ただし fail task の 0 点も含むので per-task avg は実 11.0 / task = ~12.3) | 4917.67 |
| + Refinement Loop (= 1 layer 系の cost を log で削る) | -300 task で cost 100-1000 帯 | 5500-6500 |
| + Multi-layer (object-aware task をカバー) | -100 task で cost 300-2000 帯 | 6500-7500 |
| + Multi-architecture voting | 残り 50 task で escalate | **7500-8500** (= 優勝圏) |

### IJCAI talk invitation 候補化

新規性 4 点:
1. **MDL prior に基づく cost-feedback refinement loop** (= ARC + NN compression の交点)
2. **Multi-architecture voting via LLM diversity** (= 過去 SOAR は単一 DSL、 我々は ONNX architecture 空間)
3. **Static-shape unroll で recursion を表現** (= TRM の 16 unroll を NeuroGolf の no-Loop 制約に application)
4. **Synthetic data augmentation + LLM refinement の組合せ** (= ARChitects 2024 + SOAR 2025 hybrid)

これらは ARC Prize 2024 / 2025 の review report ([arxiv.org/2412.04604](https://arxiv.org/2412.04604), [arxiv.org/2601.10904](https://arxiv.org/2601.10904)) で言及されている **refinement loop paradigm の延長**。 NeuroGolf 特化は cost 軸に MDL を載せた点が独自。

---

## Plan 分割案 (= 中期 implementation roadmap)

| plan | scope | 期待 lift | 開発期間 |
|---|---|---|---|
| **exp005**: multi_layer_conv2d_network helper | 2-3 layer ONNX 生成、 prompt template 拡張、 dummy + Agent test | +0 (基盤) | 3-5 日 |
| **exp006**: Cost-aware Refinement Loop (single-arch) | 1 layer に対し cost feedback + retry の loop、 task276 系 5-10 task で実証 | +50-100 | 2-3 日 |
| **exp007**: Multi-architecture voting | 5 architecture を per-task 競合、 simple-Spatial 系 30-50 task カバー | +500-1000 | 1-2 週 |
| **exp008**: 全 400 task 走破 + Anthropic API integration | API key 設定 + 大規模 dispatch + cost log | +2000-3000 | 2-3 週 |
| **exp009**: 5/15 metric 改訂対応 + 公開 blending 段階剥がし | 自前 ratio を 80%+ に、 公開依存 < 20% | hedged | 継続 |

合計射程: **5500 → 7500-8500 で優勝圏**。

---

## 参照

- 短期 spec: `.criteria/kaggle-neurogolf-2026-exp004-blending-bootstrap.yaml`
- exp001-design.md (= 戦略原案)
- exp002-llm-synthesis-design.md (= 候補 2 framework prior art)
- ARC Prize 2024 TR: https://arxiv.org/html/2412.04604v2
- ARC Prize 2025 TR: https://arxiv.org/html/2601.10904v1
- SOAR 2025: https://arxiv.org/abs/2507.14172
- TRM 2025: https://arxiv.org/abs/2510.04871
- CompressARC: https://github.com/iliao2345/CompressARC
- MDL principle (Solomonoff prior): Vitanyi & Li, "An Introduction to Kolmogorov Complexity"
