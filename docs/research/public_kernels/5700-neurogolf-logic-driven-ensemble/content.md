# 5700-neurogolf-logic-driven-ensemble — IMMUTABLE_TASKS lock + safe-blend 派生

## [MD]

> NeuroGolf 2026 — Safe Blend
> The simplest possible blend that actually works.
>
> (内容は magmacot/neurogolf-new-blending と同型。差分は IMMUTABLE_TASKS lock の追加と TOP_SOLUTION_PATH の指定。)

## [CODE]

```python
TOP_SOLUTION_PATH = "/kaggle/input/notebooks/konbu17/neurogolf-2026-blended-401-tasks-lb-5344/submission.zip"

# Эти таски берем ТОЛЬКО из топа и больше ни с чем не сравниваем
IMMUTABLE_TASKS = {
    #'task101.onnx',
    #'task133.onnx',
}

locked_tasks = set()

# --- ЭТАП 1: Захват ПРИОРИТЕТНЫХ моделей (только IMMUTABLE) ---
# 「TOP source の特定 task は他 source と比較せず固定する」 lock 機構
if os.path.exists(TOP_SOLUTION_PATH):
    label = "TOP_PRIORITY"
    with zipfile.ZipFile(TOP_SOLUTION_PATH) as zf:
        for entry in zf.namelist():
            if not entry.endswith('.onnx'): continue
            ...

# `safe_load_model` の本体は neurogolf-new-blending と同一
```

## 要点 (W2 抽出)

- **手法 (technique)**: ensemble_blending + immutable-task-pinning (= top source の特定 task を lock)
- **score (LB)**: kernel title が `[5700]` を主張。`konbu17/neurogolf-2026-blended-401-tasks-lb-5344` を TOP source としつつ他 source で残り task を補強した結果と推定
- **votes**: 5 (= 後発 magmacot/neurogolf-new-blending に votes が流れた)
- **核心アルゴリズム**:
  1. `TOP_SOLUTION_PATH` で指定した zip から **`IMMUTABLE_TASKS` set の task だけを最優先で lock** (= 他 source と比較すらしない)
  2. lock されなかった残り task は通常の per-task best-pick (= neurogolf-new-blending と同じ pipeline)
  3. cost = `params + len(serialized_bytes)`、identity fallback で穴埋め
- **特徴的な工夫**:
  - **`IMMUTABLE_TASKS` lock 機構**: 「他 source の方が cost 安いが採点で壊れる」という危険回避のための manual override 機能
  - lock の有効化はコメントアウトで控えめに (= 本 kernel ではコメントアウトされており、機構だけ提供)
  - magmacot 系 kernel 群 (= new-blending / 5600 / 5700 / 4200-v5 / 4213) は **同一 base から段階的に IMMUTABLE / TOP source / sub-strategy を入れ替えて** 検証している series
- **当該コンペでの応用余地**:
  - 「自分の解が確実に grader を通るが、他 source の方が cost 安い」場面で IMMUTABLE lock を使う
  - top source とそれ以外を分けて段階処理する pipeline 設計は終盤の安定提出に有効
- **限界 / 弱点**:
  - lock task を間違えると他 source の cheap 解を逃して LB を落とす
  - `IMMUTABLE_TASKS` の選定基準が public 化されておらず、再現には経験的検証が必要
  - votes 5 は本 kernel が series 内で「最終形」ではなく「中間段階」と扱われた示唆

## 出典

- Kernel URL: [https://www.kaggle.com/code/magmacot/5700-neurogolf-logic-driven-ensemble](https://www.kaggle.com/code/magmacot/5700-neurogolf-logic-driven-ensemble)
- 派生元: [https://www.kaggle.com/code/magmacot/neurogolf-new-blending](https://www.kaggle.com/code/magmacot/neurogolf-new-blending)
- このディレクトリの `kernel-metadata.json` 参照
