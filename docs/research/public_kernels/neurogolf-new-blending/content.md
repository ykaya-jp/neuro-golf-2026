# neurogolf-new-blending — "raw bytes 不変" 原則の最小 blend

## [MD]

著者 magmacot は本 kernel を **「実際に動く最小限の blend」** と位置付け、過去の blend が破壊した model を観察した教訓を inline で文書化している。原文の戦略と過去失敗の解説を要約引用する。

> NeuroGolf 2026 — Safe Blend: 実際に動く最小限の blend。
>
> 戦略の 6 ステップ:
> 1. attached notebook の submission.zip をすべて発見
> 2. 各 ONNX model を **raw bytes のまま zero modification で load** を試みる
> 3. 動いて input/output 名を持っていれば **元 bytes を一切変更しない**
> 4. rename が必要なら正しく直す (= 先に old name を capture、後で全 reference を rename)
> 5. per-task で cheapest model を残す
> 6. missing task は identity fallback で穴埋め
>
> 過去 blend が壊れた理由 (= 著者の根本診断):
> - `onnx.load_from_string()` + `SerializeToString()` は valid model を **silent に corrupt** することがある
> - 変更不要なケースでも re-serialization が custom op や metadata を破壊しうる
> - 旧 rename bug (= rename 後に old name を読む順序ミス) が intermediate node reference を破壊した
>
> 本 notebook の運用方針: 「動いている model は決して触らない」。Period (= 例外なし)。

## [CODE]

```python
SOURCE_ZIPS = []
for zp in sorted(glob.glob('/kaggle/input/notebooks/**/submission.zip', recursive=True)):
    SOURCE_ZIPS.append((zp, Path(zp).parent.name))

def safe_load_model(raw_bytes, task_id=None):
    """input/output 名を 'input'/'output' に揃えるだけの最小 fix。
    ｲﾝﾌｪﾚﾝｽﾃｽﾄは行わない。ｿｰｽを信頼する。
    """
    try:
        model = onnx.load_model_from_string(raw_bytes)
        g = model.graph
        rename_map = {}
        old_in  = g.input[0].name
        old_out = g.output[0].name
        if old_in  != 'input':  rename_map[old_in]  = 'input'
        if old_out != 'output': rename_map[old_out] = 'output'

        if rename_map:
            g.input[0].name  = rename_map.get(old_in, old_in)
            g.output[0].name = rename_map.get(old_out, old_out)
            for node in g.node:
                node.input[:]  = [rename_map.get(n, n) for n in node.input]
                node.output[:] = [rename_map.get(n, n) for n in node.output]
            for vi in g.value_info:
                vi.name = rename_map.get(vi.name, vi.name)
            for init in g.initializer:
                init.name = rename_map.get(init.name, init.name)

        model.ir_version = 8
        fixed_bytes = model.SerializeToString()
        params = sum(int(np.prod(i.dims)) for i in g.initializer if i.dims)
        cost = params + len(fixed_bytes)
        return True, fixed_bytes, cost
    except Exception:
        return False, raw_bytes, float('inf')
```

## 要点 (W2 抽出)

- **手法 (technique)**: ensemble_blending + minimal-mutation_safety
- **score (LB)**: 単独 LB 値主張なし。他人の submission を最小改変で混ぜる安全 blend
- **votes**: 27
- **核心アルゴリズム**:
  1. `/kaggle/input/notebooks/**/submission.zip` を recursive glob ですべて発見
  2. 各 ONNX を `onnx.load_model_from_string` で parse、**input/output name を `'input'`/`'output'` に正規化** (= grader が要求する規約)
  3. **rename が不要な場合は raw bytes を一切触らない** (= re-serialization 経路でモデル破壊する罠を回避)
  4. cost = `params + len(serialized_bytes)` で per-task best-pick
- **特徴的な工夫**:
  - **「working model は触るな」原則** を最優先。`onnx.load → SerializeToString` の re-serialization が custom op / metadata を silent に壊す問題を回避
  - rename 順序を **「先に old name を capture してから rename」** に修正。previous blend の bug (= rename した後で old name を読もうとする) を明示的に直す
  - **subgraph も再帰的に rename** する `_fix_subgraph` を別途定義 (= If/Loop の中の node も対応)
  - `model.ir_version = 8` を強制 (= grader が opset 13 / IR 8 を期待する)
- **当該コンペでの応用余地**:
  - **「他人の解を最小改変で混ぜる」原則** は我々の終盤統合段階の防衛策として極めて重要
  - `_fix_subgraph` の再帰 rename pattern は subgraph を持つ複雑な ONNX で必須
  - cost = `params + bytes` という単純式は MACs 削除後の現行採点に整合
- **限界 / 弱点**:
  - 自分で task を解いていない (= 集約のみ)。input source 集合を超えるスコアは出ない
  - `verify_network` を通さないので、rename 後に inference が壊れているケースを検出できない
  - identity fallback は精度ゼロ (= 入力をそのまま出力すると当然 0 点)。missing task の rescue 戦略は別途必要

## 出典

- Kernel URL: [https://www.kaggle.com/code/magmacot/neurogolf-new-blending](https://www.kaggle.com/code/magmacot/neurogolf-new-blending)
- このディレクトリの `kernel-metadata.json` 参照
