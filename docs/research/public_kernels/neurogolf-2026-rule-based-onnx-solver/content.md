# neurogolf-2026-rule-based-onnx-solver — 2 source の per-task best-pick + 厳密コスト計算

## [MD]

(notebook 自体は markdown 装飾の hero header のみで、技術的説明はコード冒頭の docstring に集中。)

`imaadmahmood` 作。**2 つの公開 dataset (`sub4979_07` / `sub4971_98`) の zip を per-task で merge** し、コスト最小の ONNX を採用する集約 builder。`compute_cost` が **静的 shape inference + Conv/Gemm/MatMul の MAC 計算** まで自前で実装している点が他 kernel との差別化。

## [CODE]

```python
SOURCE_ZIPS = {
    "sub4979_07": Path("/kaggle/input/datasets/jonathanchan/sub4979-07"),
    "sub4971_98": Path("/kaggle/input/datasets/jonathanchan/sub4971-98"),
}
MAX_BYTES = int(1.44 * 1024 * 1024)
BANNED    = {"Loop","Scan","NonZero","Unique"}

def compute_cost(raw: bytes):
    if len(raw) > MAX_BYTES: return None
    m = onnx.load_model_from_string(raw)
    m = onnx.shape_inference.infer_shapes(m)
    tb={1:4,6:4,7:8,10:2,11:8,16:2,2:1,3:1,4:2,5:2}  # type → byte width
    params=nbytes=macs=0; shapes={}; ops=set()
    for init in m.graph.initializer:
        dims=list(init.dims); p=math.prod(dims) if dims else 1
        params+=p; nbytes+=p*tb.get(init.data_type,4); shapes[init.name]=dims
    for inp in m.graph.input:
        s=[]
        for d in inp.type.tensor_type.shape.dim:
            if d.HasField('dim_value'): s.append(d.dim_value)
            else: return None  # 動的 shape は弾く
        shapes[inp.name]=s
    # value_info / output も同様に処理 ...
    for node in m.graph.node:
        ops.add(node.op_type)
        if node.op_type=="Conv":
            _,co,ho,wo=os_; _,ci,kh,kw=ws
            macs+=co*ci*kh*kw*ho*wo
        elif node.op_type in ("Gemm","MatMul"):
            macs+=math.prod(sa[:-1])*sa[-1]*sb[-1]
    if BANNED & ops: return None
    return params + nbytes + macs

def score_from_cost(cost):
    if cost is None: return 0.0
    return max(1.0, 25.0 - math.log(max(1, cost))) if cost > 0 else 25.0
```

## 要点 (W2 抽出)

- **手法 (technique)**: ensemble_blending + onnx_template_search (= 厳密 cost 計算でランク付け)
- **score (LB)**: 2 source の優れた方を per-task で選ぶ pipeline。元 source `sub4979` 系の値 (= 4979.07) を超える程度の LB に到達しうる
- **votes**: 75
- **核心アルゴリズム**:
  1. `SOURCE_ZIPS` で指定された dataset/notebook 出力の `.zip` (or 直 `.onnx`) をすべて load
  2. 各 ONNX の `compute_cost` で `params + memory_bytes + MACs` を計算 (= **採点 rule 変更前の旧公式に対応**)
  3. 各 task ごとに最安 cost の ONNX を採用、`max(1, 25 - log(cost))` で score シミュレート
  4. 1.44 MB / banned-op 違反は弾く
- **特徴的な工夫**:
  - **`onnx.shape_inference.infer_shapes()` を必ず呼んで static shape を確定**してから cost 計算する。動的 shape (= dim_value 不在) を弾く gate がある
  - 各 ONNX dtype ごとに **byte width 表 `tb`** (FP32=4 / FP64=8 / FP16=2 / INT8=1 / BOOL=2) を hard-code し正確に memory bytes を集計
  - score シミュレーションを inline で持ち、submit せずに「この組み合わせなら何点」を local で見える化
- **当該コンペでの応用余地**:
  - 我々が独自解を作った場合のローカル評価 (= cost 計算 + score シミュ) の reference 実装として最も信頼できる
  - 異なる source dataset を merge する際の `compute_cost` を再利用すれば独自評価器を実装する手間を省ける
- **限界 / 弱点**:
  - 自前で task を解いていない (= 他人の zip を集めるのみ) → ngc26 / cross-source-ensemble と同型の弱点
  - **MACs を含むコスト式は 2026-05-06 で旧式化**。`compute_cost` を `params + nbytes` のみに修正する必要あり
  - `BANNED` set に `If` / `Function` / `Script` / `COMPRESS` が無い (= konbu17 may-8-updated より緩い)

## 出典

- Kernel URL: [https://www.kaggle.com/code/imaadmahmood/neurogolf-2026-rule-based-onnx-solver](https://www.kaggle.com/code/imaadmahmood/neurogolf-2026-rule-based-onnx-solver)
- このディレクトリの `kernel-metadata.json` 参照
