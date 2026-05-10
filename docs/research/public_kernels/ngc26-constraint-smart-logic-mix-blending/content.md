# ngc26-constraint-smart-logic-mix-blending — 大規模 mix-blend 集約パイプライン

## [MD]

> Revised from rauffauzanrambe/neurogolf-constraint-smart-logic-ensemble-4k (v1), which is modified from artemnazemtsev/4180-neurogolf-logic-driven-ensembling (v21), plus rough translations from original source.

v1 から v17+ までのバージョン履歴を逐一公開しており、各バージョンで「どの notebook / dataset を取り込んだ結果 PLB がいくつ動いたか」を完全トレース。最新の公開 PLB は **v16 で 5546.64 スコア源を取り込み** (LB 付近)、blending 1 回の進歩あたり数十～数百ポイント上昇するスケーリング pattern を示している。

## [CODE]

```python
# ════════════════════════════════════════════════════════════
# CELL 1 — INSTALL & IMPORTS / static profile
# ════════════════════════════════════════════════════════════
TASK_RE   = re.compile(r'^task\d{3}\.onnx$')
C, H, W   = 10, 30, 30
HW        = H * W
NUM_TASKS = 400
MAX_BYTES = int(1.44 * 1024 * 1024)
BANNED_OPS = {'Loop', 'Scan', 'NonZero', 'Unique', 'If', 'Function'}

# ════════════════════════════════════════════════════════════
# STEP 1 — CONVERT FOLDERS → ZIPS
# folders with .onnx files を zip に正規化して以後の per-task best-pick で扱える形に揃える
# ════════════════════════════════════════════════════════════
def zip_onnx_files(input_dir, zip_name=None):
    input_dir = Path(input_dir)
    onnx_files = [
        p for p in input_dir.glob('*.onnx')
        if p.is_file() and TASK_RE.match(p.name)
    ]
    output_zip = Path('/kaggle/working') / (input_dir.name + ".zip")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in onnx_files:
            z.write(f, arcname=f.name)
    return output_zip
```

複数 source の zip / dir を吸い込み、各 task ごとに `verify_network`-pass する **最安 cost ONNX** を選ぶ pipeline。`MAX_BYTES = 1.44 MB / file`、`BANNED_OPS = {Loop, Scan, NonZero, Unique, If, Function}` を厳守。

## 要点 (W2 抽出)

- **手法 (technique)**: ensemble_blending (= mix-blend across 多 source ONNX)
- **score (LB)**: v16 で 5546.64 系 source を取り込み (= public LB 近傍 5500-5650 帯)。v15 で 5518.02、v14 で 5364.78
- **votes**: 115 (= 全 kernel 中 2 位)
- **核心アルゴリズム**:
  1. 引数として渡された `FOLDERS_WITH_ONNX` (フォルダ群) と `EXTERNAL_ZIPS` (zip 群) を全部 `/kaggle/working/` に zip 化
  2. 各 zip を loop で開き、`task001.onnx` ～ `task400.onnx` を逐次 `score_network()` で cost 算出
  3. task ごとに **cost が最小かつ banned-op を含まず 1.44 MB 以下** の ONNX を採用
  4. 残った missing task は identity fallback (= 入力をそのまま出力)
- **特徴的な工夫**:
  - **per-file 1.44 MB 制約への準拠** を v10 で明示修正 (= submission.zip 全体の制約と勘違いしないよう注意喚起)
  - **version log がそのまま「どの notebook を取り込めば +N 点上がるか」のレシピ集** になっている (= 後続の他 kernel が直接参照できる credit graph)
  - DeflateLevel ZIP_DEFLATED を明示。zip の中身名を basename に正規化することで `task101.onnx`-相対 path を保証
- **当該コンペでの応用余地**:
  - 我々の戦略が「per-task 解の集合」を持つ場合、本 pipeline をそのまま「最終 submit 機」として再利用できる
  - 異なる approach (= rule-based / tiny conv / LLM 生成) を mix する際の base infra として最適
- **限界 / 弱点**:
  - 自分で task を解いていない (= 他人の zip を集めて best-pick しているだけ)。submission の質は input source の質で律速
  - v17 までで 130+ KB の cell コードが膨れ、mainnance しづらい (cell 分割は v17 で導入)
  - `BANNED_OPS` set に `Script` / `COMPRESS` が含まれず opset 13 / IR 8 normalization の網羅性は他 kernel (konbu17) より低い

## 出典

- Kernel URL: [https://www.kaggle.com/code/jonathanchan/ngc26-constraint-smart-logic-mix-blending](https://www.kaggle.com/code/jonathanchan/ngc26-constraint-smart-logic-mix-blending)
- 派生元 v1: [https://www.kaggle.com/code/rauffauzanrambe/neurogolf-constraint-smart-logic-ensemble-4k](https://www.kaggle.com/code/rauffauzanrambe/neurogolf-constraint-smart-logic-ensemble-4k)
- このディレクトリの `kernel-metadata.json` 参照
