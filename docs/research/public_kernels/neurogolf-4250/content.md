# neurogolf-4250 — 決定論的 extract+rezip

## [MD]

> NeuroGolf — extract+rezip (deterministic)

## [CODE]

```python
import zipfile, os
from pathlib import Path

src_dir = None
for d in Path("/kaggle/input").rglob("task001.onnx"):
    src_dir = d.parent
    break
print(f"src: {src_dir}")
files = sorted(src_dir.glob("task*.onnx"))
print(f"files: {len(files)}")

dst = Path("/kaggle/working/submission.zip")
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        zf.write(f, f.name)
print(f"submission.zip: {dst.stat().st_size/1024:.1f}KB, {len(files)} files")
```

(全 cell これだけ。極めて短い。)

## 要点 (W2 抽出)

- **手法 (technique)**: ensemble_blending (= 既存 dataset の単純再 zip)
- **score (LB)**: kernel title が `4250` を主張 (= dataset 内の解だけで 4250 LB に到達した記録)
- **votes**: 53 (= 短さの割に多い = 後続 kernel が source として頻繁に attach するため)
- **核心アルゴリズム**:
  1. `/kaggle/input` 配下を recursive search して `task001.onnx` を含む dir を 1 つ見つける
  2. その dir 内の全 `task*.onnx` を sorted で submit.zip に詰める
  3. 何の判定もせず、何の改変もしない (= 完全 passthrough)
- **特徴的な工夫**:
  - **コードが極限まで短い** (= 14 line)。debug の必要がない
  - 入力 dataset が変わるたびに kernel version 更新 → 「source dataset の中身が 4250 LB 解そのもの」になっている (= notebook の役目は「version-locked submitter」)
  - 後続 kernel (= ngc26 v5/v8/v9/v11/v12) が **`needless090/neurogolf-4250` の特定 version を attach して取り込み** ているのは、本 notebook の version pin が source dataset version pin と完全一致するため
- **当該コンペでの応用余地**:
  - 我々の終盤で「特定 version の解を再現提出する shim」を作るときの参考。Kaggle の version pin を活用する
  - 「他人 dataset の中身が `task*.onnx` の集合だったら何も触らず submit する」という最小フォーマッタ
- **限界 / 弱点**:
  - 自前の解はゼロ (= dataset 提供者の解そのまま)
  - dataset の更新があると kernel 出力も変わるため、特定 LB の再現には **kernel version + dataset version の両方を pin** する必要

## 出典

- Kernel URL: [https://www.kaggle.com/code/needless090/neurogolf-4250](https://www.kaggle.com/code/needless090/neurogolf-4250)
- このディレクトリの `kernel-metadata.json` 参照
