# arc-nano-engine — 巨大 source 集合の per-task best-pick aggregator

## [MD]

(notebook 自体に markdown はほぼ無く、コードコメント "# ENSEMBLE v139b" がバージョン手がかり。)

`svanikkolli` 作。**4 つの notebook submission + 31 の dataset** という巨大集合からモデルを per-task で集めて選定する aggregator。`svanikkolli` 自身が同名の dataset 群 (= secret-dataset / 5550-dataset / lb-4995 / ngc26-dataset 等) を多数公開しており、それを取り込むハブ kernel になっている。

## [CODE]

```python
NOTEBOOK_SOURCES = [
    ('NGC_Mix', Path('/kaggle/input/notebooks/jonathanchan/ngc26-constraint-smart-logic-mix-blending/submission.zip')),
    ('Konbu_341', Path('/kaggle/input/notebooks/konbu17/neurogolf-2026-blended-341-tasks-lb-4215/submission.zip')),
    ('Magma_4200', Path('/kaggle/input/notebooks/magmacot/4200-v5-neurogolf-fix-for-new-system-soon/submission.zip')),
    ('Afr1ste_6225', Path('/kaggle/input/notebooks/afr1ste/neurogolf-6225-51-public-score-open-solution/submission.zip')),
]

ENSEMBLE_SOURCES = [
    ('Secret', Path('/kaggle/input/datasets/svanikkolli/secret-dataset')),
    ('5550', Path('/kaggle/input/datasets/svanikkolli/5550-dataset')),
    ('LB4995', Path('/kaggle/input/datasets/svanikkolli/lb-4995')),
    ('NGC26', Path('/kaggle/input/datasets/svanikkolli/ngc26-dataset')),
    ('345ONNX', Path('/kaggle/input/datasets/svanikkolli/345-onnx-submission-dataset')),
    # ... 計 31 dataset
    ('Konbu_401', Path('/kaggle/input/datasets/konbu17/neurogolf-2026-blended-401-v117')),
]

def load_zip(zip_path, label):
    models = {}
    if not zip_path.exists(): return models
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for entry in zf.namelist():
            m = re.match(r'task(\d{3})\.onnx', os.path.basename(entry))
            if m: models[...] = ...
```

## 要点 (W2 抽出)

- **手法 (technique)**: ensemble_blending (= mixed source aggregator at scale)
- **score (LB)**: アクセスする source 名から推定すると 5500-6225 帯の source を取り込んでおり、その範囲の LB を狙える
- **votes**: 63
- **核心アルゴリズム**:
  1. 4 notebook + 31 dataset = **35 source** を逐次 load
  2. ハイ priority (= notebook source) を優先、低 priority (= dataset) で補完
  3. cost-aware で per-task 最小コスト ONNX を採用
  4. submission.zip / submission/ dir 両方の format に対応
- **特徴的な工夫**:
  - **dataset と notebook source を分離** (= notebook の方が新しい score を持つので priority 高)
  - svanikkolli が「他人の submission を再 host した dataset」を大量に公開し、本 kernel をハブとして全 source を 1 つの search space に統合する **エコシステム形成戦略**
  - kernel 名 `arc-nano-engine` が示す通り、複数の "nano" 解 (= per-task tiny model) を engine 1 つで集約するメタファ
- **当該コンペでの応用余地**:
  - 我々の戦略の終盤で source 集合が 10+ になった際の **集約 backbone** として直接ポート可能
  - svanikkolli が公開した dataset 群 (= `svanikkolli/secret-dataset` etc.) を **早期に attach して内容を確認** する価値あり (= 各 dataset がどの source から来たか不明だが、cheap な解の集合になっている可能性)
- **限界 / 弱点**:
  - 35 source の auto-load は I/O 重く、Notebook 立ち上げに数分かかる
  - source 数が多すぎると **どの source の貢献で +X 点** が見えづらくなる (= ablation 困難)
  - cost 計算は単純な `params + nbytes` で済ませており shape inference / banned-op gate は他 kernel より浅い

## 出典

- Kernel URL: [https://www.kaggle.com/code/svanikkolli/arc-nano-engine](https://www.kaggle.com/code/svanikkolli/arc-nano-engine)
- このディレクトリの `kernel-metadata.json` 参照
