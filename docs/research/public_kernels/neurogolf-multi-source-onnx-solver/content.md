# neurogolf-multi-source-onnx-solver — utils-derived 厳密 cost と broadcast 修復器

## [MD]

(notebook 自体に markdown はほぼ無し。冒頭コメントで `_load_utils()` と `exact_cost_via_utils()` の存在を強調。)

`vyankteshdwivedi` 作 (alias zorojuro)。**主催者公式の `neurogolf_utils.score_network()` を直接呼んで exact cost を取る** という、cost 計算の信頼性で他 kernel と差別化する集約 solver。`SOLUTION_DIRS` 8 source の中から per-task で最小コストの ONNX を選び、`EXCLUDED_TASKS = {21, 55, 80, 184, 202, 366}` で **採点 grader が壊す既知の bad task を除外** する gate を持つ。

## [CODE]

```python
# 公式 utils を import して exact cost を計算
def _load_utils():
    for d in [TASK_DIR, Path('/kaggle/input/competitions/neurogolf-2026')]:
        up = d / 'neurogolf_utils' / 'neurogolf_utils.py'
        if up.exists():
            spec = importlib.util.spec_from_file_location('ng', up)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None

def exact_cost_via_utils(model):
    utils = _load_utils()
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        f.write(model.SerializeToString())
    macs, mem, params = utils.score_network(fname)
    return int(macs + mem + params)

EXCLUDED_TASKS = {21, 55, 80, 184, 202, 366}  # grader が壊す既知 task

SOLUTION_DIRS = {
    'aliafzal_tiny':   Path('/kaggle/input/notebooks/aliafzal9323/neurogolf-2026-tiny-onnx-solver'),
    'artem_logic2':    Path('/kaggle/input/notebooks/artemnazemtsev/neurogolf-logic-driven-ensembling-part-2'),
    'artem_logic4':    Path('/kaggle/input/notebooks/artemnazemtsev/neurogolf-logic-driven-ensembling-part-4'),
    'jonathanchan':    Path('/kaggle/input/notebooks/jonathanchan/ngc26-constraint-smart-logic-mix-blending'),
    'needless090':     Path('/kaggle/input/notebooks/needless090/neurogolf-4250'),
    'konbu17_5344':    Path('/kaggle/input/notebooks/konbu17/neurogolf-2026-blended-401-tasks-lb-5344'),
    'beicicc_6233':    Path('/kaggle/input/notebooks/beicicc/neurogolf-6233-36-public-score-open-solution'),
    'afr1ste_5501':    Path('/kaggle/input/datasets/afr...'),  # 1 dataset 含む
}
```

## 要点 (W2 抽出)

- **手法 (technique)**: ensemble_blending + exact_cost_via_official_utils
- **score (LB)**: source に `beicicc_6233` (= 6233.36) と `konbu17_5344` を含むので、6000+ 帯を狙える集約 pipeline
- **votes**: 64
- **核心アルゴリズム**:
  1. 公式 `neurogolf_utils.score_network()` を **dynamic import** して `(macs, mem, params)` を直接取得
  2. 8 source からの ONNX をすべて load → exact cost で rank
  3. `EXCLUDED_TASKS = {21, 55, 80, 184, 202, 366}` を per-source で除外 (= grader が壊すと判明している)
  4. 最小コストかつ 1.44 MB / banned-op gate を pass する解を採用
- **特徴的な工夫**:
  - **公式 score_network() を直接呼ぶ** ため、自前 cost 計算の誤差ゼロ。採点 rule 変更にも追従可能 (= utils を再 import すれば良い)
  - `EXCLUDED_TASKS` (= 6 task) の **明示的 blacklist** が他 kernel より遥かに具体的。grader bug の経験知が蓄積されている
  - `_UTILS_FAILED` flag で utils load 失敗時の retry ガードを実装 (= 一度失敗したらフォールバック cost 計算に切り替え)
- **当該コンペでの応用余地**:
  - **公式 score_network() の dynamic import** は我々が真っ先に採用すべき pattern。自前 cost 関数より正確
  - `EXCLUDED_TASKS` set は我々の早期 baseline でも同じく除外すべき (= 同じ grader bug を踏む可能性高い)
- **限界 / 弱点**:
  - 公式 utils が `/kaggle/input/competitions/neurogolf-2026/neurogolf_utils/` に居ない環境 (= ローカル open 展開) では fallback 必須
  - 8 source の選定根拠は明示されず、source 増やすと管理コストが急増する
  - `EXCLUDED_TASKS` の 6 task が「全 grader version で壊れる」のか「ある version 限定で壊れる」のか不明 (= 採点 rule 変更で復活する可能性あり)

## 出典

- Kernel URL: [https://www.kaggle.com/code/vyankteshdwivedi/neurogolf-multi-source-onnx-solver](https://www.kaggle.com/code/vyankteshdwivedi/neurogolf-multi-source-onnx-solver)
- このディレクトリの `kernel-metadata.json` 参照
