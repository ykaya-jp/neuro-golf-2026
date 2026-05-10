# cross-source-ensemble — 厳格 static check + onnx-tool による詳細プロファイル

## [MD]

> Methodology
> - Dual-Source Ensembling: Aggregates ONNX models from multiple high-performing datasets.
> - Robust Validation: Uses onnxruntime to verify solutions against Train, Test, and ARC-Gen-100K distributions.
> - Efficiency Scoring: selects the "cheapest" valid model based on a custom cost metric (Params + Bytes + MACs).
> - Compliance Filtering: Ensures strict adherence to size limits (<1.44MB) and banned operator sets.

`karnakbaevarthur` 作。3 source notebook (`artemnazemtsev1` / `artemnazemtsev2` / `jonathanchan`) を **priority 順** に取り込み、`onnx_tool==1.0.0` で詳細プロファイル → static_check の 3 段階 gate (= size / banned-op / dim_value) を通過した解だけを採用する。

## [CODE]

```python
TASK_PATTERN  = re.compile(r'^task\d{3}\.onnx$')
MAX_BYTES     = int(1.44 * 1024 * 1024)
BANNED_OPS    = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function"}

# onnx-tool バージョン強制
def ensure_onnx_tool_v1():
    try:
        if importlib.metadata.version('onnx-tool') != '1.0.0':
            need_install = True
    except importlib.metadata.PackageNotFoundError:
        need_install = True
    if need_install:
        os.system(f'{sys.executable} -m pip install onnx-tool==1.0.0')
        if 'onnx_tool' in sys.modules:
            import onnx_tool
            importlib.reload(onnx_tool)

ensure_onnx_tool_v1()
import onnx_tool

SOLUTION_DIRS = {
    "artemnazemtsev1": Path('/kaggle/input/notebooks/artemnazemtsev/4275-submission'),
    "artemnazemtsev2": Path('/kaggle/input/notebooks/artemnazemtsev/neuro-golf-gambling-is-all-you-need'),
    "jonathanchan":    Path('/kaggle/input/notebooks/jonathanchan/ngc36-constraint-smart-logic-mix-blending'),
}
PRIORITY = {"artemnazemtsev1": 1, "artemnazemtsev2": 2, "jonathanchan": 3}

def static_check(raw_bytes: bytes) -> tuple[bool, float, str]:
    if len(raw_bytes) > MAX_BYTES:
        return False, float('inf'), "Error: Size > 1.44MB"
    # banned-op + shape inference + cost via onnx_tool
    ...
```

## 要点 (W2 抽出)

- **手法 (technique)**: ensemble_blending + strict_static_validation
- **score (LB)**: artem 4275 系 + jonathanchan ngc36 系を取り込むので 4275-5500 帯
- **votes**: 64
- **核心アルゴリズム**:
  1. `onnx-tool==1.0.0` を **強制 install + reload** (= バージョン依存 bug 回避)
  2. 3 source の zip / dir を `load_graphs_from_dir` で **再帰的に zip 展開**しながら graph 抽出
  3. `static_check` で 1.44 MB / banned-op / shape inference の 3 段 gate
  4. priority 順で最初に通過した解を採用 (= cost 同点なら priority で tie-break)
- **特徴的な工夫**:
  - **`importlib.metadata.version('onnx-tool')` で version assert + 自動再 install** という防衛コード。Kaggle Notebook の cache 状態に依存しない再現性確保
  - `BANNED_OPS` に **`Script` / `Function` を含む** (= ngc26 より厳格)。これが現在の grader 仕様に近い
  - 3 source のみの **小規模 ensemble**。多すぎる source を扱う arc-nano-engine と対照的に保守可能性を重視
- **当該コンペでの応用余地**:
  - 我々の独自解が増えた段階で「自分の解 + 信頼できる 1-2 source」の小規模 priority blending として採用可能
  - `static_check` の 3 段 gate は submit 前の最終 sanity check として直接ポート
- **限界 / 弱点**:
  - 3 source のみなので絶対 LB 帯は arc-nano-engine / ngc26 より低い
  - `onnx-tool==1.0.0` 固定は今後のバージョン破壊変更で動かなくなるリスク (= 上位互換テストすべき)
  - priority による tie-break は cost が同じなら先勝ち。質的に「priority 1 が常に良い」前提で source 順を整理する必要

## 出典

- Kernel URL: [https://www.kaggle.com/code/karnakbaevarthur/cross-source-ensemble](https://www.kaggle.com/code/karnakbaevarthur/cross-source-ensemble)
- このディレクトリの `kernel-metadata.json` 参照
