# neurogolf-2026-tiny-onnx-solver — 検出器 + 学習可能 conv fallback の hybrid solver

## [MD]

著者 aliafzal9323 は本 kernel を **「公開 notebook の ensemble + 自作 solver + dataset 取り込み」のハイブリッド** と位置付け、per-task で最小サイズの有効 ONNX を残す戦略を明記する。原文の章構成と要約は次の通り。

> NeuroGolf 2026: Tiny-ONNX Solver
> 戦略は「複数の public notebook の ensemble + 自作 solver + dataset source」を全部統合し、最大スコアのために per-task で最小有効 model を残す。
>
> 章構成:
> - Detectors: identity / color_map / rotation / flip / transpose / tile を pure numpy で判定
> - Gather-index builders: ONNX Gather op の index 構築ユーティリティ
> - Learned-conv fallbacks: detector に該当しない task は multi-seed で MSE/BCE 学習し、結果を ternary {-1, 0, +1} に snap
> - Main solver: 上記を per-task で適用する駆動部
> - Blend Pipeline: ZIP source / dataset / automated solver の出力を per-task best-pick で merge

## [CODE]

```python
C, H, W = 10, 30, 30
HW = H * W
TASK_RE = re.compile(r'^task\d{3}\.onnx$')
MAX_BYTES = int(1.44 * 1024 * 1024)
EXCLUDED_OPS_UPPER = {'LOOP', 'SCAN', 'NONZERO', 'UNIQUE', 'SCRIPT', 'FUNCTION'}

# Auto-discover ALL submission.zip files from kernel inputs
SOURCE_ZIPS = {}
for zp in Path('/kaggle/input').rglob('submission.zip'):
    label = zp.parent.name
    SOURCE_ZIPS[label] = zp

# Auto-discover ALL loose ONNX dirs
DATASET_DIRS = []
for d in Path('/kaggle/input').iterdir():
    if d.is_dir() and list(d.glob('task*.onnx'))[:1]:
        DATASET_DIRS.append(d)

# --- detectors (samesize 系幾何変換)
def detect_identity(pairs): return all(p['input'] == p['output'] for p in pairs)

def detect_color_map(pairs):
    if not same_size(pairs): return None
    cmap = {}
    for p in pairs:
        for a, b in zip(np.array(p['input']).flatten(), np.array(p['output']).flatten()):
            if int(a) in cmap and cmap[int(a)] != int(b): return None
            cmap[int(a)] = int(b)
    return cmap

def detect_rotation(pairs):
    for k in (1,2,3):
        ok = all(np.array_equal(np.rot90(np.array(p['input']), k), np.array(p['output'])) for p in pairs)
        if ok: return 90*k
    return None

def detect_flip(pairs): ...
def detect_transpose(pairs): ...
def detect_tile(pairs): ...
```

## 要点 (W2 抽出)

- **手法 (technique)**: rule_based_program_synthesis + tiny_onnx_compression + ensemble_blending
- **score (LB)**: 単独 LB は明示なし (= 公開 ensemble 候補として扱われる)。multi-source blend 時に他 kernel から **64 votes** の信頼を得ている
- **votes**: 64
- **核心アルゴリズム**:
  1. **detectors** で各 task が「identity / color_map / rotation / flip / transpose / tile」のどれかに該当するかを純 numpy で判定
  2. 該当した場合は **対応する最小 ONNX template** を組み立てる (= ハードコード)
  3. どの detector も該当しない場合は **learned-conv fallback** (= multi-seed で MSE/BCE 損失で fit、結果を ternary {-1, 0, +1} に snap)
  4. 既存 source zip / dataset と自前 solver の出力を per-task で best-pick (= cost 最小)
- **特徴的な工夫**:
  - **`/kaggle/input/**/submission.zip` を auto-discover** (= attached notebook 一覧を hard-code 不要)
  - **multi-seed + ternary snap** (= 学習後の重みを `{-1, 0, +1}` に丸める) で ONNX サイズを大幅削減。Conv 重みの 32-bit FP を 2-bit 表現相当に圧縮しつつ精度維持
  - detectors は **closed-form / pure numpy** で書かれているので動作が決定論的、デバッグ容易
- **当該コンペでの応用余地**:
  - **Tier 1**: detectors 部分は我々の baseline として直接ポート可能 (= identity / color_map / rotation / flip だけで 50+ task 解ける可能性)
  - **Tier 2**: learned-conv fallback の ternary snap technique は ONNX サイズ削減の main lever として再利用
  - **Tier 3**: auto-discover source の仕組みは public dataset を機械的に取り込む infra として優秀
- **限界 / 弱点**:
  - detectors は 6-7 個しか無く、複雑な reshape / object motion / pathfinding 系には届かない
  - learned-conv は per-task で学習するので 400 task 全部に走らせると Kaggle Notebook の 9-12h GPU/CPU 時間制約に当たる
  - ternary snap は **正解が三値で表現可能な task** にしか効かない。連続値が必要なタスクでは精度劣化する

## 出典

- Kernel URL: [https://www.kaggle.com/code/aliafzal9323/neurogolf-2026-tiny-onnx-solver](https://www.kaggle.com/code/aliafzal9323/neurogolf-2026-tiny-onnx-solver)
- このディレクトリの `kernel-metadata.json` 参照
