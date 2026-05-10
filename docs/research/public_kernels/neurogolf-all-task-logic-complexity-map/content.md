# neurogolf-all-task-logic-complexity-map — task primitive 分類と難易度スコア

## [MD]

> ARC Logic Decoder — This tool converts complex ARC-AGI grids into a structured list of building blocks. Instead of reading long descriptions, it helps you identify the exact logic needed to build the smallest, most efficient AI models for the competition.
>
> Why this is helpful:
> - Pick the Best Tools: Identify which transformations (like "Mirror" or "Shift") are used.
> - Win Faster: Use the Complexity (1-10) score for a quick evaluation.
> - Grid Changes: Instantly track if a task changes dimensions.

## [CODE]

```python
SYSTEM_PROMPT = """You are a world-class ARC-AGI pattern recognizer.
You MUST output a valid JSON object where the key is the Task ID, and the value follows this exact schema:

{
  "TASK_ID_HERE": {
    "Spatial_and_Geometric": [], // Use ONLY: Rotation, Reflection, Translation, Shifting, Tiling, Cropping, Magnification
    "Object_Based": [],          // Use ONLY: Object Detection, Gravity, Collision Detection, Shape Matching, Sorting, Outlier Detection
    "Color_and_Logical": [],     // Use ONLY: Color Swapping, Background Separation, Flood Fill, Bitwise Logic, Color Mapping
    "Pattern_Recognition": [],   // Use ONLY: Symmetry Completion, Filling Regions, Line Extrapolation, Intersection Finding, Pathfinding, In-painting
    "Primary_Category": "string",
    "Grid_Size_Changed": boolean,
    "Estimated_Complexity": integer  // 1-10
  }
}

### COMPLEXITY GUIDELINES (Neurogolf Edition):
- 1-2: ELIMINATION/TRIVIA
"""
```

DeepSeek に各 task の primitive 集合と difficulty 1-10 を **構造化 JSON** で出力させる。CSV header は `Task_ID, All_Used_Transformations, Total_Transformations, Spatial_Count, Object_Count, Color_Count, Pattern_Count, Primary_Category, Grid_Size_Changed, Estimated_Complexity`。

## 要点 (W2 抽出)

- **手法 (technique)**: logic_decoder + difficulty_estimation (= LLM による taxonomy 構築)
- **score (LB)**: submission 自体は作らない (= meta-data 生成 kernel)。出力は [Neurogolf 2026 Task Transformation Library](https://www.kaggle.com/datasets/karnakbaevarthur/neurogolf-2026-task-transformation-library) として公開
- **votes**: 71
- **核心アルゴリズム**:
  1. 4 軸 × 6-7 primitive (= 全 24 種) の **closed vocabulary** を system prompt で強制
  2. task ごとに `{Spatial: [...], Object: [...], Color: [...], Pattern: [...], Primary_Category, Grid_Size_Changed, Complexity}` を生成
  3. 結果を CSV / JSON 両方で永続化、可視化用の per-task 入出力比較画像 cell も用意
- **特徴的な工夫**:
  - **closed vocabulary を prompt 内で明示** することで LLM が新規語を勝手に作らないよう制約 (= 後段の機械処理しやすい)
  - `Primary_Category` (= 主要カテゴリ 1 個) と `Estimated_Complexity` (= 1-10 整数) を別フィールドにして bucket-vs-difficulty matrix を組みやすくしている
  - logic-decoder と異なり **Grid_Size_Changed** という ONNX-side で必要な reshape 必要性を直接出力に含めている
- **当該コンペでの応用余地**:
  - **task の bucket 化** (= 24 primitive × 400 task の matrix) → 構造原理が同じ task は同じ ONNX template を使い回せるという「per-task tiny model」戦略の前段に必須
  - difficulty 1-2 task は handcrafted Conv で速攻攻略、5-10 task は別アプローチ、と attention budget を分配する判断材料
- **限界 / 弱点**:
  - LLM の primitive 分類精度は **未検証**。誤分類した task を ONNX template で攻めると失敗する
  - 24 primitive の closed vocabulary は ARC タスクの全表現空間をカバーしない (= 実際の ARC は continuous な変換を含む)
  - `Estimated_Complexity` 1-10 は LLM の subjective scoring。同じ task でも seed/temperature で揺れる

## 出典

- Kernel URL: [https://www.kaggle.com/code/karnakbaevarthur/neurogolf-all-task-logic-complexity-map](https://www.kaggle.com/code/karnakbaevarthur/neurogolf-all-task-logic-complexity-map)
- 出力 dataset: [https://www.kaggle.com/datasets/karnakbaevarthur/neurogolf-2026-task-transformation-library](https://www.kaggle.com/datasets/karnakbaevarthur/neurogolf-2026-task-transformation-library)
- このディレクトリの `kernel-metadata.json` 参照
