# logic-decoder — DeepSeek を用いた task 自然言語 logic 抽出

## [MD]

著者は本 kernel を **ARC Logic Engine** と呼び、ARC-AGI パターンを人間可読な rule に分解するためのツールと位置付けている。原文要約は次の通り。

> ARC Logic Engine: ARC-AGI patterns を人間可読な rule に decode し、似た task を group 化して ONNX アーキ設計を高速化する。
>
> 想定用途は以下の 3 点と説明されている。
> - Group Tasks: "Symmetry" や "Rotation" の task を見つけてアーキを共有する
> - Identify Difficulty: LLM が失敗する地点を可視化して手作業の優先度を決める
> - Architecture Mapping: rule から Conv2D / Pooling / Tiling の選択を導く
>
> 次の発展方向として、rule の主要キーワード (= "Rotate" / "Mirror" / "Scale") から低コスト ONNX operator への automatic mapping、および NLP embedding によるタスククラスタリングが言及されている。

## [CODE]

```python
import json, os, glob, time, csv
from openai import OpenAI
from kaggle_secrets import UserSecretsClient

START_TASK, END_TASK = 1, 10
BATCH_SIZE = 1
JSON_OUTPUT = '/kaggle/working/arc_explanations.json'
CSV_OUTPUT = '/kaggle/working/arc_explanations.csv'

user_secrets = UserSecretsClient()
client = OpenAI(
    api_key=user_secrets.get_secret("deepseek_api_key"),
    base_url="https://api.deepseek.com"
)

# --- The "Logic First" Prompt ---
prompt = """Analyze the transformation logic between the Input and Output grids.
Focus on:
1. Objects (groups of same-colored pixels) and their movement or scaling.
2. Color changes (which color replaces which).
3. Geometry (rotation, reflection, or symmetry).

Provide a specific 2-3 sentence rule that works for ALL examples. Make rule specific.
Return a JSON object where keys are Task IDs and values are the specific rule string."""

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a world-class ARC-AGI pattern recognizer. ..."},
        {"role": "user", "content": prompt}
    ],
    response_format={'type': 'json_object'}
)
```

`/kaggle/input/competitions/neurogolf-2026/task*.json` を順番に読み、各 task の train pair 全部を prompt に詰めて DeepSeek (deepseek-chat) に「2-3 sentence rule」を要求 → JSON / CSV で永続化。

## 要点 (W2 抽出)

- **手法 (technique)**: logic_decoder (= LLM-driven program synthesis の前段、= 自然言語化)
- **score (LB)**: 単独では submission を作らない (= 補助 dataset 生成 kernel)。出力は [Logic for each ARC task](https://www.kaggle.com/datasets/karnakbaevarthur/logic-for-each-arc-task) として公開
- **votes**: 73
- **核心アルゴリズム**:
  1. 各 task の train pair を formatted text grid に変換 (`R0: [0,0,1,0]` 形式)
  2. DeepSeek API に「2-3 sentence specific rule」と JSON 構造を要求
  3. 結果を逐次 save (3 retry リトライ付)
- **特徴的な工夫**:
  - **system prompt で "world-class ARC-AGI pattern recognizer" の persona** を強制。response_format を `json_object` に固定
  - prompt は具体的に **"Objects / Color / Geometry"** の 3 軸を明示し、答えが JSON-parseable になるよう誘導
  - DeepSeek を選ぶ理由は明示されないが、コスト ($0.14 / 1M token) と JSON mode 対応が GPT-4 系より低リスク
- **当該コンペでの応用余地**:
  - ARC task 群を「Rotation / Symmetry / Color-mapping / Object-movement」の bucket に振り分けるための bootstrapping データ源
  - bucket 別に「最小 ONNX template」を 1 つずつ手書き → LLM 出力 description を template selector に使う pipeline が可能
- **限界 / 弱点**:
  - LLM が ARC を **十分に理解しない** ことが多い (= 公開記事 multiple sources で報告 / 公式 ARC-AGI Prize の base rate でも GPT-4 系は ≤30% 程度)
  - 自然言語 description から ONNX op に直接変換する「keyword-to-operator mapping」は **未実装**。本 kernel は description 生成までで止まっている
  - DeepSeek API key が必須で、Kaggle Notebook で再実行する際は user secret 設定が必要

## 出典

- Kernel URL: [https://www.kaggle.com/code/karnakbaevarthur/logic-decoder](https://www.kaggle.com/code/karnakbaevarthur/logic-decoder)
- 出力 dataset: [https://www.kaggle.com/datasets/karnakbaevarthur/logic-for-each-arc-task](https://www.kaggle.com/datasets/karnakbaevarthur/logic-for-each-arc-task)
- このディレクトリの `kernel-metadata.json` 参照
