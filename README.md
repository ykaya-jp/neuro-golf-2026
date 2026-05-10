# neurogolf-2026

**The 2026 NeuroGolf Championship** (IJCAI-ECAI 2026 Competitions Track) — ARC-AGI v1 公開 train 400 task を **解きつつ最小 ONNX neural network** を構築するコンペ。

- Kaggle URL: <https://www.kaggle.com/competitions/neurogolf-2026>
- GitHub: <https://github.com/ykaya-jp/neuro-golf-2026>
- Deadline: 2026-07-15 23:59 UTC
- 賞金: $50,000 (1st $12K + 2nd $10K + 3rd $10K + Top Student $8K + Longest Leader $10K)

## 評価

```
score_t = max(1, 25 - ln(cost_t))    for t in {1..400}
cost_t  = params(NN_t) + memory_bytes(NN_t)
total   = sum_t score_t              # 理論最大 = 10,000
```

詳細・数式・helper 関数は `docs/strategy/first-principles.dense.md` 参照。
公式 helper: `data/raw/neurogolf_utils/neurogolf_utils.py` (validator + scoring)

## Setup

```bash
make install                                 # uv sync
uv run kaggle competitions download -c neurogolf-2026 -p data/raw && unzip data/raw/*.zip -d data/raw/
```

(既に `data/raw/task001-400.json` が DL 済の場合は不要)

## Workflow

```bash
make eda                                     # notebooks/00_eda.ipynb
# baseline → /plan kaggle-neurogolf-2026-baseline (Claude Code 経由) で構築
```

## 研究遺産 (kaggle-onboard skill 出力)

| パス | 内容 |
|---|---|
| `docs/research/host_datasets.md` | host 公開 dataset + 400 task の EDA (N=101,718 pair) |
| `docs/research/past-comps.{dense,kids}.md` | 過去類似コンペ (ARC Prize / MicroNet / lottery ticket 系) の優勝解法 |
| `docs/research/public_kernels/INDEX.md` | 公開 kernel カタログ (technique 別) + 各 kernel の `content.md` |
| `docs/research/lb_snapshot_<date>.csv` | LB top 200 の日次 snapshot |
| `docs/discussion/insights.md` | discussion 統合解釈 |
| `docs/strategy/first-principles.dense.md` | scoring 数式 / ONNX 制約 / 不変条件 |
| `docs/strategy/exp001-design.md` | 構造原理が異なる 3+ 案 + critic 反論 |
| `ONBOARDING_CHECKLIST.md` | 6 軸進捗 + QG-1〜9 結果 |

## Public LB

| date | rank | score | comment |
|---|---|---|---|
| 2026-05-10 | (未提出) | - | bootstrap 段階 |

## 次のステップ

`/plan kaggle-neurogolf-2026-baseline` で baseline 実装 plan を起動 (= `.criteria/<task-id>.yaml` 生成 → 形名参同 Plan-Verify ループに入る)。
