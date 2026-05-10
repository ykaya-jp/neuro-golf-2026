# neurogolf-5480-41-current-rules-score — 公開 family 圧縮転移と manifest 公開

## [MD]

著者 afr1ste は本 kernel を「現行採点 rule 下で **public score 5480.41 を再現する parser-clean な artifact** の公開」と位置付け、attached dataset に正確な submission.zip (= 400 ONNX) を同梱する。手法サマリと運用ルールの原文を要約引用する。

> 現行 scoring/parser rule 下で確認された public score: **5480.41**。attached dataset は提出時の submission.zip (400 ONNX) を完全に保持している。
>
> 手法スケッチ:
> 公開 family の **per-task 圧縮転移** + dataset マイニングで構成。具体的には task128 = Jazivxt、task037 = Konbu 5 月 7 日、task310 = Soban、21-task batch (Konbu 5 月 7 日)、7-task batch (Dimok / Imaad / Soban)、4-task public tail、task127 (limprog dataset)。task089 / task017 への negative probe と旧 localtree の sparse / selector variant は **意図的に除外** する。
>
> 運用ルール:
> 「scored / parser-clean な artifact を exact manifest 付きで公開する」のがシンプルかつ実用的。広範な未検証 rewrite は避け、**local cost を最終判断にせず public score を信じる**。

## [CODE]

```python
EXPECTED_FILE_COUNT = 400
EXPECTED_PUBLIC_SCORE = "5480.41"
EXPECTED_ZIP_SHA256 = "25f12cd90994fdaae60faa4f394b141a2ae4f4b6407103bb4bcfc1a5b0ea6fa7"
EXPECTED_MANIFEST_SHA256 = "e66d6e7a4a0fa7b3f14cf8e7c13a513e1852b3c6eaa36533d9076400e8d8dbe9"
OUTPUT_ZIP = Path("submission.zip")

def manifest_from_zip(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(name for name in zf.namelist() if name.endswith(".onnx"))
        lines = []
        for name in names:
            data = zf.read(name)
            lines.append(f"{name}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}")
    manifest_hash = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return names, manifest_hash

# zip を見つけたら shutil.copy で /kaggle/working/submission.zip に置き、sha256 を assert
# 見つからなければ task*.onnx を集めて再 zip
```

manifest の SHA256 を hard-code し、再現できなければ assert で死ぬという「**確定スコアの再現性ガード**」を最重視する設計。

## 要点 (W2 抽出)

- **手法 (technique)**: ensemble_blending (= 公開 family の per-task 圧縮転移のみで構成、自前 train なし)
- **score (LB)**: 5480.41 (= 2026-05-07 時点の現行採点ルール下で確認済み public score)
- **votes**: 82
- **核心アルゴリズム**:
  1. 既知の高スコア public family (Jazivxt / Konbu May 7 / Soban / Dimok / Imaad / limprog) から **task ごとに最小コスト解を選んで copy**
  2. negative probes (= task089 / task017 / 旧 localtree sparse / selector) は意図的に **除外**
  3. submission.zip の SHA256 manifest を発行して再現性を確定する
- **特徴的な工夫**:
  - **manifest hash を埋め込んで「これが確かに 5480.41 を出した zip」を機械的に証明可能**にした (= 公開 LB の credibility 担保)
  - 「local cost を final judge にするな (= public LB を信じろ)」という運用原則をコメントで明示。これは MACs 採点削除 (2026-05-06) を経験した教訓
  - 失敗 source (negative probes) を明示的にリストアップ → 後続 kernel が「混ぜると score が落ちる task」を回避できる
- **当該コンペでの応用余地**:
  - 我々の戦略の **public LB 安定基準点**。ここから差分を積めば +∆ public LB が得られる
  - 「local cost と public score がズレる場合は public 優先」という運用ルールを採用すべき
- **限界 / 弱点**:
  - **新規 task は 1 つも solve していない**。既存 family を再配布しているだけなので独創性ゼロ
  - 5480.41 を超えるには別 source からの転移か、自前 solver が必要
  - 失敗 task の理由を analyse していない (= negative probes を除外した理由が公開されていない)

## 出典

- Kernel URL: [https://www.kaggle.com/code/afr1ste/neurogolf-5480-41-current-rules-score](https://www.kaggle.com/code/afr1ste/neurogolf-5480-41-current-rules-score)
- このディレクトリの `kernel-metadata.json` 参照
