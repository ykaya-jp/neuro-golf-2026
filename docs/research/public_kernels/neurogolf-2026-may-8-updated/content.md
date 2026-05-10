# neurogolf-2026-may-8-updated — 採点 rule 変更追従 + broadcast Mul/Add 修復

## [MD]

著者 konbu17 は **2026-05-06 の採点 rule 変更** を背景に置き、それに追従する multi-source per-task best-pick で **public LB 5571.69** を獲得したと報告している。原文の主要主張を要約引用する。

> 採点 rule 変更の背景:
> 5 月 6 日、公式 `neurogolf_utils.py` の目的関数が `25 - log(macs + memory + params)` から `25 - log(max(1, memory + params))` に切り替わり、**MACs が採点に寄与しなくなった**。memory は ONNX Runtime profiler trace から読まれる仕様で、zero-cost network は満点 25 点を得る。MACs に penalize されていた Conv-heavy / fixed-shape rewrite が再び競争力を持つようになった。
>
> 5 月 7 日 baseline 5211.69 比 +276.45 ポイントの内訳:
> - **2026-04-29 broken-task の救出** (10 task): 二分探索で post-May-5 grader が broadcast `Mul`/`Add`/`Sub` (例: `(1, 10, 30, 30) × (1, 1, 30, 1)`) を含む ONNX を弾くと特定。10 task (t045/t067/t111/t159/t176/t192/t210/t256/t309/t320) は **`Tile` op を inject して shape を align** + opset-13 / IR-8 / BOOL 出力 normalization で全滅から復活
> - **新採点下での multi-source per-task swap** (138 task): cheaper / correct ONNX に置換。内訳は afr_5377 119 task / konbu17 local v62 14 task / sigmaborov 4 task / svanikkolli 1 task
> - 残された broken-task stub なし。全 task が post-May-6 の strict shape inference を pass する real ONNX

## [CODE]

```python
EXPECTED_FILE_COUNT = 400
EXPECTED_PUBLIC_SCORE = '5571.69'
EXPECTED_MANIFEST_SHA256 = 'a73756e5c47702db480dd14433179e7721a781dedeb2099a5556ac3b9104d0aa'

INPUT_ROOT = Path('/kaggle/input')
WORKING = Path('/kaggle/working'); WORKING.mkdir(exist_ok=True)
OUT = WORKING / 'submission.zip'

# dataset-shipped submission.zip があればそれを使う (= 再 zip 回避)
found_zips = list(INPUT_ROOT.rglob('submission.zip'))
if found_zips:
    src = found_zips[0]
    shutil.copy(src, OUT)
else:
    onnx_files = sorted(INPUT_ROOT.rglob('task*.onnx'))
    by_name = {}
    for f in onnx_files:
        if re.fullmatch(r'task\d{3}\.onnx', f.name) and f.name not in by_name:
            by_name[f.name] = f
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(by_name):
            z.write(by_name[name], arcname=name)

# manifest sha256 を assert で再現性検証
with zipfile.ZipFile(OUT) as z:
    names = sorted(n for n in z.namelist() if n.endswith('.onnx'))
    lines = [f'{n}\t{len(z.read(n))}\t{hashlib.sha256(z.read(n)).hexdigest()}' for n in names]
manifest_sha = hashlib.sha256('\n'.join(lines).encode()).hexdigest()

assert manifest_sha == EXPECTED_MANIFEST_SHA256, (manifest_sha, EXPECTED_MANIFEST_SHA256)
```

## 要点 (W2 抽出)

- **手法 (technique)**: ensemble_blending + grader-bug-rescue + scoring-rule-adaptation
- **score (LB)**: **5571.69** (= 2026-05-08 時点で公開 kernel として上位帯)。base baseline 5211.69 から +360 ポイント上昇
- **votes**: 48 (= 直近 update なので vote 集計途中)
- **核心アルゴリズム**:
  1. **broken-task rescue**: `t045/t067/t111/t159/t176/t192/t210/t256/t309/t320` の 10 task が grader の broadcast `Mul`/`Add`/`Sub` を弾くバグで死んでいたのを特定 (`(1, 10, 30, 30) × (1, 1, 30, 1)` 等の shape mismatch)
  2. 各 broken task に **`Tile` op を inject** して broadcast を消し、broadcast を許す `(1,10,30,30) × (1,10,30,30)` 形式に正規化
  3. 全 ONNX を **opset 13 / IR 8** に統一 + **BOOL 出力 normalization** (= grader が float 出力期待だが BOOL を許容しないバグへの対策)
  4. **2026-05-06 採点 rule 変更後** に MACs を含まない `25 - log(max(1, memory + params))` で再評価し、138 task で source swap (`afr_5377` 119 task / `konbu17 v62` 14 / `sigmaborov` 4 / `svanikkolli` 1)
- **特徴的な工夫**:
  - **bisection で grader bug の根本原因を特定** (= broadcast `Mul`/`Add`/`Sub` の shape pattern 限定 reject) → 後続 kernel が同じ罠を踏まなくなる重要発見
  - **per-broken-task の修復策が再現可能** (= `Tile` op で broadcast 消す + opset/IR normalization)
  - 採点 rule 変更を **数式レベルで明示** し、MACs 重視だった旧戦略 (= Conv-heavy 解を避けていた) を逆転 → Conv 解が逆に有利になったと指摘
  - LB progression テーブル (= May 4 5264 → May 5 5156 → … → May 8 5571.69) で **採点 rule 変更前後の LB 動き** を完全公開
- **当該コンペでの応用余地**:
  - **採点 rule 変更追従**: 我々が cost 関数を実装する際は **必ず最新の `neurogolf_utils.py` を読み直し**、MACs 含むかは毎週 verify
  - **broken-task list (= t045/t067/t111/t159/t176/t192/t210/t256/t309/t320)** は我々の最初の baseline でもチェック必須
  - **`Tile` op で broadcast 消す pattern** は ONNX 生成時の defensive style として採用すべき
- **限界 / 弱点**:
  - manifest hash 固定なので「この notebook を再実行しても同じ score を再現する」ことしかできない (= 改善には別 source 投入が必要)
  - broadcast bug が grader version で再修正されると、`Tile` inject の冗長さが残る (= 余計な params/bytes でコスト上がる)

## 出典

- Kernel URL: [https://www.kaggle.com/code/konbu17/neurogolf-2026-may-8-updated](https://www.kaggle.com/code/konbu17/neurogolf-2026-may-8-updated)
- このディレクトリの `kernel-metadata.json` 参照
