# 小さく始めるRAG（学習用・最小構成）作業メモ

このリポジトリで実際に行った「小さくRAGを作る」流れを、学習用に再現できる形でまとめたドキュメントです。

> この文書の役割: Tutorial（学習手順）
> 
> 仕様値（環境変数/APIの正本）は `../README.md` を参照してください。
> 文書運用ルールは `DOCUMENTATION.md` を参照してください。

目的は、いきなりLLMやベクトルDBに飛び込まず、まず **データ整備とRetrieval（検索）** の基礎を手元で理解することです。

---

## 0. 全体像（いまどこまでできている？）

現時点でできていること：
- チャンク（Markdown/TXT）を用意し、メタデータを付与できる
- 検索（Retrieval）をローカルで実行し、上位K件の候補を確認できる
- チャンク群を1行1チャンクのJSONL（中間成果物）に変換できる
- FastAPI + Ollama で回答生成まで動作している
- 回答モード（strict / hybrid）を切り替えられる
- Phase1（前後チャンク補完）とPhase2（条件付き再読）を実装済み
- **BM25検索を実装し、TF-IDFとBM25を加重合成する hybrid 検索モードを追加済み**
- **評価セット（data/eval_tuning.csv: 30問）を整備し、グリッドサーチによるパラメータチューニングを実施済み**
  - 最適パラメータ: hybrid（alpha=0.8）、ngram=(2,3)、BM25 k1=0.5 / b=0.3
  - 達成指標: P@1=1.000 / MRR=1.000 / MAP=1.000
- **UIから検索モード（hybrid / TF-IDF / BM25）を切替可能**

次の段階（改善対象）：
- チャンク数増加時の精度劣化を監視し、Re-Rank / embedding 検索を検討する
- ストリーミング応答、会話履歴要約など UX 改善を進める

---

## 1. データ方針（原本と成果物）

- 原本：問い合わせメモなど（data配下）
- 学習用成果物：チャンク化して読みやすく整形したもの（data/chunk配下）
- 中間成果物：JSONL（data/chunks.jsonl）

重要：
- 最初は手作業で整備して理解を優先
- ただし「JSONのエスケープ」などミスりやすいところはスクリプトで事故を避ける

---

## 2. チャンクの作り方（手作業）

### 2.1 チャンク置き場
- チャンクは data/chunk 配下に置く

例：
- data/chunk/LicenseUpdate.md
- data/chunk/AddFieldSerach.md
- data/chunk/Script.md
- data/chunk/TableLayoutOutput.md

### 2.2 メタデータ（MarkdownのYAML front matter）

Markdownの先頭に `---` で囲ったYAMLを置く（例）：

```md
---
title: "製品X ライセンス更新"
product: "製品X"
topic: "ライセンス/更新/手順"
doc_type: "howto"
os: ["Windows"]
tags: ["ライセンス", "更新", "手順書"]
source_path: "data/chunk/LicenseUpdate.md"
created: "2026-02-18"
---

# 本文...
```

メタデータの最小セット（おすすめ）：
- title（人が見てすぐ内容が分かる）
- source_path（出典、根拠提示に使う）
- tags（検索の手掛かり）

---

## 3. Retrieval（検索）だけをテストする

### 3.1 検索スクリプト
- 検索テスト用スクリプトは [scripts/rag_retrieve.py](../scripts/rag_retrieve.py)
- 解説は [rag_retrieve.md](rag_retrieve.md)

### 3.2 何をやっている？（要点）

- チャンク本文＋メタ（title/tags等）を混ぜた「検索用テキスト」を作る
- TF-IDFでベクトル化（日本語向けに **文字2〜3gram**）
- クエリも同様にベクトル化
- コサイン類似度で上位K件を出す

これは「embedding検索」ではなく、古典的なTF-IDF検索（まず理解しやすい）

### 3.3 実行例

```powershell
python scripts/rag_retrieve.py --data-dir data/chunk --query "VPN 接続 トラブル" --top-k 3
```

期待：
- 「ライセンス更新」関連チャンクが1位でヒットする

---

## 4. JSONL（1行=1チャンク）を作る

### 4.1 なぜJSONLを作る？

- チャンクを「機械的に扱いやすい形」に固定できる
- 後でベクトルDB投入や差分更新、根拠提示がしやすい
- ただしLLMに渡すときは、常に“上位K件のtextだけ”を渡す（全量を毎回渡すわけではない）

### 4.2 JSONL生成スクリプト

- 生成スクリプトは [scripts/rag_build_jsonl.py](../scripts/rag_build_jsonl.py)
- 出力は data/chunks.jsonl

### 4.3 実行例

```powershell
python scripts/rag_build_jsonl.py --data-dir data/chunk --out data/chunks.jsonl
```

### 4.4 出力スキーマ（最小）

- chunk_id: 一意ID（sha256）
- title
- source_path
- tags
- text（本文）

メタがあれば product/topic/doc_type/os/created も追加で出力する。

---

## 5. chunk_id（sha256:...）とは？

- `sha256:...` はハッシュ関数SHA-256の結果をIDとして使っている
- 入力（source_path/title/text）が少しでも変わるとIDも変わる
- 目的：チャンクの一意性、差分検知、追跡に使う

このリポジトリでは、[scripts/rag_build_jsonl.py](../scripts/rag_build_jsonl.py) で概ね以下の材料から作成：
- source_path + title + text

---

## 6. JSONLの簡易チェック（壊れていないか）

全行がJSONとして読めるかだけ確認（例）：

```powershell
python -c "import json; p='data/chunks.jsonl'; n=0
for line in open(p,'r',encoding='utf-8'):
  json.loads(line); n+=1
print('ok', n)"
```

---

## 7. 現在の回答生成フロー（実装済み）

実装済みフロー：
1) クエリを投げて **retrieval_mode**（hybrid / tfidf / bm25）に応じて上位K件を取得
2) スコア閾値と許可パスでフィルタ
3) Phase1で同一ファイルの前後チャンクを補完
4) Phase2で不足時のみ元ファイルを再読して補足
5) ローカルLLMに「質問＋CONTEXT」を渡して **ストリーミング（SSE）** で回答生成
6) トークン単位でフロントエンドへ逐次送出し、最後に sources を返却

ポイント：
- LLMは通常ローカルのファイルパスを勝手に読めないので、**パスではなく本文（CONTEXT）を渡す**

### 7.1 mode の使い分け（回答モード）

- `strict`: CONTEXT内のみで回答。根拠不足時は不足を明示。
- `hybrid`: CONTEXT優先で回答し、足りない部分のみ一般知識を参考情報として補足。

社内運用では、既定は `strict` を推奨。

### 7.2 retrieval_mode の使い分け（検索モード）

| モード | 内容 | 既定 |
|--------|------|------|
| `hybrid` | TF-IDFスコアとBM25スコアを alpha で加重合成（alpha=0.8 → TF-IDF 80%＋BM25 20%） | **既定** |
| `tfidf` | TF-IDF（文字2〜3gram）のみ | - |
| `bm25` | BM25（k1=0.5, b=0.3）のみ | - |

UIの「検索モード」セレクタからも切り替え可能。
hybrid はグリッドサーチ（754通り）で P@1=1.000 を達成した最適設定。

### 7.3 Phase1/2 の意図

- Phase1（前後チャンク補完）: 手順の分断を緩和し、抜け漏れを減らす。
- Phase2（条件付き再読）: 手順系質問や文脈不足時だけ再読し、過剰な全文投入を避ける。

---

## 8. よくあるつまずき（メモ）

- Python環境とpipのインストール先がズレると `import yaml` が失敗することがある
  - 対策：実行するPythonに対して `python.exe -m pip install ...` を使う

---

## 付録：関連資料

- 体系化した一般ガイドは [rag_architecture_guide.md](rag_architecture_guide.md)
- 検索パラメータのチューニングガイド: [search_weights_guide.md](search_weights_guide.md)
- 評価スクリプト: [../scripts/eval_retrieval.py](../scripts/eval_retrieval.py)
- グリッドサーチスクリプト: [../scripts/tune_retrieval.py](../scripts/tune_retrieval.py)
- 評価セット（30問）: [../data/eval_tuning.csv](../data/eval_tuning.csv)
