# rag_retrieve.py ドキュメント

このドキュメントは、[scripts/rag_retrieve.py](../scripts/rag_retrieve.py) が「何をしているか」を手早く理解するためのメモです。

## これは何？
- **RAGの Retrieval（検索）だけ**を、最小構成で試すスクリプトです
- LLMは呼びません（APIキー不要）
- `data/chunk` 配下のチャンク（`.md` / `.txt`）から、クエリに近いものを上位K件返します

## できること / できないこと
- できること
  - チャンクの読み込み（Markdown + YAML front matter 対応）
  - TF-IDF + コサイン類似度で類似チャンクの順位付け
  - 上位結果の `title` / `source_path` / 先頭プレビューの表示
- できないこと（このスクリプトのスコープ外）
  - LLMへの問い合わせ（回答生成）
  - Embeddingモデルによる意味検索
  - フィルタ検索（タグや顧客で絞り込み等）

## 使い方

### 1回だけ検索（おすすめ）
```powershell
python scripts/rag_retrieve.py --data-dir data/chunk --query "VPN 接続 トラブル" --top-k 3
```

### 対話モード（複数クエリを連続で試す）
```powershell
python scripts/rag_retrieve.py --data-dir data/chunk
```

## 入力データ（チャンク）の前提
- 対象ディレクトリ: `--data-dir`（デフォルトは `data/chunk`）
- 拡張子: `.md` / `.txt`
- `.md` は先頭の YAML front matter（`--- ... ---`）があれば読み取ります

front matter の例:
```md
---
title: "製品X ライセンス更新"
product: "製品X"
topic: "ライセンス/更新/手順"
tags: ["ライセンス", "更新"]
source_path: "data/chunk/LicenseUpdate.md"
---

# 本文...
```

## 中でやっていること（処理の流れ）

### ステップ1: チャンク読み込み
- `load_chunks()`
  - `.md` は `parse_markdown_with_front_matter()` でメタと本文に分割
  - `.txt` はメタ無しで本文だけ読む

### ステップ2: 検索用コーパス作成
- `build_search_corpus()`
  - 本文だけでなく、`title/product/topic/tags` も一緒に混ぜた文字列を作ります
  - 目的: **短いクエリでも当てやすくする**

### ステップ3: TF-IDFでベクトル化
- `TfidfVectorizer(analyzer="char", ngram_range=(2, 4))`
  - 日本語は単語分割が難しいので、**文字2〜4gram** で特徴量を作ります

### ステップ4: コサイン類似度でスコア計算
- `cosine_similarity(query_vec, doc_matrix)`
  - ざっくり「ベクトルの向きが近いほどスコアが高い」指標です

## 次にやると良いこと（この後のステップ）
- 検索で上位K件を取り出したら、その本文を `CONTEXT` としてローカルLLMに渡す
- その際、回答に `source_path` を出させて「根拠」を残す

必要になったら
- `context.txt` を自動生成する
- JSONL（1行=1チャンク）に正規化する
- Embeddingに切り替える
…の順で拡張していくのがおすすめです
