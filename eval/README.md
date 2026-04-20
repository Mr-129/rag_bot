# chank評価セット ひな形ガイド

このフォルダは、`data/chunks.jsonl` を対象にした評価セットのひな形を置く場所です。

## ファイル

- `chank_eval_set_template.csv`
  - 現在の `chunks.jsonl` から作成した初期ひな形（30問）
  - `needs_chunk=true` 行は「現行データでは評価不能で、追加チャンクが必要」なテーマ

## 評価・チューニング用リソース（`data/` 配下）

- `data/eval_tuning.csv`
  - 30問の評価セット（全9ソース文書カバー、direct/paraphrase/troubleshoot等の多様なクエリタイプ）
  - `scripts/eval_retrieval.py` および `scripts/tune_retrieval.py` から参照される
- `data/eval_sample.csv`
  - 3問の簡易評価セット（旧版）

## チューニングスクリプト

- `scripts/tune_retrieval.py`
  - BM25(k1, b)、TF-IDF ngram_range、hybrid alpha のグリッドサーチ
  - 実行例: `python scripts/tune_retrieval.py --chunks data/chunks.jsonl --eval data/eval_tuning.csv`
  - 出力: `eval_tuning_results.csv`（754設定の結果一覧、MRR順ソート）
- `scripts/eval_retrieval.py`
  - 単一設定での詳細評価（per-question の P@1, P@3, MRR, MAP）
  - 実行例: `python scripts/eval_retrieval.py --chunks data/chunks.jsonl --eval data/eval_tuning.csv`

## 2026-03-31 チューニング結果

| モード | パラメーター | P@1 | MRR | MAP |
|--------|-------------|-----|-----|-----|
| tfidf | ngram=(2,3) | 0.967 | 0.983 | 0.983 |
| bm25 | k1=0.5, b=0.3 | 0.967 | 0.983 | 0.983 |
| **hybrid** | **alpha=0.8, ngram=(2,3), k1=0.5, b=0.3** | **1.000** | **1.000** | **1.000** |

## カラム定義

- `eval_id`: 評価ID
- `query`: 質問文
- `query_type`: `direct` / `paraphrase` / `followup` / `troubleshoot` / `comparison` / `safety` / `missing`
- `expected_source_path`: 期待出典（不足時は空）
- `expected_heading_path`: 期待見出し（不足時は空）
- `must_include_keywords`: 回答に含めたい語（`;`区切り）
- `forbidden_keywords`: 禁止語（必要時のみ）
- `risk_level`: `low` / `medium` / `high`
- `needs_chunk`: 追加チャンクが必要なら `true`
- `needs_chunk_reason`: 不足理由
- `notes`: 補足

## 運用手順（最小）

1. まず `needs_chunk=false` の行だけでベースライン評価する
2. `FAIL` 上位を見て、辞典改善かチャンク追加かを切り分ける
3. `needs_chunk=true` の行を、次のチャンク整備バックログとして使う
4. チャンク追加後に `needs_chunk` を `false` にして再評価する

## 更新ルール

- `data/chunk` を更新したら `python rag_build_jsonl.py` を実行
- 追加したチャンクに対応する評価行を増やす
- 既存評価行の `expected_source_path` が変わった場合は必ず更新
