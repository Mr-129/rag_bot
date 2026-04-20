# チャンクメタデータ運用ガイド（product / topic / doc_type / tags）

この文書は、`data/chunk/*.md` の front matter に入れるメタデータの役割を揃えるための実務ガイドです。

## 1. 目的

- `product` / `topic` / `doc_type` / `tags` の重複を減らす
- 後で検索改善・集計・辞典開発に使える状態にする
- 「ほぼ同じ値を全部の項目に入れる」状態を解消する

## 2. 役割分担（最重要）

- `product`: 何の製品・モジュールか（対象システム）
- `topic`: 何の業務/技術テーマか（中分類）
- `doc_type`: 文書の用途・性質
- `tags`: 検索補助用の自由語（表記ゆれ・別名・固有語を含めてよい）

要点:
- `product/topic/doc_type` は管理用の固定ラベル
- `tags` は自由語（検索に効かせる語）

## 3. 推奨入力ルール

### 3.1 product

- 基本は製品名 + 必要なら版
- 例: `製品X`, `Tx51`, `Tx41`, `製品X+Oracle`
- 1文書につき1値を推奨（複数は本当に必要なときだけ）

### 3.2 topic

- 文書の主要テーマを 1〜2 個に絞る
- 例: `DB接続`, `ライセンス更新`, `製番管理`, `コマンドライン起動`, `テーブル定義確認`
- 迷ったら「質問者が検索窓に入れそうな業務語」を優先

### 3.3 doc_type

- `howto` だけに偏らせない
- 最小セット（推奨）:
  - `howto`: 手順説明
  - `troubleshooting`: 障害切り分け/復旧
  - `reference`: 仕様・対応表・定義情報
  - `faq`: Q&A形式
  - `checklist`: 実施前後の確認項目
  - `policy`: 禁止事項・運用ルール
  - `manual`:マニュアル

### 3.4 tags

- 同義語、表記ゆれ、略語、旧称を入れてよい
- 例: `製番`, `計画明細`, `注番`, `PORDER`, `SEIBAN`, `VMSEI`
- ただし `product/topic/doc_type` と同じ語の重複は最小限にする

## 4. 迷ったときの判定フロー

1. これは「対象製品名」か？ → `product`
2. これは「業務テーマ」か？ → `topic`
3. これは「文書の用途」か？ → `doc_type`
4. それ以外（検索に効く語）か？ → `tags`

## 5. 例（front matter）

```yaml
---
title: "製品X ORACLE接続設定"
product: "製品X+Oracle"
topic: "DB接続"
doc_type: "howto"
tags: ["Oracle", "接続", "設定", "tnsnames", "listener", "DB"]
source_path: "data/chunk/製品X_ORACLE接続.md"
related_ids: []
created_at: "2026-03-02"
updated_at: "2026-03-02"
---
```

障害対応なら:

```yaml
doc_type: "troubleshooting"
topic: "DB接続"
tags: ["接続失敗", "ORA-", "疎通", "設定確認"]
```

## 6. 現状データの移行方針（最小）

1. `doc_type=howto` の文書を棚卸しし、実際は障害対応なら `troubleshooting` へ変更
2. `topic` を 1〜2 個へ圧縮（タグとの重複を削る）
3. `tags` に固有語（画面名/テーブル名/略語）を寄せる
4. 更新後は `python rag_build_jsonl.py` を実行して再生成

補足:
- 新規データは `created_at` / `updated_at` を使う
- 旧データの `created` は生成スクリプト側で互換吸収してよい

## 7. 命名の一貫性ルール（簡易）

- 全角/半角や大小表記はどちらかに統一（例: `Tx51`）
- 同義語は `tags` に寄せ、`topic` では代表語だけを使う
- 不明な場合は空欄ではなく、暫定で `topic: unknown` を使う
