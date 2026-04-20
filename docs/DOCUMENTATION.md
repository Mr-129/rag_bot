# ドキュメント運用ガイド（製品X Small RAG Bot）

このリポジトリの文書が増えても、重複と更新漏れを防ぐための運用ルールです。

## 1. 文書の役割分担（Diátaxis準拠）

- Tutorial（学習手順）: `hands_on_rag_flow.md`
  - 目的: 学習者が手を動かして理解する
  - 含める内容: 手順、実行例、学習メモ
- How-to / Quickstart（実用手順）: `README.md`
  - 目的: 最短で動かす、設定する
  - 含める内容: セットアップ、API例、運用設定
- Explanation（設計思想/ベストプラクティス）: `rag_architecture_guide.md`
  - 目的: なぜその設計にするかを理解する
  - 含める内容: 原則、設計選択、拡張方針
- Status / Changelog（進捗管理）: `project_status.md`
  - 目的: いつ何を変更したかを残す
  - 含める内容: 更新履歴、既知課題、次アクション

## 2. 正本（Single Source of Truth）

- 環境変数・API仕様の正本: `README.md`
- 学習ステップの正本: `hands_on_rag_flow.md`
- 設計原則の正本: `rag_architecture_guide.md`
- 日次/週次の変更履歴の正本: `project_status.md`

## 3. 更新ルール（ベストプラクティス）

変更を入れるときは、次の順で更新する。

1. 実装変更（コード）
2. `README.md`（仕様・設定の更新）
3. `project_status.md`（実施ログと次アクション）
4. 必要に応じて `hands_on_rag_flow.md` / `rag_architecture_guide.md`

## 4. 重複を防ぐ記述ルール

- 同じ詳細説明を複数ファイルへコピペしない
- 他文書にある詳細は「要点 + リンク」にする
- 数値デフォルトは `README.md` の値を正とし、他文書は参照に留める

## 5. レビュー観点

- 役割逸脱がないか（Tutorialに運用仕様を詰め込みすぎていないか）
- 正本との不整合がないか（例: 既定値、API項目）
- 次の行動に繋がるか（Statusに次アクションがあるか）
