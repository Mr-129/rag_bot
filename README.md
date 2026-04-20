# 製品X Small RAG Bot (FastAPI + OpenAI互換API)

社内向けの最小RAGチャットボットです。

- Web UI で質問し、LLMに回答させる
- `data/llm_config.json` で接続先を切り替え可能（Ollama / OpenAI / Azure OpenAI / GitHub Copilot 等）
- RAG ON 時は `data/chunk` の文書を検索して根拠付きで回答する
- 回答モード（`strict` / `hybrid` / `general`）を選択できる
- `doc_type` に応じて回答スタイルを切り替える（`troubleshooting` は類似事例ベースで回答）
- 手順系の文脈不足を補うため、前後チャンク補完（Phase1）と条件付き再読（Phase2）を実装
- Cross-Encoder Rerank（bge-reranker-v2-m3）で検索結果を精密にリランキング
- チャンクオーバーラップ（100文字）で分割境界の情報損失を防止
- システムプロンプト・同義語辞書の外部ファイル化による運用柔軟性
- API (`/api/chat`) からも同じ処理を呼び出せる
- すべてOpenAI互換API（`/v1/chat/completions`, `/v1/embeddings`）で統一

## ドキュメント構成（最初にここだけ確認）

- 全体運用ルール: `docs/DOCUMENTATION.md`
- 最短で動かす/使う: `README.md`（この文書）
- 学習しながら理解する: `docs/hands_on_rag_flow.md`
- 設計原則を確認する: `docs/rag_architecture_guide.md`
- チャンクのメタ設計を揃える: `docs/chunk_metadata_guideline.md`
- 今後の開発課題と優先度: `docs/rag_support_roadmap.md`
- Dify/NotebookLM水準を目指す改善計画: `docs/rag_improvement_roadmap.md`
- 変更履歴と次アクション: `docs/project_status.md`

更新ルール（推奨）:
- 実装変更時は、まず `README.md` を正本として更新
- 実施記録は `docs/project_status.md` に追記
- 学習資料/設計資料は必要な差分だけ反映

## このリポジトリの構成

- `app.py`: FastAPI本体（UI配信、LLM呼び出し、RAG検索）
- `static/`: UI (`index.html`, `app.js`, `style.css`)
- `data/chunk/`: RAG用の元ドキュメント（`.md` / `.txt`）
- `data/chunks.jsonl`: 検索対象のJSONL
- `scripts/rag_build_jsonl.py`: `data/chunk` から `chunks.jsonl` を生成
- `scripts/rag_retrieve.py`: 検索だけを単体確認するスクリプト
- `scripts/eval_retrieval.py`: 検索精度の評価スクリプト（tfidf/bm25/hybrid/embedding/hybrid3）
- `scripts/tune_retrieval.py`: パラメーターグリッドサーチ（BM25 k1/b, TF-IDF ngram, hybrid alpha）
- `scripts/build_embeddings.py`: Ollama 経由でチャンク埋め込みベクトルを生成（オフラインツール）
- `scripts/check_quality.py`: front matter メタデータの品質チェック
- `scripts/check_source_match.py`: 評価セット↔チャンク間の source_path 整合チェック
- `scripts/show_failures.py`: 評価失敗問の一覧表示・モード別比較
- `scripts/_test_integrity.py`: データファイル整合性テスト
- `scripts/_test_app.py`: アプリケーション機能テスト
- `scripts/_test_rerank.py`: Rerank統合テスト（9項目）
- `scripts/_eval_rerank_fast.py`: Rerank付き検索精度の高速評価（LLM不要）
- `data/prompts/`: 外部化されたシステムプロンプト（6ファイル）
- `data/synonyms.json`: 外部化された同義語辞書（7グループ）
- `data/llm_config.json`: LLM接続設定（プロバイダ切り替え用）
- `docs/`: 設計・学習・進捗のドキュメント群
- `rag_build_jsonl.py` / `rag_retrieve.py`: 旧コマンド互換ラッパー

## 開発者向け（コードコメント）

- **目的**: 各スクリプトと `app.py` に詳細なdocstring/モジュール説明を追加し、保守性を高めました。
- **該当ファイル**: `app.py`, `scripts/rag_build_jsonl.py`, `scripts/rag_retrieve.py`, ルートの互換ラッパー (`rag_build_jsonl.py`, `rag_retrieve.py`)。
- **内容**: BM25用トークナイザやスコア計算など重要関数にdocstringを追加。ルートラッパーには "互換性ラッパー" の説明を追記しました。


## 前提

- Python 3.10+
- LLM バックエンド（以下いずれか）:
  - **Ollama**（既定）: ローカル起動済み（`http://localhost:11434`）
  - **OpenAI API**: API Key を取得済み
  - **Azure OpenAI**: エンドポイントと API Key を取得済み
  - **GitHub Copilot**: API Key を取得済み
- LLM 接続先は `data/llm_config.json` で切り替え（詳細は「LLM接続設定」セクション参照）

Ollama 使用時の例:
```powershell
ollama pull gemma3:4b
```

## セットアップ

```powershell
pip install -r requirements.txt
```

## 使い始め手順（最短）

### 1. JSONLを生成

```powershell
python scripts/rag_build_jsonl.py --data-dir data/chunk --out data/chunks.jsonl --max-chars 700
```

### 2. サーバ起動

```powershell
uvicorn app:app --reload
```

### 3. アクセス

- UI: `http://127.0.0.1:8000/`
- API Docs: `http://127.0.0.1:8000/docs`

## API

### `POST /api/chat`

入力例:

```json
{
  "message": "ライセンス更新の手順を教えて",
  "model": "gemma3:4b",
  "rag": true,
  "top_k": 2,
  "mode": "strict",
  "retrieval_mode": "hybrid",
  "history": [
    {"role": "user", "content": "前の質問"},
    {"role": "assistant", "content": "前の回答"}
  ]
}
```

出力例:

```json
{
  "reply": "...",
  "sources": [
    {
      "rank": 1,
      "title": "製品X ライセンス更新",
      "source_path": "data/chunk/LicenseUpdate.md",
      "score": 0.51
    }
  ],
  "rag": true,
  "top_k": 2,
  "mode": "strict",
  "retrieval_mode": "hybrid"
}
```

`POST /chat` は互換エンドポイントです（`/api/chat` と同じ）。

### `POST /api/chat/stream`

SSE（Server-Sent Events）によるストリーミング応答エンドポイント。
入力は `/api/chat` と同じ JSON ボディ。レスポンスは `text/event-stream` で、トークン単位で逐次返却される。

イベント形式:

```
data: {"token":"ライセンス","done":false}
data: {"token":"更新の","done":false}
...
data: {"token":"","done":true,"sources":[...]}
```

- 各行は `data: {JSON}\n\n` 形式
- `done: true` の最終イベントに `sources` 配列が付与される（RAG ON 時のみ）
- エラー時は `data: {"error":"..."}\n\n` を返却
- UI はデフォルトでこのエンドポイントを使用する

## RAGの挙動（この実装のポイント）

- 優先: `RAG_INDEX_PATH` にjoblib索引があればそれを使用
- フォールバック: joblibが無ければ `data/chunks.jsonl` をその場でTF-IDF索引化
- 検索対象制限: `RAG_ALLOWED_SOURCE_PREFIXES`（既定: `data/chunk/`）
- `top_k` は 1〜20 の範囲
- `chunks.jsonl` は見出し単位で機械分割され、`--max-chars`（既定: `700`）超は再分割される
- `chunks.jsonl` は任意メタ `related_ids`（関連チャンクID配列）を保持可能（現時点では予約メタ。参照展開は未使用）
- 日付メタは `created_at` を正本として扱い、互換のため JSONL には `created_at` と `created` の両方を保持する
- `mode=strict` は CONTEXT範囲のみで回答する
- `mode=hybrid` は CONTEXT優先で回答し、不足分のみ一般知識を参考情報として補足する
- `mode=general` は RAG文書を参考にしつつ、CONTEXTに縛られない汎用回答を返す（コード生成・翻訳・分析も可能）
  - UI のデフォルトは `general`。API のデフォルトは `strict`（既存利用者への互換性維持）
- `doc_type=troubleshooting` が優勢なときは、回答を「類似事例 → 事例概要 → その時の対応 → 今回考えられる可能性 → 確認事項 → 根拠」の順で返す
- `retrieval_mode=tfidf|bm25|hybrid|hybrid3` で検索アルゴリズムを切替できる
  - `hybrid3`（既定）: TF-IDF + BM25 のキーワードスコアと bge-m3 Embedding スコアをブレンド
  - `hybrid`: TF-IDF と BM25 を minmax正規化後に alpha ブレンド
  - `RAG_HYBRID_ALPHA`（既定 `0.8`）: TF-IDF の重み。1-alpha が BM25 の重み
  - `RAG_EMBEDDING_ALPHA`（既定 `0.7`）: hybrid3 時の Embedding 重み。1-alpha がキーワード重み
  - `RAG_EMBEDDING_MODEL`（既定 `bge-m3`）: Embeddingモデル名
  - 2026-04-08 最新精度: hybrid3 + Rerank(bge-reranker-v2-m3) で **P@1=0.925, MRR=0.963**（80問）
  - hybrid3(Rerankなし): P@1=0.912, MRR=0.956（80問）
  - hybrid(alpha=0.8): P@1=0.863, MRR=0.909（80問）
  - チューニング詳細: `scripts/eval_retrieval.py`, `eval_tuning_results.csv`
- Rerank: 初回検索で広めに候補を取得（`RAG_RERANK_INITIAL_K=10`）し、Cross-Encoder（`bge-reranker-v2-m3`）で精密にリランキングして最終 top_k 件を返す
  - `RAG_ENABLE_RERANK=0` で無効化可能。`sentence-transformers` 未インストール時は自動的に無効化
  - 初回起動時にモデルを自動ダウンロード（~2.3GB、`~/.cache/huggingface/` にキャッシュ）
- チャンクオーバーラップ: `--overlap 100`（既定100文字）で分割境界の情報損失を防止
- システムプロンプト外部化: `data/prompts/` 配下の `.txt` ファイルで回答スタイルを管理（6パターン）
- 同義語辞書外部化: `data/synonyms.json` で同義語グループを管理。ファイル不在時は内蔵辞書にフォールバック
- Phase1: 検索ヒット後、同一ファイルの前後チャンク（`chunk_index ± window`）を補完する
- Phase2: 文脈不足または手順系質問時に、元ファイル（md/txt）を条件付き再読して補足する
- 会話履歴（マルチターン）:
  - `history` にフロントエンドが蓄積した過去のやり取りを配列で送信する
  - LLM には system + history + 今回の質問 の messages で問い合わせる
  - 指示語（「それ」「さっき」等）を検出すると、直前の質問を検索クエリに結合して検索精度を維持する
  - `RAG_MAX_HISTORY_TURNS`（既定 5）で履歴量を制限し、トークン爆発を防止する
  - 「履歴をクリア」ボタンで画面と履歴を同時リセット

## LLM接続設定

`data/llm_config.json` を編集するだけで接続先を切り替えられます。すべて OpenAI互換API で統一されているため、コード変更は不要です。

```json
// Ollama（既定・変更不要）
{"base_url": "http://localhost:11434/v1", "api_key": "", "model": "gemma3:4b", "embedding_model": "bge-m3", "temperature": 0.2, "timeout": 300}

// OpenAI (例: 実運用では `api_key` をファイルや環境変数に直接含めないでください)
{"base_url": "https://api.openai.com/v1", "api_key": "<REDACTED>", "model": "gpt-4o-mini", "embedding_model": "text-embedding-3-small", "temperature": 0.2, "timeout": 60}

// Azure OpenAI (実運用ではキーを環境変数 or data/llm_config.json に設定し、公開リポジトリには含めないこと)
{"base_url": "https://<resource>.openai.azure.com/openai/deployments/<deployment>/v1", "api_key": "<REDACTED>", "model": "gpt-4o", "embedding_model": "text-embedding-3-small", "temperature": 0.2, "timeout": 60}

// GitHub Copilot
{"base_url": "https://api.githubcopilot.com", "api_key": "<REDACTED>", "model": "gpt-4o", "embedding_model": "text-embedding-3-small", "temperature": 0.2, "timeout": 60}
```

| キー | 説明 |
|------|------|
| `base_url` | OpenAI互換APIのベースURL（`/v1` まで含む） |
| `api_key` | API Key（Ollama は空文字で可） |
| `model` | チャット用モデル名 |
| `embedding_model` | Embedding用モデル名（環境変数 `RAG_EMBEDDING_MODEL` でも上書き可） |
| `temperature` | 生成温度（既定 0.2） |
| `timeout` | 応答タイムアウト秒数（既定 300） |

フロントエンドのモデル名デフォルトは `/api/llm-config` エンドポイントから自動取得されます。

> **後方互換**: JSON ファイルがない場合は環境変数 `OLLAMA_URL` / `OLLAMA_TIMEOUT` にフォールバックします。

## 環境変数一覧

| 変数名 | 既定値 | 説明 |
|--------|--------|------|
| `LLM_CONFIG_PATH` | `data/llm_config.json` | LLM接続設定ファイルのパス |
| `RAG_INDEX_PATH` | `data/ProductXManual_chunks_100/...` | joblib索引パス（なければ chunks.jsonl にフォールバック） |
| `RAG_TOP_K` | `2` | 取得チャンク数 |
| `RAG_MAX_CHUNK_CHARS` | `2500` | 1チャンクの最大文字数 |
| `RAG_MIN_SCORE` | `0.25` | スコアしきい値（これ未満は除外） |
| `RAG_RETRIEVAL_MODE` | `hybrid3` | 検索モード: `tfidf` / `bm25` / `hybrid` / `hybrid3` |
| `RAG_HYBRID_ALPHA` | `0.8` | hybrid 時の TF-IDF 重み（1-alpha が BM25 重み） |
| `RAG_EMBEDDING_ALPHA` | `0.7` | hybrid3 時の Embedding 重み（1-alpha がキーワード重み） |
| `RAG_EMBEDDING_MODEL` | `bge-m3` | Embeddingモデル名（`llm_config.json` でも設定可） |
| `RAG_EMBEDDINGS_PATH` | `data/embeddings_bge_m3.json` | 事前計算済み埋め込みベクトルのパス |
| `RAG_NEIGHBOR_WINDOW` | `1` | Phase1: 前後チャンク補完の窓幅 |
| `RAG_MAX_CONTEXT_DOCS` | `8` | LLMに渡す最大コンテキスト数 |
| `RAG_ENABLE_REREAD` | `1` | Phase2: 元ファイル再読の有効/無効 |
| `RAG_REREAD_MAX_FILES` | `1` | Phase2: 再読ファイル数上限 |
| `RAG_REREAD_MAX_CHARS` | `1800` | Phase2: 再読最大文字数 |
| `RAG_REREAD_MIN_CONTEXT_CHARS` | `900` | Phase2: 再読発動の文脈量閾値 |
| `RAG_ENABLE_QUERY_EXPANSION` | `1` | 同義語クエリ展開の有効/無効 |
| `RAG_BM25_K1` | `0.5` | BM25 の k1 パラメータ（TF飽和制御） |
| `RAG_BM25_B` | `0.3` | BM25 の b パラメータ（文書長正規化制御） |
| `RAG_SYNONYMS_PATH` | `data/synonyms.json` | 同義語辞書のパス（不在時は内蔵辞書を使用） |
| `RAG_PROMPTS_DIR` | `data/prompts` | システムプロンプトのディレクトリ |
| `RAG_ENABLE_RERANK` | `1` | Cross-Encoder Rerank の有効/無効 |
| `RAG_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Rerank モデル名（HuggingFace） |
| `RAG_RERANK_INITIAL_K` | `10` | Rerank 用の初回検索候補数 |
| `RAG_MAX_HISTORY_TURNS` | `5` | 会話履歴の最大往復数（トークン爆発防止） |
| `RAG_ALLOWED_SOURCE_PREFIXES` | `data/chunk/` | 検索対象のソースパス接頭辞 |
| `APP_LOG_DIR` | `Log` | ログ出力先ディレクトリ |
| `APP_LOG_FILE` | `app.log` | ログファイル名 |
| `APP_LOG_LEVEL` | `INFO` | ログレベル |

## 今後の開発課題（要約）

- ~~評価基盤の整備~~ → **完了**（80問評価セット `data/eval_tuning.csv` + グリッドサーチ `scripts/tune_retrieval.py`）
- ~~中長期の検索改善: `bm25` と `tfidf` を比較して既定方式を再検討~~ → **完了**（hybrid 採用）
- ~~チャンクメタ品質の安定化~~ → **完了**（11件修正、check_quality.pyで自動検証）
- ~~評価セットの拡充~~ → **完了**（30問→80問、21/21ソースカバレッジ）
- ~~埋め込み検索の検証~~ → **完了・統合済み**（bge-m3 で hybrid3 モード実装、P@1=0.912）
- ~~パラフレーズ・口語クエリの検索精度改善~~（80問中11問が P@1=0）→ **改善**（hybrid3 で 7問→Rerank導入で 6問に削減）
- ~~チャンクオーバーラップ~~ → **完了**（100文字オーバーラップ、`--overlap` パラメータで設定可能）
- ~~BM25パラメータの外部化~~ → **完了**（`RAG_BM25_K1`, `RAG_BM25_B` 環境変数）
- ~~同義語辞書の外部化~~ → **完了**（`data/synonyms.json` + `RAG_SYNONYMS_PATH`）
- ~~システムプロンプトの外部化~~ → **完了**（`data/prompts/*.txt` + `RAG_PROMPTS_DIR`）
- ~~汎用モード（general）の追加~~ → **完了**（RAG参照+自由生成、UIデフォルト）
- ~~Rerank モデル導入~~ → **完了**（bge-reranker-v2-m3、P@1=0.912→0.925）
- ~~LLMバックエンド切り替え~~ → **完了**（`data/llm_config.json` で Ollama/OpenAI/Azure/GitHub Copilot に切替可能、OpenAI互換APIで統一）
- 取得方式ごとのしきい値整理
  - hybrid モードではスコアが [0, 1] に正規化されるため `RAG_MIN_SCORE=0.25` で統一可能
  - tfidf / bm25 単体使用時は個別閾値の検討が残る
- トラブルシュート文書向けの検索軸追加
  - 見出しが共通化しやすいため、将来的に症状・画面名・テーブル名・原因などの軸を増やす

詳細は `docs/rag_support_roadmap.md` を参照。

## テンプレート互換ルール

- `title` `product` `topic` `doc_type` `tags` `source_path` は検索品質と出典表示に直結するため、改名する場合は生成スクリプトと検索コードを同時に確認する
- 本文の見出し構成を変えると、見出し単位のチャンク分割と `chunk_id` が変わる
- `created_at` は利用可能だが、旧データ互換のため JSONL では `created` も併記する
- `source_path` は許可パス判定と元ファイル再読に使うため、空欄運用にはしない

PowerShell例:

```powershell
# LLM接続は data/llm_config.json で設定（環境変数より推奨）
$env:APP_LOG_DIR = "Log"
$env:APP_LOG_FILE = "app.log"
$env:APP_LOG_LEVEL = "INFO"
$env:RAG_RETRIEVAL_MODE = "hybrid3"
$env:RAG_TOP_K = "3"
$env:RAG_MIN_SCORE = "0.25"
$env:RAG_ENABLE_QUERY_EXPANSION = "1"
$env:RAG_NEIGHBOR_WINDOW = "1"
$env:RAG_ENABLE_REREAD = "1"
$env:RAG_EMBEDDING_ALPHA = "0.7"
$env:RAG_EMBEDDING_MODEL = "bge-m3"
$env:RAG_EMBEDDINGS_PATH = "data/embeddings_bge_m3.json"
$env:RAG_BM25_K1 = "0.5"
$env:RAG_BM25_B = "0.3"
$env:RAG_SYNONYMS_PATH = "data/synonyms.json"
$env:RAG_PROMPTS_DIR = "data/prompts"
$env:RAG_ENABLE_RERANK = "1"
$env:RAG_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
$env:RAG_RERANK_INITIAL_K = "10"
```

`RAG_ENABLE_QUERY_EXPANSION=1` の場合、質問文に含まれる語（例: 更新/アップデート、重い/遅い、接続/通信）を同義語で補強して検索し、類義語に弱い問題を軽減します。

## LAN公開（同一ネットワーク）

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

必要なら管理者PowerShellでFW許可:

```powershell
New-NetFirewallRule -DisplayName "製品X RAG UI (8000)" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -Profile Domain,Private -RemoteAddress LocalSubnet
```

## ローカル専用で使う（推奨: 社内セキュリティ制約がある場合）

ESET などの社内セキュリティ製品配下では、LAN公開時に TCP/8000 が遮断されることがあります。
その場合は公開せず、ローカルループバックで運用してください。

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

- UI: `http://127.0.0.1:8000/`
- API Docs: `http://127.0.0.1:8000/docs`

補足:
- `0.0.0.0` はLAN公開用です（クライアント接続が必要な場合のみ）
- ローカル運用ではFW/ESETの影響を受けにくく、切り分けも容易です

## 停止

```powershell
$pid = (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess)
if ($pid) { Stop-Process -Id $pid -Force; "Stopped pid=$pid" } else { "No process listening on 8000" }
```

## トラブルシュート

- `RAG索引がロードされていません`:
  - `python scripts/rag_build_jsonl.py --data-dir data/chunk --out data/chunks.jsonl --max-chars 700` を再実行
- `Ollamaに接続できません`:
  - Ollamaが起動しているか確認 (`http://localhost:11434/api/tags`)
- 別PCからタイムアウト:
  - `--host 0.0.0.0` で起動しているか
  - ファイアウォール許可があるか
  - ネットワーク経路（VLAN/ACL）で遮断されていないか
