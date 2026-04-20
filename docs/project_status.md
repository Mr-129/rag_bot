# 製品X Small RAG Bot 作業記録（2026-02-24 / 更新: 2026-04-21）

> この文書の役割: Status / Changelog（実施履歴・既知課題・次アクション）
>
> 仕様の正本は `../README.md`、設計原則は `rag_architecture_guide.md`、
> 学習手順は `hands_on_rag_flow.md` を参照。

## 追記サマリ（2026-04-21）

### 機能レビュー・品質改善

プロジェクト全体の機能レビューを実施し、以下の改善を適用。

#### レビュー結果サマリ

| カテゴリ | 評価 | 詳細 |
|----------|------|------|
| コアロジック（app.py） | A | BM25/TF-IDF/hybrid/hybrid3 + Rerank、適切なエラーハンドリング、履歴インジェクション防御 |
| テストスイート | A | 75テスト全パス、データ整合性・ユニット・検索品質をカバー |
| ドキュメント | A | Diátaxis準拠、README/docs 間の整合性良好 |
| フロントエンド | A | DOMPurify によるXSS対策、SSEストリーミング、レスポンシブUI |
| セキュリティ | A | ハードコード秘匿情報なし、入力検証あり、ソースパス制限あり |
| 依存関係 | B → A | `pyyaml` 漏れを修正 |

#### 修正内容

| 項目 | 修正 |
|------|------|
| `requirements.txt` | `pyyaml` を追加（YAML front matter パースに必要） |

#### 未対応の軽微事項（将来課題）

- `.env.example` の整備（環境変数一覧のテンプレート化）
- `scripts/_test_*.py` の pytest スイートへの統合
- Dockerfile / デプロイガイドの作成

---

## 追記サマリ（2026-04-20）

### pytest テストスイートの整備（75テスト）

デモデータ（ITヘルプデスク）への全面差し替えに合わせ、`tests/` ディレクトリに pytest ベースの自動テストを新規作成。

#### テストファイル一覧

| ファイル | テスト数 | カバー範囲 |
|----------|---------|-----------|
| `tests/test_data_integrity.py` | 19 | chunks.jsonl 整合性、chunk ファイル、eval CSV、プロンプトファイル |
| `tests/test_build_jsonl.py` | 21 | `scripts/rag_build_jsonl.py` の各関数（front matter パース、見出し分割、文字数分割、チャンクID、JSONL変換） |
| `tests/test_app_unit.py` | 33 | `app.py` のロジック（SimpleBM25、minmax正規化、検索テキスト構築、トークナイズ、同義語展開、ヘルパー関数、インデックスロード、retrieve_topk） |
| `tests/test_retrieval_quality.py` | 3 | `eval_tuning.csv`（85問）を使った検索品質テスト（hybrid/tfidf/bm25 の Recall@5） |

#### 実行結果

```
75 passed in 5.12s
```

#### デモデータ差し替え

| 項目 | Before | After |
|------|--------|-------|
| チャンクファイル | 製品X固有データ（21ファイル） | ITヘルプデスクデモデータ（36ファイル） |
| chunks.jsonl | 244レコード | 213レコード |
| eval_sample.csv | 製品X向け質問 | ITヘルプデスク向け3問 |
| eval_tuning.csv | 80問（製品X） | 85問（ITヘルプデスク） |
| ドメイン | 製造業向け生産管理 | 汎用ITヘルプデスク |

#### 新規追加ファイル

| ファイル | 役割 |
|----------|------|
| `tests/__init__.py` | テストパッケージ初期化 |
| `tests/test_data_integrity.py` | データ整合性テスト（19テスト） |
| `tests/test_build_jsonl.py` | JSONLビルド処理ユニットテスト（21テスト） |
| `tests/test_app_unit.py` | app.py ロジックテスト（33テスト） |
| `tests/test_retrieval_quality.py` | 検索品質テスト（3テスト） |

---

## 追記サマリ（2026-04-10）

### LLMバックエンド切り替え機能の実装

Ollama 固有のAPI (`/api/chat`, `/api/embed`) から **OpenAI互換API** (`/v1/chat/completions`, `/v1/embeddings`) に統一し、`data/llm_config.json` で接続先を切り替え可能にした。

#### 変更概要

| 項目 | Before | After |
|------|--------|-------|
| Chat API | `/api/chat`（Ollama独自） | `/v1/chat/completions`（OpenAI互換） |
| Embedding API | `/api/embed`（Ollama独自） | `/v1/embeddings`（OpenAI互換） |
| Streaming形式 | NDJSON | SSE (`data: {...}`) |
| Payload | `"options": {"temperature": 0.2}` | `"temperature": 0.2`（トップレベル） |
| Response | `data["message"]["content"]` | `data["choices"][0]["message"]["content"]` |
| 設定管理 | 環境変数 `OLLAMA_URL`/`OLLAMA_TIMEOUT` | `data/llm_config.json`（環境変数フォールバックあり） |
| 認証 | なし | `Authorization: Bearer <api_key>`（設定時のみ） |
| 関数名 | `_call_ollama()` / `_call_ollama_rag()` / `_stream_ollama_tokens()` | `_call_llm()` / `_call_llm_rag()` / `_stream_llm_tokens()` |

#### 対応プロバイダ

| プロバイダ | `base_url` | `api_key` |
|-----------|-----------|----------|
| Ollama（既定） | `http://localhost:11434/v1` | 不要（空文字） |
| OpenAI | `https://api.openai.com/v1` | `sk-...` |
| Azure OpenAI | `https://<resource>.openai.azure.com/.../v1` | Azure Key |
| GitHub Copilot | `https://api.githubcopilot.com` | `ghu_...` |

#### 修正ファイル

| ファイル | 変更内容 |
|----------|---------|
| `app.py` | LLM設定読み込み（`_load_llm_config()`）、Embedding/Chat/Streaming関数をOpenAI互換に書き換え、`/api/llm-config` エンドポイント追加 |
| `static/app.js` | `fetch('/api/llm-config')` でモデル名デフォルトを動的取得 |
| `data/llm_config.json` | 新規: LLM接続設定ファイル |

#### 新規追加ファイル

| ファイル | 役割 |
|----------|------|
| `data/llm_config.json` | LLM接続設定（base_url, api_key, model, embedding_model, temperature, timeout） |

#### 新規追加環境変数

| 変数名 | 既定値 | 説明 |
|--------|--------|------|
| `LLM_CONFIG_PATH` | `data/llm_config.json` | LLM接続設定ファイルのパス |

#### 後方互換

- `OLLAMA_URL` / `OLLAMA_TIMEOUT` 環境変数は引き続きフォールバックとして使用可能
- `data/llm_config.json` が存在しない場合は従来の Ollama 直接接続と同等の動作
- スクリプト（`build_embeddings.py`、`eval_retrieval.py` 等）はオフラインツールのため変更なし

#### 精度への影響

APIレイヤーのみの変更のため、検索精度に変化なし:
- P@1 = 0.925（変化なし）
- MRR = 0.963（変化なし）

---

## 追記サマリ（2026-04-08）

### 精度改善ロードマップの実装（6項目完了）

Dify/NotebookLM 水準を目指した `docs/rag_improvement_roadmap.md` の改善項目を、工数が小さい順に実装・テスト・レビューを繰り返して実施。

#### 完了項目一覧

| # | 項目 | 内容 | テスト結果 |
|---|------|------|-----------|
| 1-B | チャンクオーバーラップ | `--overlap 100`（既定100文字）を `rag_build_jsonl.py` に追加。行境界で整列。244チャンク再生成・Embedding再計算済み | P@1=0.912（退行なし） |
| D2 | BM25パラメータ外部化 | `RAG_BM25_K1`（既定0.5）、`RAG_BM25_B`（既定0.3）を環境変数化 | 4/4 PASS |
| D3 | 同義語辞書の外部化 | `data/synonyms.json`（7グループ）に分離。`RAG_SYNONYMS_PATH` で指定可能。ファイル不在時は内蔵辞書にフォールバック | 3/3 PASS |
| D4 | システムプロンプト外部化 | `data/prompts/` 配下に6ファイル（strict/hybrid/general × 通常/troubleshooting）。`RAG_PROMPTS_DIR` で指定可能 | 動作確認済み |
| — | 汎用モード（general）追加 | `mode=general` で RAG文書を参考にしつつ自由回答（コード生成・翻訳等も可能）。UIデフォルト=general、APIデフォルト=strict | 9/9 PASS |
| 1-A | Rerank モデル導入 | `sentence-transformers` + `BAAI/bge-reranker-v2-m3`（Cross-Encoder）。初回検索 initial_k=10 → リランク → top_k 件に絞り込み。`RAG_ENABLE_RERANK=0` で無効化可能 | 9/9 PASS |

#### 精度評価結果（80問、eval_tuning.csv）

| 設定 | P@1 | MRR | 失敗問数 |
|------|-----|-----|---------|
| hybrid（改善前ベースライン） | 0.863 | 0.909 | 11 |
| hybrid3（Rerank導入前） | 0.912 | 0.956 | 7 |
| **hybrid3 + Rerank（initial_k=10）** | **0.925** | **0.963** | **6** |

#### 設計上の注意点

- **Reranker のグレースフルデグレード**: `sentence-transformers` 未インストール時は自動的に無効化（warning ログのみ）。既存環境への影響なし
- **初回起動時間**: Reranker モデル（~2.3GB）を HuggingFace から自動ダウンロード。2回目以降は `~/.cache/huggingface/` からロード
- **CPU 版 torch**: GPU なしでも動作するよう torch CPU 版を使用
- **initial_k=10 が最適**: initial_k=20 では改善なし（80問評価で同一スコア）
- **プロンプト運用**: `data/prompts/*.txt` を編集することで、app.py を変更せずに回答スタイルを調整可能
- **同義語運用**: `data/synonyms.json` を編集するだけでクエリ展開用語を追加可能

#### 新規追加ファイル

| ファイル | 役割 |
|----------|------|
| `data/prompts/general.txt` | general モード用プロンプト（CONTEXT非拘束） |
| `data/prompts/troubleshooting_general.txt` | troubleshooting版 general プロンプト |
| `data/synonyms.json` | 同義語グループ外部辞書（7グループ） |
| `scripts/_test_rerank.py` | Rerank 統合テスト（9項目） |
| `scripts/_eval_rerank_fast.py` | Rerank 付き検索精度の高速評価（LLM 不要） |

#### 新規追加環境変数

| 変数名 | 既定値 | 説明 |
|--------|--------|------|
| `RAG_BM25_K1` | `0.5` | BM25 の k1 パラメータ |
| `RAG_BM25_B` | `0.3` | BM25 の b パラメータ |
| `RAG_SYNONYMS_PATH` | `data/synonyms.json` | 同義語辞書のパス |
| `RAG_PROMPTS_DIR` | `data/prompts` | システムプロンプトのディレクトリ |
| `RAG_ENABLE_RERANK` | `1` | Cross-Encoder Rerank の有効/無効 |
| `RAG_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Rerank モデル名 |
| `RAG_RERANK_INITIAL_K` | `10` | Rerank 用の初回検索候補数 |

---

### B1/B2 高優先度バグ修正

レビューで検出した高優先度バグ B1・B2 を修正し、全 15 テスト項目で PASS を確認。

#### B2: FastAPI lifespan 移行

- 非推奨の `@app.on_event("startup")` を削除し、`lifespan` 非同期コンテキストマネージャに移行。
- `app = FastAPI()` → `app = FastAPI(lifespan=lifespan)` に変更。
- `from contextlib import asynccontextmanager` を追加。
- 起動時のエンベディングロード処理は `lifespan()` 内の `yield` 前に配置。
- 変更範囲: `app.py` のみ（インポート1行追加、startup関数→lifespan関数に置換）。

#### B1: 同期 retrieve_topk() のイベントループブロッキング解消

- `retrieve_topk()`（内部で同期 `httpx.post()` を呼ぶ）の呼び出し箇所2件を `await asyncio.to_thread()` でラップ。
  - `_call_llm_rag()` 内（非ストリーミング RAG 応答）
  - `_sse_chat_rag()` 内（SSE ストリーミング RAG 応答）
- `import asyncio` を追加。
- `retrieve_topk()` 関数自体は変更なし（呼び出し側のみの修正で影響範囲を最小化）。

#### 修正後テスト結果（15/15 PASS）

| テスト | 結果 |
|--------|------|
| GET / フロントエンド配信 + DOMPurify | PASS |
| 静的ファイル配信 (app.js) | PASS |
| バリデーション (top_k=0, bad mode, bad retrieval) | PASS ×3 |
| RAG OFF (1+1=?) | PASS |
| RAG ON hybrid3 (reply, sources, source_path) | PASS ×3 |
| SSE ストリーミング (200, event-stream, tokens, done, sources) | PASS ×5 |

---

## 追記サマリ（2026-04-06）

### アプリケーションレビュー・テスト結果

2026-04-06 に app.py（~1400行）、フロントエンド（index.html / app.js / style.css）、
補助スクリプト10本、データファイル3種を対象とした包括的レビューとテストを実施。

#### テスト実施内容

新規テストスクリプト2本（`scripts/_test_integrity.py`, `scripts/_test_app.py`）を作成して実行:

| テスト | 結果 | 備考 |
|--------|------|------|
| データ整合性（chunks/embeddings/eval） | PASS | 244チャンク, 244埋め込み, 80評価問, 重複なし, source_path全一致 |
| GET / フロントエンド配信 | PASS | HTML/CSS/JS正常配信, DOMPurify(XSS対策)読込確認 |
| GET /favicon.ico | PASS | 204返却（ログ抑制） |
| 静的ファイル配信 | PASS | app.js, style.css アクセス可 |
| POST /api/chat (RAG OFF) | PASS | 「1+1=2」正常応答 |
| POST /api/chat (RAG ON, hybrid3) | PASS | 検索スコア0.988, LLM応答正常 |
| バリデーション(top_k/mode/retrieval_mode) | PASS | 不正値全て400返却 |
| SSEストリーミング | PASS | トークン逐次配信正常動作 |
| /chat 互換エンドポイント | PASS | 正常応答 |
| 会話履歴＋指示語解決 | PASS | 「それについて」→直前質問結合を確認 |
| 履歴インジェクション防止 | PASS | role:"system" 混入を _sanitize_history で除去確認 |

#### 発見した問題点

**高優先度（修正推奨）:**

| # | 場所 | 問題 | 影響 |
|---|------|------|------|
| B1 | app.py `_compute_embedding_scores()` | ~~同期 `httpx.post()` が async イベントループをブロック~~ | **2026-04-06 修正済**: `asyncio.to_thread()` でラップ |
| B2 | app.py `@app.on_event("startup")` | ~~FastAPI 非推奨API（v0.109+）~~ | **2026-04-06 修正済**: `lifespan` コンテキストマネージャに移行 |

**中優先度:**

| # | 場所 | 問題 |
|---|------|------|
| B3 | scripts/show_failures.py | ファイルリーク（`open()` を `with` なしで使用）、ファイル不存在時にクラッシュ |
| B4 | scripts/_check_emb.py | エラーハンドリングなし、ハードコードパス `data/embeddings.json`（旧ファイル参照） |
| B5 | scripts/check_quality.py, check_source_match.py | ファイル不存在時にクラッシュ（try-except なし） |

**低優先度:**

| # | 場所 | 問題 |
|---|------|------|
| B6 | data/chunk/ | ディレクトリ名「chank」は「chunk」の誤字。`RAG_ALLOWED_SOURCE_PREFIXES` で両方許可済み |
| B7 | data/embeddings.json | nomic-embed-text（768次元・不使用）のファイルが残存。混同リスク |

#### セキュリティ評価

| 項目 | 評価 |
|------|------|
| XSS対策 | 良好 — DOMPurify + escapeHtml() 二重防御 |
| プロンプトインジェクション | 良好 — _sanitize_history でroleフィルタリング |
| パストラバーサル | 良好 — _resolve_source_file で BASE_DIR 外アクセスを遮断 |
| ソースパス制限 | 良好 — RAG_ALLOWED_SOURCE_PREFIXES でホワイトリスト制御 |
| 入力バリデーション | 良好 — top_k/mode/retrieval_mode 全てサーバ側で検証 |

#### 新規追加ファイル

| ファイル | 役割 |
|----------|------|
| `scripts/_test_integrity.py` | データファイル整合性テスト（chunks/embeddings/eval の一貫性検証） |
| `scripts/_test_app.py` | アプリケーション機能テスト（エンドポイント・バリデーション・SSE・セキュリティ） |

## 追記サマリ（2026-04-02 その2）

### Step 4: bge-m3 による Embedding 検索の再検証と app.py 統合

- nomic-embed-text（日本語性能不足、P@1=0.350）に代わり、**bge-m3**（BAAI製、多言語対応、1024次元）で再検証。
- `ollama pull bge-m3`（1.2GB）で導入し、`data/embeddings_bge_m3.json`（244チャンク × 1024次元）を生成。
- 評価結果:

| モード | alpha | P@1 | MRR | 備考 |
|--------|-------|-----|-----|------|
| hybrid | 0.8 | 0.863 | 0.909 | キーワードのみ（ベースライン） |
| embedding 単体（bge-m3） | — | 0.825 | 0.904 | nomic-embed-text の 0.350 から大幅改善 |
| **hybrid3** | **0.7** | **0.912** | **0.956** | **最高性能** |
| hybrid3 | 0.75 | 0.912 | 0.956 | 同等 |
| hybrid3 | 0.8 | 0.900 | 0.948 | やや劣化 |

- 失敗問の変化（hybrid → hybrid3(alpha=0.7)）:
  - **改善 6問**: E020, E063, E066, E067, E072, E074（パラフレーズ・口語クエリ）
  - **退行 2問**: E039, E052（プレリリース版関連の直接質問）
  - **未解決 5問**: E050, E062, E073, E077, E078
  - **純改善: 11問 → 7問**（P@1=0.863 → 0.912）
- **app.py への統合を実施**:
  - `_load_embeddings()` / `_compute_embedding_scores()` 関数を追加
  - `retrieve_topk()` に `hybrid3` モードを追加
  - 起動時に `data/embeddings_bge_m3.json` を自動ロード
  - デフォルト検索モードを `hybrid3` に変更
  - 環境変数追加: `RAG_EMBEDDING_ALPHA`、`RAG_EMBEDDING_MODEL`、`RAG_EMBEDDINGS_PATH`
  - UI(`index.html`) に `hybrid3（キーワード+Embedding）` 選択肢を追加
  - `requirements.txt` に `numpy` を追加

## 追記サマリ（2026-04-02）

### Step 1: チャンクメタデータ品質の監査と修正

- 全 21 ソースファイル・244 チャンクを対象にメタデータ品質監査を実施。
- 検出した 11 件の不整合を修正。
  - P1（5件）: `source_path` 不一致（コピーファイル名の "copy" サフィックスとの齟齬、別ファイルを指す誤参照）
  - P2（6件）: `tags` 空配列、`title` 重複、`topic`/`product` 形式違反
- 品質チェックスクリプト `scripts/check_quality.py` を新規作成。
- 修正後の再検証で不整合 0 件を確認。
- `data/chunks.jsonl` を再生成（239→244 チャンク）。

### Step 2: 評価セットの拡充（30→80問）

- 評価セットを `data/eval_tuning.csv` に 80 問へ拡充。
  - E001–E030: 既存30問（source_path バグ修正: E021, E022, E023, E029）
  - E031–E060: 未カバーだったソースファイル向けの直接質問（30問追加）
  - E061–E080: パラフレーズ・曖昧表現クエリ（20問追加）
- ソースカバレッジ: 21/21（全ソースファイルをカバー）。
- ベースライン計測: Hybrid(alpha=0.8) P@1=0.863, MRR=0.909（80問基準）。
- 評価↔チャンク整合チェックスクリプト `scripts/check_source_match.py` を新規作成。

### Step 3: Embedding検索の導入検証

- Ollama で `nomic-embed-text` モデルを導入（768次元）。
- `scripts/build_embeddings.py` を新規作成し、全244チャンクの埋め込みベクトルを生成（`data/embeddings.json`）。
- `scripts/eval_retrieval.py` に `embedding` と `hybrid3`（キーワード+埋め込みブレンド）モードを追加。
- `scripts/show_failures.py` にモード引数対応を追加。
- 評価結果:

| モード | alpha | P@1 | MRR | 備考 |
|--------|-------|-----|-----|------|
| hybrid | 0.8 | 0.863 | 0.909 | ベースライン |
| embedding 単体 | — | 0.350 | — | 日本語に弱い |
| hybrid3 | 0.05 | 0.863 | 0.909 | hybrid と同等（改善なし） |
| hybrid3 | 0.1–0.2 | 0.850 | — | わずかに劣化 |
| hybrid3 | 0.8 | 0.537 | 0.672 | 大幅劣化 |

- **結論**: `nomic-embed-text` は日本語テキストへの効果が極めて低い。
  - Embedding 単体で P@1=0.350（キーワード検索の半分以下）。
  - hybrid3(alpha=0.05) でもキーワード検索の失敗11問を1問も改善できず。
  - 失敗問は hybrid と hybrid3 で完全一致（差分ゼロ）。
- **判断**: nomic-embed-text による埋め込み検索の app.py 統合は見送り。日本語対応モデル（bge-m3 等）での再検証、またはキーワード検索の強化（同義語辞書拡充・クエリ書き換え）を次の候補とする。

### 新規追加ファイル

| ファイル | 役割 |
|----------|------|
| `scripts/build_embeddings.py` | Ollama 経由でチャンク埋め込みベクトルを生成 |
| `scripts/check_quality.py` | front matter メタデータの品質チェック |
| `scripts/check_source_match.py` | 評価セット↔チャンク間の source_path 整合チェック |
| `data/embeddings.json` | 244チャンク × 768次元の埋め込みベクトル（nomic-embed-text） |

## 追記サマリ（2026-03-31 その3）

### 会話履歴（マルチターン）の実装

- `ChatRequest` に `history` フィールドを追加（`list[dict] | None`）。
- フロントエンド（`app.js`）で `conversationHistory` 配列を管理し、毎回のリクエストに含める。
- バックエンド4関数（`_call_llm`, `_call_llm_rag`, `_sse_chat`, `_sse_chat_rag`）でLLM messages に履歴を組み込み。
- 指示語検出（「それ」「さっき」等）で直前のユーザー発言を検索クエリに結合する `_augment_query_with_history()` を追加。
- 履歴サニタイズ（role制限・空除去・往復数上限）で安全性を確保。
- 「履歴をクリア」ボタンで画面と内部履歴を同時リセット。
- 環境変数 `RAG_MAX_HISTORY_TURNS`（既定 5）で履歴上限を制御。
- テスト6件追加、全22テスト合格。

## 追記サマリ（2026-03-31 その2）

### SSEストリーミング応答の実装

- `POST /api/chat/stream` エンドポイントを新設。
  - Ollama `stream=True` → SSE (`text/event-stream`) でトークン単位中継。
  - 形式: `data: {"token":"...","done":false}\n\n`、完了時に `sources` 付き。
- フロントエンド（`app.js`）を `ReadableStream` ベースに書き換え。
  - トークン到着ごとにMarkdownを逐次レンダリング。
  - ステータス表示を「検索中…」→「推論中…」に分離。
- 既存 `POST /api/chat`（非ストリーミング）は互換のため残置。
- UI検索モードに `hybrid（TF-IDF+BM25）` をデフォルト追加。

## 追記サマリ（2026-03-31）

### 検索パラメーターチューニング実施

- 30問の評価セットを作成（`data/eval_tuning.csv`）。
  - 全 9 ソース文書をカバー、direct/paraphrase/troubleshoot 等の多様なクエリタイプ。
- `scripts/tune_retrieval.py` で 754 設定をグリッドサーチ。
  - BM25 k1=[0.5..2.0] x b=[0.3..0.85]
  - TF-IDF ngram_range: (2,3), (2,4), (2,5), (3,5)
  - Hybrid alpha: [0.3..0.8]
- 最適パラメーター:
  - **検索モード**: `hybrid`（TF-IDF+BM25 ブレンド）
  - **TF-IDF ngram_range**: `(2, 3)`（元: (2, 4)）
  - **BM25 k1**: `0.5`（元: 1.5）
  - **BM25 b**: `0.3`（元: 0.75）
  - **Hybrid alpha**: `0.8`（TF-IDF 80% + BM25 20%）
- 改善結果:
  - tfidf 単体: P@1=0.967, MRR=0.983
  - bm25 単体: P@1=0.967, MRR=0.983
  - **hybrid**: P@1=1.000, MRR=1.000, MAP=1.000
- `app.py` に hybrid モードを実装、デフォルトを hybrid に変更。
- `scripts/eval_retrieval.py` のデフォルトもチューニング後の値に更新。
- 環境変数 `RAG_HYBRID_ALPHA` を新規追加。

## 追記サマリ（2026-03-23）

- `data/chunk` の front matter をレビューし、検索補助タグを追加。
- `data/chunks.jsonl` を再生成し、タグ更新内容を検索対象へ反映。
- `LicenseUpdate.md` を使った取得確認を行い、次を整理。
  - タイトル一致は `tfidf` でも十分ヒットする
  - 自然文質問では、正解文書が1位でも `RAG_MIN_SCORE=0.45` を下回るケースがある
  - `bm25` は同種の質問で比較的安定する
- 今後の開発課題を `rag_support_roadmap.md` に整理。

## 0. 更新サマリ（2026-02-27）

- RAG検索品質改善のため、`rag_build_jsonl.py` を見出し単位分割 + 文字数上限再分割（既定700文字）対応に更新。
- `data/chunks.jsonl` を再生成し、1ファイル1チャンクから複数チャンク構成へ移行（現在12件）。
- `app.py` に検索ログ（質問・類似度・回答）を追加し、`Log/app.log` へファイル保存する仕組みを実装。
- UIをチャットアプリ風に再構成し、表示領域最大化・入力欄下固定のレイアウトへ調整。
- LAN公開は環境要因（ESET / ネットワーク経路）で制約があるため、ローカル運用を基本方針に整理。
- 回答モード `strict` / `hybrid` をAPIとフロントに追加。
- RAG文脈補完の Phase1/2 を実装。
  - Phase1: ヒットしたチャンクの前後チャンクを同一ファイル内で補完
  - Phase2: 手順系質問または文脈不足時のみ元ファイルを条件付き再読

## 1. 現在の到達点（完了済み）

### RAGデータ整備
- `data/chunk/*.md` から `data/chunks.jsonl` を生成できる状態。
- `rag_build_jsonl.py` を調整し、以下を実装済み。
  - `source_path` の正規化（`\\` → `/`）
  - `source_path` 未指定時の相対パス補完
  - `tags` の型ゆれ対策（文字列・不正型の吸収）
  - 見出し単位で機械分割
  - 文字数上限（`--max-chars`, 既定700）で再分割
  - `heading_path` / `chunk_index` を出力
  - コードブロック内の `#` を見出し誤判定しないよう修正
  - 見出し行だけの短小チャンクを除外
- `data/chunks.jsonl` は再生成済み（12件）。

### API / RAG実装
- `app.py` で `POST /api/chat` / `POST /chat` が動作。
- RAG有効時の検索・回答フローが動作。
- `load_index` / `retrieve_topk` / `build_messages` を実装済み。
- `RAG_INDEX_PATH` のjoblibが無い場合、`data/chunks.jsonl` へフォールバックするよう修正。
- 検索結果のソース制限を追加。
  - 環境変数: `RAG_ALLOWED_SOURCE_PREFIXES`
  - 既定: `data/chunk/`
- クエリ拡張（同義語展開）を追加。
- スコアしきい値を導入（`RAG_MIN_SCORE`, 既定0.45）。
- ログ出力を追加（質問・類似度・回答）。
- `Log/app.log` へのファイル出力を追加（`APP_LOG_DIR` / `APP_LOG_FILE` / `APP_LOG_LEVEL`）。
- 回答モードを追加。
  - リクエスト項目: `mode`（`strict` / `hybrid`）
  - レスポンスに `mode` を返却
- Phase1/2 を追加。
  - `RAG_NEIGHBOR_WINDOW`（既定1）
  - `RAG_ENABLE_REREAD` / `RAG_REREAD_MAX_FILES` / `RAG_REREAD_MAX_CHARS` / `RAG_REREAD_MIN_CONTEXT_CHARS`
  - `RAG_MAX_CONTEXT_DOCS`（既定8）

### 動作確認
- `http://127.0.0.1:8000/` でUI疎通（200）確認済み。
- `POST /api/chat`（RAG ON）で応答＋sources返却確認済み。
- `python scripts/rag_build_jsonl.py --data-dir data/chunk --out data/chunks.jsonl --max-chars 700` で再生成確認済み。

### ドキュメント
- `README.md` に以下を追記済み。
  - ローカル専用運用手順
  - クエリ拡張とスコアしきい値の設定
  - ログ出力設定（`APP_LOG_*`）
  - 回答モードとPhase1/2設定の説明

---

## 2. 既知の課題（現状）

1. 回答の質にばらつきがある
- Phase1/2で改善したが、検索ベースがTF-IDF中心のため語彙依存が残る
- 手順系で改善効果はあるが、質問によっては不足・冗長が発生

2. 類語・言い換え耐性に限界
- 現状はクエリ拡張で緩和しているが、意味検索ほどの耐性はない

3. LAN公開はネットワーク制約の影響を受ける
- ホスト側待受/FWルールは確認済みでも、クライアントから `TcpTestSucceeded=False` となるケースあり
- ESET / VLAN / ACL など経路側要因を排除できないため、当面はローカル運用を基本とする

4. 評価基盤が未整備 → **2026-04-02に完了**
- → 80問評価セット (`data/eval_tuning.csv`) を作成（30→80問に拡充）
- → グリッドサーチスクリプト (`scripts/tune_retrieval.py`) を追加
- → hybrid(alpha=0.8) で P@1=0.863（80問）、MRR=0.909 を達成
- → embedding / hybrid3 モードの評価もサポート

5. 追質問（フォローアップ）に弱い
- 現状のAPIは基本的に単発質問ベースで、会話履歴を検索クエリへ十分反映していない
- 「それ」「さっきの手順」など省略表現で検索が外れやすい

9. パラフレーズ・曖昧表現クエリの検索精度が課題 → **2026-04-02 確認**
- 80問評価で 11問が P@1=0（E020, E050, E062, E063, E066, E067, E072, E073, E074, E077, E078）
- いずれも口語的・間接的な表現で、キーワード一致が不足するパターン
- nomic-embed-text による意味検索では改善できなかった（日本語性能不足）
- 次の候補: 日本語対応埋め込みモデル（bge-m3 等）、同義語辞書の拡充、クエリ書き換え

10. 評価セットが 80 問に拡充済み → **2026-04-02 完了**
- 30問→80問（直接質問30問追加 + パラフレーズ20問追加）
- 21/21 ソースファイルをカバー

11. front matter の品質差 → **2026-04-02 解消**
- 全 21 ソースの front matter を監査し、11件の不整合（source_path/tags/title/topic/product）を修正
- `scripts/check_quality.py` で再検証 0 件を確認

6. 取得方式ごとのしきい値設計が未整理 → **部分解消**
- hybrid モードではスコアが [0, 1] に正規化されるため `RAG_MIN_SCORE=0.25` で統一的に運用可能
- tfidf / bm25 単体モードでの個別閾値は今後の課題として残る

7. front matter の品質差が検索精度へ直結している → **2026-04-02 解消**
- ✅ 全 21 ソース・244 チャンクの front matter を監査
- ✅ 11 件の不整合を修正（source_path 5件、tags/title/topic/product 6件）
- ✅ `scripts/check_quality.py` で再検証 0 件を確認

8. トラブルシュート文書の検索軸が不足している
- 見出しが共通化しやすく、見出しだけでは差別化しにくい
- 将来的に症状、画面名、テーブル名、原因などの専用軸を持たせる余地がある
---

## 3. 次にやる候補（実装優先順）

### 優先A（効果が高く小さく入れられる） → **2026-03-31に完了**
1. ~~評価セット作成（10〜20問）~~ → 30問で作成済み (`data/eval_tuning.csv`)
2. ~~`RAG_MIN_SCORE` と再読閾値の再チューニング~~ → hybrid導入で `RAG_MIN_SCORE=0.25` に統一
3. ~~BM25プロトタイプ導入~~ → hybrid モードとして TF-IDF + BM25 ブレンドで実装済み
4. ~~SSE ストリーミング応答~~ → `POST /api/chat/stream` 新設、フロントエンドも対応済み（16テスト全合格）
5. ~~UI検索モードに hybrid 追加~~ → UIセレクタに hybrid（TF-IDF+BM25）をデフォルト追加

### 優先B（中期）→ 次の対応課題
4. 再読ロジックの精緻化
- heading_path中心の局所再読精度を改善
- context_type別の採用上限を追加検討

5. パラフレーズ・口語クエリの検索精度改善
- 80問中11問が P@1=0（すべて口語的・間接的表現のキーワード不一致）
- 候補A: 日本語対応埋め込みモデル（bge-m3）での再検証（インフラ準備済み）
- 候補B: 同義語辞書の拡充（現行7グループ→20グループ以上）
- 候補C: LLMによるクエリ書き換え（口語→正式用語変換）

6. ~~評価セットの拡充（30問→50問以上）~~ → **2026-04-02 完了**（80問、21/21カバレッジ）

7. ~~会話履歴（マルチターン）対応~~ → **2026-03-31 完了**

### 優先C（将来）
8. 埋め込み検索への移行検討 → **初回検証済み（nomic-embed-text は不採用）**
- nomic-embed-text は日本語性能不足（P@1=0.350）
- 日本語対応モデル（bge-m3 等）で再検証する余地あり
- 評価インフラ（embedding / hybrid3 モード）は整備済み

---

## 4. 次回着手のおすすめ（最小）

- Step A: 日本語対応埋め込みモデル（bge-m3）でのEmbedding再検証
- Step B: 同義語辞書の拡充（パラフレーズ失敗11問の分析結果に基づく）
- Step C: 再読ロジックの精緻化（heading_path 局所再読の改善）
- 目標: P@1 ≥ 0.90（80問基準）

---

## 8. 完成度見積り（2026-04-06 更新）

### 総合評価
- **約 95%**（MVP＋α として実用レベル、検索精度目標を達成、高優先度バグ全件解消）
- 2026-02-27「75〜85%」→ 04-02「90%」→ bge-m3 統合「93%」→ B1/B2 修正「95%」
- 2026-04-06 レビュー: 全項目合格、B1/B2 修正完了。セキュリティ良好。

### 観点別

| 観点 | 完成度 | 状況 |
|------|--------|------|
| API / サーバ機能 | 100% | RAG ON/OFF、SSE、検索4モード、会話履歴、Phase1/2 全動作。lifespan 移行済、async 安全 |
| 検索エンジン | 93% | hybrid3(bge-m3) P@1=0.912, MRR=0.956（80問）。残り7問が課題 |
| データ品質 | 95% | 21ソース/244チャンク、front matter 監査済み（不整合0件） |
| 評価基盤 | 100% | 80問、5モード対応、失敗分析・整合チェック・チューニング完備 |
| フロントエンド | 95% | SSE、Markdown描画、DOMPurify、会話履歴、モード切替(hybrid3含む) |
| テスト | 75% | test_stream.py(22件) + _test_integrity.py + _test_app.py(11項目)。検索ロジック単体テスト不足 |
| ドキュメント | 95% | README + docs/ 8ファイル。主要仕様は網羅、最新の検索構成を反映 |
| セキュリティ | 90% | XSS(DOMPurify)、インジェクション防止、パストラバーサル防止、入力検証、asyncブロッキング解消。CORS/認証はローカル専用のため省略 |
| 運用性 | 85% | 環境変数23項目、ログ完備。デプロイ自動化・ヘルスチェック未着手 |

### 完了条件の達成状況
- ✅ 評価セット（80問）で再現可能なスコアが取れる
- ✅ 失敗パターンが説明可能（パラフレーズ・口語7問 = キーワード＋意味検索でもカバー不足）
- ✅ 運用設定の推奨値が文書化されている
- ✅ 再評価手順が確立されている（eval_retrieval.py + eval_tuning.csv）
- ✅ P@1 ≥ 0.90 を達成（hybrid3 bge-m3 P@1=0.912）
- ✅ 包括的レビュー・テスト実施済み（2026-04-06: 全項目合格、重大バグなし）
- ☐ 残り7問の失敗改善（P@1 ≥ 0.94 目標）
- ☐ 検索ロジックの単体テスト追加
- ✅ B1（同期httpx.post のブロッキング）修正 → `asyncio.to_thread()` で解消（2026-04-06）
- ✅ B2（FastAPI lifespan 移行）修正 → `lifespan` コンテキストマネージャに移行（2026-04-06）

---

## 5. 参照ファイル

- 実装本体: `app.py`
- JSONL生成: `scripts/rag_build_jsonl.py`
- データ: `data/chunk/*`, `data/chunks.jsonl`
- 更新README: `README.md`
- ロードマップ: `rag_support_roadmap.md`

---

## 6. 作業ログ（2026-02-26）

### ログ形式
- `[カテゴリ] 実施内容 -> 結果/確認`

### 本日の記録
- `[ネットワーク] LAN公開の疎通を調査 -> ホスト側待受/FWはOK、クライアント側はTCP到達不可（環境要因の可能性高）。`
- `[運用方針] 社内制約（ESET等）を考慮 -> ローカル運用（127.0.0.1）を基本とする方針に整理。`
- `[RAG] クエリ拡張（同義語）を実装 -> 類語に弱い問題を緩和。`
- `[RAG] スコアしきい値（`RAG_MIN_SCORE`）を導入 -> 低関連チャンクの混入を制御可能に。`
- `[ログ] 検索ログを追加 -> 質問・類似度・回答を `Log/app.log` に保存。`
- `[データ] `scripts/rag_build_jsonl.py` を見出し分割 + 700文字再分割へ更新 -> `chunks.jsonl` を再生成（12件）。`
- `[データ] 分割の誤判定を修正 -> コードブロック内 `#` の見出し誤判定を除去、見出しのみチャンクを除外。`
- `[UI] チャット画面を再構成 -> チャット領域最大化、入力欄下固定、レイアウト崩れ修正。`

## 7. 作業ログ（2026-02-27）

- `[生成] 回答モードを追加 -> strict/hybridをAPIとフロント設定に反映。`
- `[RAG] Phase1を追加 -> ヒットチャンクの前後チャンク補完を実装。`
- `[RAG] Phase2を追加 -> 手順系/文脈不足時のみ元ファイル再読を実装。`
- `[ログ] similaritiesにcontext_typeを追加 -> chunk/neighbor/rereadを識別可能化。`

## 9. 作業ログ（2026-03-23）

- `[レビュー] data/chunk の front matter をレビュー -> タグ不足とメタ不整合を確認。`
- `[データ] 各チャンク文書へ検索補助タグを追記 -> 画面名・テーブル名・固有語を追加。`
- `[生成] chunks.jsonl を再生成 -> タグ更新を検索対象へ反映。`
- `[検証] LicenseUpdate.md の取得挙動を確認 -> tfidf は自然文質問で閾値未満になるケースを確認。`
- `[課題整理] 今後の開発課題を README / rag_support_roadmap.md / project_status.md に反映。`

## 10. 作業ログ（2026-04-02）

### Step 1: メタデータ品質
- `[監査] 全 21 ソース・244 チャンクの front matter を品質チェック -> 11 件の不整合を検出。`
- `[修正] P1（source_path 不一致 5件）を修正 -> コピーファイル名と front matter の齟齬を解消。`
- `[修正] P2（tags 空配列・title 重複・topic/product 形式 6件）を修正。`
- `[ツール] scripts/check_quality.py を新規作成 -> メタデータ品質の自動チェック。`
- `[検証] 修正後の再チェックで不整合 0 件を確認。`
- `[生成] chunks.jsonl を再生成 -> 244 チャンク。`

### Step 2: 評価セット拡充
- `[評価] eval_tuning.csv を 30→80 問に拡充 -> E031–E060（直接質問30問）、E061–E080（パラフレーズ20問）。`
- `[修正] E021, E022, E023, E029 の source_path バグを修正。`
- `[ツール] scripts/check_source_match.py を新規作成 -> 評価↔チャンク間の source_path 整合チェック。`
- `[検証] ソースカバレッジ 21/21 を確認。`
- `[計測] ベースライン: Hybrid(alpha=0.8) P@1=0.863, MRR=0.909（80問）。`

### Step 3: Embedding 検索検証
- `[モデル] ollama pull nomic-embed-text -> 768 次元埋め込みモデルを導入。`
- `[ツール] scripts/build_embeddings.py を新規作成 -> 全 244 チャンクの埋め込みベクトルを生成。`
- `[データ] data/embeddings.json を生成 -> 244 × 768 次元。`
- `[評価] eval_retrieval.py に embedding / hybrid3 モードを追加。`
- `[計測] embedding 単体: P@1=0.350（日本語に弱い）。`
- `[計測] hybrid3(alpha=0.05): P@1=0.863（hybrid と同等、改善なし）。`
- `[分析] hybrid と hybrid3 の失敗問を比較 -> 11 問が完全一致、差分ゼロ。`
- `[判断] nomic-embed-text の app.py 統合は見送り。日本語対応モデルでの再検証を次候補とする。`
