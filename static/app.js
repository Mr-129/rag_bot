// 画面上の主要要素を取得（このID名は index.html と対応）
const chatEl = document.getElementById('chat');
const msgEl = document.getElementById('message');
const modelEl = document.getElementById('model');
const ragEl = document.getElementById('rag');
const topkEl = document.getElementById('topk');
const modeEl = document.getElementById('mode');
const retrievalModeEl = document.getElementById('retrievalMode');
const sendBtn = document.getElementById('send');
const clearBtn = document.getElementById('clear');
const statusEl = document.getElementById('status');

// ---------------------------------------------------------------------------
// 会話履歴管理
// ---------------------------------------------------------------------------
// フロントエンド側で会話履歴を配列として保持し、毎回のリクエストに含める。
// サーバ側はステートレスで、セッション管理不要。
// 「履歴をクリア」ボタンで画面と履歴を同時リセットする。
let conversationHistory = [];

// LLM設定をサーバーから取得してデフォルトモデル名を設定
fetch('/api/llm-config')
  .then(r => r.json())
  .then(cfg => { if (cfg.model && modelEl) modelEl.value = cfg.model; })
  .catch(() => {});

// 文字列をHTMLエスケープ（script混入などを防ぐため）
function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

// Markdownを安全なHTMLへ変換する
function renderMarkdownToSafeHtml(markdownText) {
  const md = String(markdownText ?? '');

  // marked/DOMPurify がない環境（CDNブロック等）でも壊れないようにフォールバック
  if (!window.marked) return escapeHtml(md).replaceAll('\n', '<br>');

  const rawHtml = window.marked.parse(md, {
    gfm: true,
    breaks: true,
    mangle: false,
    headerIds: false,
  });

  // DOMPurify が無い状態で生HTMLを描画すると、LLM応答由来のHTMLをそのまま通すことになる。
  // 表示品質より安全性を優先し、サニタイズ不能時はプレーン表示へ戻す。
  if (!window.DOMPurify) return escapeHtml(md).replaceAll('\n', '<br>');
  return window.DOMPurify.sanitize(rawHtml);
}

// チャット欄に1メッセージ追加する
function appendMessage(role, content, kind = role) {
  const row = document.createElement('div');
  row.className = `msg ${kind}`;

  const roleEl = document.createElement('div');
  roleEl.className = 'role';
  roleEl.textContent = role;

  const contentEl = document.createElement('div');
  contentEl.className = 'content';
  if (kind === 'assistant') {
    contentEl.innerHTML = renderMarkdownToSafeHtml(content);
  } else {
    contentEl.textContent = content;
  }

  row.appendChild(roleEl);
  row.appendChild(contentEl);
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
}

// 入力欄の高さを内容に合わせて自動調整する
function autoResizeMessageBox() {
  msgEl.style.height = 'auto';
  const next = Math.min(msgEl.scrollHeight, 220);
  msgEl.style.height = `${Math.max(next, 56)}px`;
}

// 送信中のUI状態（入力不可/ステータス表示）をまとめて制御する
function setBusy(busy, text = '') {
  sendBtn.disabled = busy;
  msgEl.disabled = busy;
  modelEl.disabled = busy;
  if (ragEl) ragEl.disabled = busy;
  if (topkEl) topkEl.disabled = busy;
  if (modeEl) modeEl.disabled = busy;
  if (retrievalModeEl) retrievalModeEl.disabled = busy;
  statusEl.textContent = text;
}

// APIから返るsources配列をMarkdownの箇条書きに整形する
function sourcesToMarkdown(sources) {
  if (!Array.isArray(sources) || sources.length === 0) return '';
  const lines = ['\n---\n', '### 参照した文書'];
  for (const s of sources) {
    const rank = s.rank ?? '';
    const title = String(s.title ?? '').trim() || '(untitled)';
    const score = typeof s.score === 'number' ? s.score.toFixed(4) : '';
    const sourcePath = String(s.source_path ?? '').trim();
    const label = rank !== '' ? `Doc ${rank}: ${title}` : title;
    const details = [];
    if (sourcePath) details.push(`出典: ${sourcePath}`);
    if (score) details.push(`関連度: ${score}`);
    const detailText = details.length > 0 ? ` / ${details.join(' / ')}` : '';
    lines.push(`- ${label}${detailText}`);
  }
  return lines.join('\n');
}

// 送信ボタン/Enterで呼ばれるメイン処理（SSEストリーミング対応）
//
// 処理の流れ:
//   1) 設定パネルからパラメータを収集
//   2) ユーザー発言を画面に即座反映
//   3) 空のアシスタントバブルを先に用意
//   4) POST /api/chat/stream へ送信し、SSE でトークンを逐次受信
//   5) 受信のたびに Markdown → HTML をリアルタイム描画
//   6) 最終イベント (done=true) に付随する sources を末尾に追記
//
// SSE 形式: 各行 "data: {JSON}\n\n"
//   - 通常: {"token":"...","done":false}
//   - 完了: {"token":"","done":true,"sources":[...]}
//   - エラー: {"error":"..."}
async function send() {
  // --- 1) パラメータ収集（設定パネルの各 select/checkbox から取得） ---
  const message = msgEl.value.trim();
  const model = modelEl.value.trim() || 'gemma3:4b';
  const rag = ragEl ? !!ragEl.checked : true;
  const top_k = topkEl ? Number(topkEl.value || 2) : 2;
  const mode = modeEl ? (modeEl.value || 'strict') : 'strict';
  const retrieval_mode = retrievalModeEl ? (retrievalModeEl.value || 'hybrid') : 'hybrid';
  if (!message) return;

  // --- 2) ユーザー発言を画面へ即座反映 ---
  appendMessage('user', message, 'user');
  msgEl.value = '';
  autoResizeMessageBox();
  setBusy(true, rag ? '検索中…' : '送信中…');

  // --- 3) アシスタント回答用の空バブルを先に用意（ストリーム受信で徐々に埋める） ---
  const row = document.createElement('div');
  row.className = 'msg assistant';
  const roleEl2 = document.createElement('div');
  roleEl2.className = 'role';
  roleEl2.textContent = 'assistant';
  const contentEl2 = document.createElement('div');
  contentEl2.className = 'content';
  row.appendChild(roleEl2);
  row.appendChild(contentEl2);
  chatEl.appendChild(row);

  let fullReply = '';   // 全トークンの蓄積（Markdown再描画用）
  let sources = null;   // 最終イベントで受け取る根拠一覧

  try {
    // --- 4) SSE エンドポイントへ POST（会話履歴を含める） ---
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, model, rag, top_k, mode, retrieval_mode, history: conversationHistory }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }

    // 検索フェーズを経て LLM への問い合わせが始まったことを表示
    setBusy(true, rag ? '推論中…' : '生成中…');

    // ReadableStream をバイト列→テキストとして逐次読み取る
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';  // TCP チャンク境界で行が途切れるためバッファリングが必要

    // --- 5) トークン逐次受信ループ ---
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE は改行区切り: "data: {...}\n\n" → "\n" で分割してパース
      const lines = buffer.split('\n');
      // 最後の行は不完全な可能性があるのでバッファに残す
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        // SSE イベントは "data: " プレフィックス付き。それ以外はスキップ
        if (!trimmed.startsWith('data: ')) continue;
        // "data: " の 6 文字を除去して JSON をパース
        const jsonStr = trimmed.slice(6);
        let evt;
        try { evt = JSON.parse(jsonStr); } catch { continue; }

        // エラーイベント（バリデーション失敗、索引なし等）
        if (evt.error) {
          throw new Error(evt.error);
        }

        // トークン追記 → Markdown 再レンダリングで逐次的に画面が更新される
        if (evt.token) {
          fullReply += evt.token;
          contentEl2.innerHTML = renderMarkdownToSafeHtml(fullReply);
          chatEl.scrollTop = chatEl.scrollHeight;
        }

        // ストリーム完了: done=true の最終イベントに sources が付与されている
        if (evt.done && evt.sources) {
          sources = evt.sources;
        }
      }
    }

    // --- 6) 根拠一覧を末尾に追記 ---
    if (sources) {
      const md = fullReply + sourcesToMarkdown(sources);
      contentEl2.innerHTML = renderMarkdownToSafeHtml(md);
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    // --- 7) 会話履歴に今回のやり取りを追加 ---
    // 回答本体のみ（sources部分は除く）を履歴に保存する。
    // sources まで入れるとトークン量が膨張し、LLMコンテキストを圧迫する。
    conversationHistory.push({ role: 'user', content: message });
    if (fullReply) {
      conversationHistory.push({ role: 'assistant', content: fullReply });
    }

  } catch (err) {
    // エラー時: まだ回答が空ならバブル自体をエラー表示に差し替え
    if (!fullReply) {
      row.className = 'msg error';
      roleEl2.textContent = 'error';
      contentEl2.textContent = String(err);
    } else {
      appendMessage('error', String(err), 'error');
    }
  } finally {
    setBusy(false, '');
    msgEl.focus();
  }
}

// クリック送信
sendBtn.addEventListener('click', send);
clearBtn.addEventListener('click', () => {
  // 画面上の会話履歴と内部配列を同時にクリア
  chatEl.innerHTML = '';
  conversationHistory = [];
  appendMessage('assistant', '履歴をクリアしました。新しい会話を始められます。', 'assistant');
  msgEl.focus();
});

// Enter送信（Shift+Enterは改行）
msgEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

msgEl.addEventListener('input', autoResizeMessageBox);

// 初回メッセージ
appendMessage('assistant', '準備できました。メッセージを送ってください。', 'assistant');
autoResizeMessageBox();
msgEl.focus();
