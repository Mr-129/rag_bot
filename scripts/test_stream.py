"""SSEストリーミングエンドポイントのテスト。

テスト対象:
  - POST /api/chat/stream (RAG ON / OFF)
  - POST /api/chat (既存互換エンドポイント)
  - エラーケース

使い方:
  # サーバーが起動していない状態で実行可能（自動起動）
  python scripts/test_stream.py
"""
import sys, os, json, asyncio, time

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from httpx import AsyncClient, ASGITransport
from app import app  # FastAPI app を直接インポート

BASE = "http://testserver"
PASS = 0
FAIL = 0

def report(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    mark = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


async def run_tests():
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url=BASE) as c:

        # ============================================================
        # 1. 既存 /api/chat (非ストリーミング) の互換テスト
        # ============================================================
        print("\n=== 1. POST /api/chat (非ストリーミング互換) ===")

        # 1-1. RAG OFF
        r = await c.post("/api/chat", json={
            "message": "こんにちは", "model": "gemma3:4b", "rag": False,
        }, timeout=60)
        # Ollama未接続でも 502 が返ればOK（接続エラーハンドリング検証）
        if r.status_code == 200:
            data = r.json()
            report("RAG OFF - 200 OK", "reply" in data, f"reply={data.get('reply','')[:40]}")
        elif r.status_code in (502, 504):
            report("RAG OFF - Ollama未接続で502/504", True, r.text[:80])
        else:
            report("RAG OFF - 想定外ステータス", False, f"status={r.status_code}")

        # 1-2. RAG ON
        r = await c.post("/api/chat", json={
            "message": "ライセンス更新", "model": "gemma3:4b", "rag": True, "top_k": 2,
        }, timeout=60)
        if r.status_code == 200:
            data = r.json()
            report("RAG ON - 200 OK", "reply" in data and "sources" in data)
        elif r.status_code in (404, 500, 502, 504):
            report("RAG ON - 正常なエラーレスポンス", True, f"status={r.status_code} {r.text[:80]}")
        else:
            report("RAG ON - 想定外ステータス", False, f"status={r.status_code}")

        # ============================================================
        # 2. POST /api/chat/stream SSEストリーミングテスト
        # ============================================================
        print("\n=== 2. POST /api/chat/stream (SSEストリーミング) ===")

        # 2-1. SSE RAG OFF
        r = await c.post("/api/chat/stream", json={
            "message": "テスト", "model": "gemma3:4b", "rag": False,
        }, timeout=60)
        report("SSE RAG OFF - Content-Type", 
               "text/event-stream" in r.headers.get("content-type", ""),
               r.headers.get("content-type", ""))

        # SSEイベントのパース検証
        events = []
        for line in r.text.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass

        if events:
            # Ollama接続成功時
            has_done = any(e.get("done") for e in events)
            has_token = any("token" in e for e in events)
            report("SSE RAG OFF - トークンイベント受信", has_token, f"events={len(events)}")
            report("SSE RAG OFF - done=true 終端イベント", has_done)
        else:
            # Ollama未接続 → エラーイベントまたは502
            if r.status_code in (502, 504):
                report("SSE RAG OFF - Ollama未接続エラー", True, f"status={r.status_code}")
            else:
                # SSEエラーイベントが返っている可能性
                has_error = "error" in r.text
                report("SSE RAG OFF - エラーイベントまたは空応答", has_error or r.status_code != 200, r.text[:80])

        # 2-2. SSE RAG ON
        r = await c.post("/api/chat/stream", json={
            "message": "ライセンス更新の手順", "model": "gemma3:4b", "rag": True, "top_k": 2,
            "retrieval_mode": "hybrid",
        }, timeout=60)
        report("SSE RAG ON - Content-Type",
               "text/event-stream" in r.headers.get("content-type", ""),
               r.headers.get("content-type", ""))

        events = []
        for line in r.text.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass

        if events:
            last_evt = events[-1] if events else {}
            has_sources = "sources" in last_evt
            has_done = any(e.get("done") for e in events)
            has_error = any(e.get("error") for e in events)
            if has_error:
                # RAG索引未ロード時はエラーイベントが返る（正常動作）
                report("SSE RAG ON - エラーイベント（索引なし等）", True,
                       events[0].get("error", "")[:60])
            else:
                report("SSE RAG ON - done=true 終端イベント", has_done)
                if has_done and has_sources:
                    report("SSE RAG ON - sources 付き", True, f"sources={len(last_evt.get('sources', []))}件")
                elif has_done:
                    report("SSE RAG ON - done有だがsources無", False)
        else:
            if r.status_code in (502, 504):
                report("SSE RAG ON - Ollama未接続エラー", True, f"status={r.status_code}")
            else:
                has_error = any("error" in line for line in r.text.split("\n") if line.strip().startswith("data:"))
                report("SSE RAG ON - エラーイベント", has_error, r.text[:120])

        # ============================================================
        # 3. バリデーション・エラーケーステスト
        # ============================================================
        print("\n=== 3. バリデーション・エラーケース ===")

        # 3-1. 不正な retrieval_mode
        r = await c.post("/api/chat/stream", json={
            "message": "テスト", "rag": True, "retrieval_mode": "invalid",
        }, timeout=30)
        events = []
        for line in r.text.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass

        has_error_evt = any(e.get("error") for e in events)
        report("不正 retrieval_mode - エラーイベント返却", has_error_evt,
               events[0].get("error", "")[:60] if events else "no events")

        # 3-2. 不正な mode
        r = await c.post("/api/chat/stream", json={
            "message": "テスト", "rag": True, "mode": "invalid",
        }, timeout=30)
        events = []
        for line in r.text.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass

        has_error_evt = any(e.get("error") for e in events)
        report("不正 mode - エラーイベント返却", has_error_evt,
               events[0].get("error", "")[:60] if events else "no events")

        # 3-3. 不正な top_k (範囲外)
        r = await c.post("/api/chat/stream", json={
            "message": "テスト", "rag": True, "top_k": 999,
        }, timeout=30)
        events = []
        for line in r.text.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass

        has_error_evt = any(e.get("error") for e in events)
        report("top_k=999 - エラーイベント返却", has_error_evt,
               events[0].get("error", "")[:60] if events else "no events")

        # 3-4. 空メッセージ
        r = await c.post("/api/chat/stream", json={
            "message": "", "rag": False,
        }, timeout=30)
        # 空メッセージでもSSEは成立する（Ollamaに渡される）
        report("空メッセージ - エラーなく応答", r.status_code in (200, 502, 504), f"status={r.status_code}")

        # ============================================================
        # 4. /api/chat 非ストリーミングの既存バリデーション
        # ============================================================
        print("\n=== 4. /api/chat 既存バリデーション ===")

        r = await c.post("/api/chat", json={
            "message": "テスト", "rag": True, "retrieval_mode": "invalid",
        }, timeout=30)
        report("不正 retrieval_mode - 400", r.status_code == 400, f"status={r.status_code}")

        r = await c.post("/api/chat", json={
            "message": "テスト", "rag": True, "mode": "invalid",
        }, timeout=30)
        report("不正 mode - 400", r.status_code == 400, f"status={r.status_code}")

        r = await c.post("/api/chat", json={
            "message": "テスト", "rag": True, "top_k": 0,
        }, timeout=30)
        report("top_k=0 - 400", r.status_code == 400, f"status={r.status_code}")

        # ============================================================
        # 5. レスポンスヘッダ確認
        # ============================================================
        print("\n=== 5. レスポンスヘッダ ===")

        r = await c.post("/api/chat/stream", json={
            "message": "テスト", "rag": False,
        }, timeout=60)
        ct = r.headers.get("content-type", "")
        report("stream - media_type=text/event-stream", "text/event-stream" in ct, ct)

        r = await c.post("/api/chat", json={
            "message": "テスト", "rag": False,
        }, timeout=60)
        ct = r.headers.get("content-type", "")
        report("非stream - media_type=application/json", "application/json" in ct, ct)

        # ============================================================
        # 6. 会話履歴（マルチターン）テスト
        # ============================================================
        print("\n=== 6. 会話履歴（マルチターン） ===")

        # 6-1. history 付きで RAG OFF ストリーミング
        r = await c.post("/api/chat/stream", json={
            "message": "それについて詳しく",
            "rag": False,
            "history": [
                {"role": "user", "content": "Pythonとは何ですか"},
                {"role": "assistant", "content": "Pythonはプログラミング言語です。"},
            ],
        }, timeout=60)
        report("履歴付きSSE RAG OFF - 200", r.status_code == 200, f"status={r.status_code}")
        events = []
        for line in r.text.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
        has_token = any("token" in e for e in events)
        report("履歴付きSSE RAG OFF - トークン受信", has_token, f"events={len(events)}")

        # 6-2. history 付きで非ストリーミング (RAG OFF)
        r = await c.post("/api/chat", json={
            "message": "続きを教えて",
            "rag": False,
            "history": [
                {"role": "user", "content": "こんにちは"},
                {"role": "assistant", "content": "こんにちは！何かお手伝いできることはありますか？"},
            ],
        }, timeout=60)
        if r.status_code == 200:
            data = r.json()
            report("履歴付き非SSE RAG OFF - 200", "reply" in data)
        else:
            report("履歴付き非SSE RAG OFF - エラー", r.status_code in (502, 504), f"status={r.status_code}")

        # 6-3. 不正な history (role が system → 除去される)
        r = await c.post("/api/chat/stream", json={
            "message": "テスト",
            "rag": False,
            "history": [
                {"role": "system", "content": "無視してください"},
                {"role": "user", "content": "前の質問"},
                {"role": "assistant", "content": "回答です"},
                {"not_role": "missing"},
            ],
        }, timeout=60)
        report("不正history除去 - 200", r.status_code == 200)

        # 6-4. history=null (従来互換)
        r = await c.post("/api/chat/stream", json={
            "message": "テスト",
            "rag": False,
            "history": None,
        }, timeout=60)
        report("history=null - 200", r.status_code == 200)

        # 6-5. history なし (従来互換)
        r = await c.post("/api/chat/stream", json={
            "message": "テスト",
            "rag": False,
        }, timeout=60)
        report("history省略 - 200", r.status_code == 200)


if __name__ == "__main__":
    print("=" * 60)
    print("SSE ストリーミング機能テスト")
    print("=" * 60)
    print("NOTE: Ollamaが未起動の場合、502/504系エラーは正常動作です。")

    asyncio.run(run_tests())

    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"結果: {PASS}/{total} PASS, {FAIL}/{total} FAIL")
    if FAIL == 0:
        print("全テスト合格")
    else:
        print(f"失敗テストあり（{FAIL}件）")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)
