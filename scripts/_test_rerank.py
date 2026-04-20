"""Rerank 統合テスト"""
import httpx
import sys

BASE = "http://127.0.0.1:8772/api/chat"
passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        ok, detail = fn()
        if ok:
            print(f"[PASS] {name}")
            passed += 1
        else:
            print(f"[FAIL] {name}: {detail}")
            failed += 1
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        failed += 1


# 1. strict+RAG with rerank
def t1():
    r = httpx.post(BASE, json={"message": "製品Xのバージョンアップ手順を教えて", "rag": True, "mode": "strict", "top_k": 2}, timeout=300)
    if r.status_code != 200:
        return False, f"status={r.status_code}"
    d = r.json()
    if not d.get("reply"):
        return False, "no reply"
    return True, ""

test("strict+RAG with rerank", t1)


# 2. sources returned
def t2():
    r = httpx.post(BASE, json={"message": "製品Xのバージョンアップ手順を教えて", "rag": True, "mode": "strict", "top_k": 2}, timeout=120)
    d = r.json()
    sources = d.get("sources", [])
    if len(sources) == 0:
        return False, "no sources"
    return True, ""

test("sources returned", t2)


# 3. reply mentions 製品X
def t3():
    r = httpx.post(BASE, json={"message": "製品Xのバージョンアップ手順を教えて", "rag": True, "mode": "strict", "top_k": 2}, timeout=120)
    d = r.json()
    reply = d.get("reply", "")
    if "製品X" not in reply and "製品x" not in reply.lower():
        return False, f"製品X not in reply: {reply[:100]}"
    return True, ""

test("reply mentions 製品X", t3)


# 4. general+RAG with rerank
def t4():
    r = httpx.post(BASE, json={"message": "製品Xのライセンスについて教えて", "rag": True, "mode": "general", "top_k": 2}, timeout=120)
    if r.status_code != 200:
        return False, f"status={r.status_code}"
    d = r.json()
    if d.get("mode") != "general":
        return False, f"mode={d.get('mode')}"
    return True, ""

test("general+RAG with rerank", t4)


# 5. invalid mode rejected
def t5():
    r = httpx.post(BASE, json={"message": "test", "rag": True, "mode": "invalid"}, timeout=30)
    return r.status_code == 400, f"status={r.status_code}"

test("invalid mode rejected", t5)


# 6. non-RAG still works
def t6():
    r = httpx.post(BASE, json={"message": "hello", "rag": False}, timeout=60)
    if r.status_code != 200:
        return False, f"status={r.status_code}"
    d = r.json()
    if not d.get("reply"):
        return False, "no reply"
    return True, ""

test("non-RAG still works", t6)


# 7. rerank narrows results (initial_k=10 -> top_k=2)
def t7():
    r = httpx.post(BASE, json={"message": "OracleDBのバックアップ方法", "rag": True, "mode": "strict", "top_k": 2}, timeout=120)
    d = r.json()
    sources = d.get("sources", [])
    if len(sources) < 1:
        return False, f"sources count={len(sources)}"
    return True, ""

test("rerank narrows results", t7)


# 8. top_k=5 returns multiple sources
def t8():
    r = httpx.post(BASE, json={"message": "製品Xの設定について", "rag": True, "mode": "strict", "top_k": 5}, timeout=300)
    d = r.json()
    sources = d.get("sources", [])
    if len(sources) < 3:
        return False, f"sources count={len(sources)}, expected >=3"
    return True, ""

test("top_k=5 returns multiple sources", t8)


# 9. hybrid mode with rerank
def t9():
    r = httpx.post(BASE, json={"message": "SQL Serverのインストール方法", "rag": True, "mode": "hybrid", "top_k": 2}, timeout=120)
    if r.status_code != 200:
        return False, f"status={r.status_code}"
    d = r.json()
    if d.get("mode") != "hybrid":
        return False, f"mode={d.get('mode')}"
    return True, ""

test("hybrid+RAG with rerank", t9)


print(f"\n=== {passed} passed, {failed} failed ===")
sys.exit(1 if failed > 0 else 0)
