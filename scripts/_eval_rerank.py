"""API経由でRerank付きRAG検索の精度を評価する"""
import csv
import sys
import httpx

BASE = "http://127.0.0.1:8772/api/chat"
EVAL_CSV = "data/eval_tuning.csv"

rows = list(csv.DictReader(open(EVAL_CSV, encoding="utf-8")))
print(f"Eval questions: {len(rows)}")
print(f"Columns: {list(rows[0].keys())}")

hit_at_1 = 0
rr_sum = 0.0
total = 0
errors = []

for row in rows:
    qid = row.get("id", "")
    question = row.get("question", "")
    expected_raw = (
        row.get("expected_source_path")
        or row.get("correct_source_path")
        or row.get("expected_source_paths")
        or row.get("correct_source_paths")
        or ""
    )
    expected_set = {
        p.strip().replace("\\", "/")
        for p in expected_raw.split("|")
        if p.strip()
    }
    if not expected_set or not question:
        continue

    total += 1
    try:
        r = httpx.post(
            BASE,
            json={
                "message": question,
                "rag": True,
                "mode": "strict",
                "top_k": 5,
            },
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        errors.append(f"  Q{qid}: {e}")
        continue

    sources = data.get("sources", [])
    ranked_paths = [
        s.get("source_path", "").replace("\\", "/")
        for s in sources
    ]

    # P@1
    if ranked_paths and ranked_paths[0] in expected_set:
        hit_at_1 += 1

    # MRR
    found_rr = 0.0
    for i, p in enumerate(ranked_paths, 1):
        if p in expected_set:
            found_rr = 1.0 / i
            break
    rr_sum += found_rr

    status = "HIT" if (ranked_paths and ranked_paths[0] in expected_set) else "MISS"
    if status == "MISS":
        print(f"  Q{qid} MISS: expected={expected_raw} got={ranked_paths[:3]}")

if total == 0:
    print("No eval questions found")
    sys.exit(1)

p1 = hit_at_1 / total
mrr = rr_sum / total
print(f"\n=== Rerank Evaluation (top_k=5, rerank from initial_k=10) ===")
print(f"Total questions: {total}")
print(f"P@1  = {p1:.4f} ({hit_at_1}/{total})")
print(f"MRR  = {mrr:.4f}")
if errors:
    print(f"Errors: {len(errors)}")
    for e in errors:
        print(e)
