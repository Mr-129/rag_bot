"""失敗問の原因分析: source_path不一致チェック

使い方:
  python scripts/check_source_match.py

chunks.jsonl の source_path と eval_tuning.csv の expected_source_path を照合し、
不一致を検出する。
"""
import csv
import json
import sys
from pathlib import Path

chunks_path = Path("data/chunks.jsonl")
eval_path = Path("data/eval_tuning.csv")

for p in (chunks_path, eval_path):
    if not p.exists():
        print(f"エラー: {p} が見つかりません。")
        sys.exit(1)

# chunks.jsonl の source_path 一覧
with chunks_path.open("r", encoding="utf-8") as f:
    chunks = [json.loads(l) for l in f if l.strip()]
chunk_sources = sorted(set(c.get('source_path', '') for c in chunks))

# eval の expected_source_path 一覧
with eval_path.open("r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
eval_sources = sorted(set(r['expected_source_path'] for r in rows))

# 不一致の検出
print("=== eval expected_source_path NOT in chunks ===")
missing = set()
for es in eval_sources:
    normalized = es.strip().replace('\\', '/')
    if normalized not in chunk_sources:
        missing.add(normalized)
        affected = [r['id'] for r in rows if r['expected_source_path'].strip().replace('\\', '/') == normalized]
        print(f"  {normalized}")
        print(f"    affected: {affected}")

print(f"\n=== chunk source_paths ({len(chunk_sources)}) ===")
for cs in chunk_sources:
    print(f"  {cs}")

print(f"\n=== Summary ===")
print(f"  Eval unique sources: {len(eval_sources)}")
print(f"  Chunk unique sources: {len(chunk_sources)}")
print(f"  Missing in chunks: {len(missing)}")
