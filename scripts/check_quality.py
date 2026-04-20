"""チャンクメタデータ品質チェックスクリプト

使い方:
  python scripts/check_quality.py [chunks.jsonlパス]
  例: python scripts/check_quality.py data/chunks.jsonl

デフォルト: data/chunks.jsonl
"""
import json
import sys
from pathlib import Path

chunks_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/chunks.jsonl")
if not chunks_path.exists():
    print(f"エラー: {chunks_path} が見つかりません。")
    sys.exit(1)

with chunks_path.open("r", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f if l.strip()]

issues = []
for c in lines:
    sp = c.get("source_path", "")
    cid = c.get("chunk_id", "?")[:12]
    if not c.get("tags"):
        issues.append(f"  [tags empty]  {sp}  chunk={cid}")
    if not c.get("title"):
        issues.append(f"  [no title]    {sp}  chunk={cid}")
    if sp.endswith("/.md") or sp == "data/chank/.md":
        issues.append(f"  [bad path]    {sp}  chunk={cid}")

print(f"Total chunks: {len(lines)}")
print(f"Issues found: {len(issues)}")
for i in issues:
    print(i)

# source_path の一覧
paths = sorted(set(c.get("source_path", "") for c in lines))
print(f"\nUnique source_paths ({len(paths)}):")
for p in paths:
    print(f"  {p}")
