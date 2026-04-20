"""データファイル整合性テスト（2026-04-06 作成）

RAG Bot のデータパイプラインが健全かを検証するスクリプト。
サーバを起動せず、ファイル単体でチェックできる。

検証内容:
  1. chunks.jsonl
     - JSON行の構文エラーがないか
     - 必須フィールド（chunk_id, title, source_path, text）が全レコードに存在するか
     - chunk_id の重複がないか
  2. embeddings_bge_m3.json
     - モデル名・次元数の確認
     - 全チャンクに対応する埋め込みベクトルが存在するか（逆方向のチェックも）
  3. eval_tuning.csv
     - 評価問の件数・カラム構成
     - expected_source_path が chunks.jsonl のソースに存在するか
  4. joblib インデックスの有無確認

使い方:
  python scripts/_test_integrity.py

戻り値:
  全チェック合格なら exit(0)、1件でもエラーがあれば exit(1)。
"""
import json
import csv
import sys
from pathlib import Path

# このスクリプトは scripts/ 配下にあるので、親ディレクトリがプロジェクトルート
BASE = Path(__file__).resolve().parent.parent

def main():
    errors = []

    # 1. chunks.jsonl
    print("=== chunks.jsonl check ===")
    chunks_path = BASE / "data" / "chunks.jsonl"
    records = []
    with chunks_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as e:
                errors.append(f"JSON error line {i}: {e}")
                print(f"  JSON error line {i}: {e}")
    print(f"  Total chunks: {len(records)}")

    required = ["chunk_id", "title", "source_path", "text"]
    for rec in records:
        for field in required:
            if not rec.get(field):
                cid = rec.get("chunk_id", "?")
                errors.append(f"Missing {field} in {cid}")
                print(f"  Missing {field} in {cid}")

    ids = [r["chunk_id"] for r in records]
    dups = [x for x in set(ids) if ids.count(x) > 1]
    if dups:
        errors.append(f"Duplicate chunk_ids: {dups}")
        print(f"  Duplicate chunk_ids: {dups}")
    else:
        print("  No duplicate chunk_ids")

    # 2. embeddings_bge_m3.json
    print()
    print("=== embeddings_bge_m3.json check ===")
    emb_path = BASE / "data" / "embeddings_bge_m3.json"
    with emb_path.open(encoding="utf-8") as f:
        emb_data = json.load(f)
    emb_list = emb_data.get("embeddings", [])
    print(f"  Model: {emb_data.get('model', '?')}")
    print(f"  Count: {len(emb_list)}")
    if emb_list:
        print(f"  Dim: {len(emb_list[0]['embedding'])}")

    emb_ids = {e["chunk_id"] for e in emb_list}
    chunk_ids = {r["chunk_id"] for r in records}
    missing_emb = chunk_ids - emb_ids
    extra_emb = emb_ids - chunk_ids
    if missing_emb:
        errors.append(f"Chunks missing embeddings: {len(missing_emb)}")
        print(f"  Chunks missing embeddings ({len(missing_emb)}): {list(missing_emb)[:5]}")
    else:
        print("  All chunks have embeddings")
    if extra_emb:
        print(f"  Extra embeddings (no chunk): {list(extra_emb)[:5]}")

    # 3. eval_tuning.csv
    print()
    print("=== eval_tuning.csv check ===")
    eval_path = BASE / "data" / "eval_tuning.csv"
    with eval_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        evals = list(reader)
    print(f"  Total eval questions: {len(evals)}")
    print(f"  Columns: {list(evals[0].keys()) if evals else 'empty'}")

    sources_in_chunks = {r.get("source_path", "") for r in records}
    mismatches = []
    for e in evals:
        esp = e.get("expected_source_path", "")
        if esp and esp not in sources_in_chunks:
            mismatches.append((e.get("id", "?"), esp))
    if mismatches:
        errors.append(f"Source path mismatches: {len(mismatches)}")
        print(f"  Source path mismatches ({len(mismatches)}):")
        for mid, mp in mismatches[:5]:
            print(f"    {mid}: {mp}")
    else:
        print("  All expected sources found in chunks")

    # 4. index joblib チェック
    print()
    print("=== Index joblib check ===")
    idx_dir = BASE / "data" / "ProductXManual_chunks_100"
    if idx_dir.exists():
        joblib_files = list(idx_dir.glob("*.joblib"))
        print(f"  joblib files: {[f.name for f in joblib_files]}")
    else:
        print("  Index directory not found (will build from chunks.jsonl)")

    print()
    if errors:
        print(f"ERRORS FOUND: {len(errors)}")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("=== All data integrity checks PASSED ===")

if __name__ == "__main__":
    main()
