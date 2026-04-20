"""P@1 失敗問の一覧表示 + モード別比較

使い方:
  python scripts/show_failures.py [モード名]
  例: python scripts/show_failures.py hybrid3

デフォルトモード: hybrid3
入力: eval_results.csv（scripts/eval_retrieval.py が出力）
"""
import csv
import sys
from pathlib import Path

target_mode = sys.argv[1] if len(sys.argv) > 1 else "hybrid3"

results_path = Path("eval_results.csv")
if not results_path.exists():
    print(f"エラー: {results_path} が見つかりません。")
    print("先に scripts/eval_retrieval.py を実行してください。")
    sys.exit(1)

with results_path.open("r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

mode_rows = [r for r in rows if r["mode"] == target_mode]
if not mode_rows:
    # alpha付きモードの場合（例: hybrid3 alpha=0.7）
    mode_rows = [r for r in rows if r["mode"] == target_mode and r.get("alpha")]

if not mode_rows:
    print(f"モード '{target_mode}' の結果が見つかりません。")
    available = sorted(set(r["mode"] for r in rows))
    print(f"利用可能なモード: {', '.join(available)}")
    sys.exit(1)

failed = [r for r in mode_rows if float(r["P@1"]) == 0.0]
print(f"{target_mode} P@1 failures: {len(failed)}/{len(mode_rows)}")
print()
for r in failed:
    print(f"  {r['id']}: {r['question'][:60]}")
