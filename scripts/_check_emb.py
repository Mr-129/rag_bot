"""埋め込みベクトルファイルのメタ情報を表示するユーティリティ

使い方:
  python scripts/_check_emb.py [ファイルパス]
  例: python scripts/_check_emb.py data/embeddings_bge_m3.json

デフォルト: data/embeddings_bge_m3.json（現行の bge-m3 ベクトル）
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/embeddings_bge_m3.json")

if not path.exists():
    print(f"エラー: {path} が見つかりません。")
    sys.exit(1)

try:
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    embs = d.get("embeddings", [])
    print(f"File:  {path}")
    print(f"Model: {d.get('model', '(不明)')}")
    print(f"Count: {d.get('count', len(embs))}")
    if embs:
        print(f"Dim:   {len(embs[0]['embedding'])}")
    else:
        print("Dim:   (埋め込みデータなし)")
except (json.JSONDecodeError, KeyError) as e:
    print(f"エラー: ファイルの読み込みに失敗しました: {e}")
    sys.exit(1)
