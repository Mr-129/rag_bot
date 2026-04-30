"""chunks.jsonl の各チャンクに対して Ollama Embedding を計算し、embeddings.json に保存する。

Usage:
  python scripts/build_embeddings.py
  python scripts/build_embeddings.py --chunks data/chunks.jsonl --out data/embeddings.json --model nomic-embed-text
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def build_search_text(rec: dict) -> str:
    """チャンクからEmbedding対象のテキストを作成する（app.pyの_build_search_textと同等）。"""
    tags = rec.get("tags", [])
    if isinstance(tags, list):
        tags_text = " ".join(str(t) for t in tags)
    else:
        tags_text = str(tags) if tags else ""
    # app.py 側の検索テキストと同じ並びを使う。
    # ここがずれると、事前計算した embedding と実行時の検索文脈が噛み合わなくなる。
    return "\n".join(
        [
            f"title: {rec.get('title', '')}",
            f"topic: {rec.get('topic', '')}",
            f"product: {rec.get('product', '')}",
            f"tags: {tags_text}",
            str(rec.get("text", "")),
        ]
    )


def get_embedding(text: str, model: str, ollama_url: str) -> list[float]:
    """Ollama /api/embed エンドポイントでEmbeddingベクトルを取得する。"""
    # このスクリプトは 1 チャンクずつ API を呼ぶシンプルな作りにしている。
    # 速度最適化より、失敗箇所が追いやすいことを優先したオフライン前処理ツール。
    resp = httpx.post(
        f"{ollama_url}/api/embed",
        json={"model": model, "input": text},
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()
    # Ollama /api/embed は {"embeddings": [[float, ...]]} を返す
    embeddings = data.get("embeddings")
    if embeddings and len(embeddings) > 0:
        return embeddings[0]
    raise ValueError(f"Unexpected response format: {data}")


def main() -> int:
    """CLIエントリーポイント。

    処理の流れ:
    1) chunks.jsonl を読み込む
    2) 各チャンクを embedding API へ送る
    3) chunk_id と embedding の対応表を JSON で保存する
    """
    parser = argparse.ArgumentParser(description="Build embedding vectors for chunks")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--out", default="data/embeddings.json")
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found")
        return 1

    records: list[dict] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # JSONL は「1行 = 1 JSON」なので、空行を除いてそのまま積む。
                records.append(json.loads(line))

    print(f"Loaded {len(records)} chunks from {chunks_path}")
    print(f"Model: {args.model}")
    print(f"Ollama: {args.ollama_url}")

    embeddings: list[dict] = []
    start = time.time()

    for i, rec in enumerate(records):
        chunk_id = rec.get("chunk_id", str(i))
        text = build_search_text(rec)

        try:
            vec = get_embedding(text, args.model, args.ollama_url)
        except Exception as e:
            # 一部失敗しても全体を止めず、最後まで作れるところだけ作る。
            # どの chunk_id が失敗したかを後で追えるようログへ出す。
            print(f"  [{i+1}/{len(records)}] FAILED chunk_id={chunk_id}: {e}")
            continue

        embeddings.append({
            "chunk_id": chunk_id,
            "embedding": vec,
        })

        # 長い前処理でも進捗が見えるよう、10件ごとに経過を表示する。
        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(records)}] {elapsed:.1f}s elapsed, dim={len(vec)}")

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        # 出力は JSONL ではなく 1 個の JSON。
        # モデル名や件数も持たせて、読み込む側が整合性を確認しやすくする。
        json.dump({"model": args.model, "count": len(embeddings), "embeddings": embeddings}, f, ensure_ascii=False)

    elapsed = time.time() - start
    print(f"\nDone: {len(embeddings)} embeddings -> {out_path} ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
