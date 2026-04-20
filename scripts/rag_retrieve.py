from __future__ import annotations

"""Tiny RAG retrieval test (no LLM).

このスクリプトは「RAGの Retrieval（検索）部分だけ」を最小構成で試すためのものです。

やっていること（ざっくり）
1) data-dir 配下のチャンク（.md / .txt）を読み込む
2) Markdownの YAML front matter（--- ... ---）があればメタデータとして読む
3) 各チャンクを TF-IDF でベクトル化する（日本語向けに文字 n-gram を使用）
4) クエリも同じ方法でベクトル化する
5) コサイン類似度で「最も近いチャンク」を上位K件返す

注意
- これは埋め込み(embedding)モデルを使うベクトル検索ではなく、古典的な TF-IDF 検索です。
- LLMは呼び出しません（APIキー不要）。まず検索が効くかの動作確認用です。
- このスクリプトは「簡易検証用」で、1ファイル=1チャンク扱いです。
    本番用の分割ロジック（見出し分割など）は rag_build_jsonl.py 側を参照してください。
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class Chunk:
    """1ファイル = 1チャンクとして扱うための入れ物。

    meta: YAML front matter から得たメタデータ（title/tags/source_path 等）
    text: 本文（front matter を除外した残り）
    """

    path: Path
    meta: dict[str, Any]
    text: str

    @property
    def title(self) -> str:
        return str(self.meta.get("title") or self.path.stem)

    @property
    def source_path(self) -> str:
        return str(self.meta.get("source_path") or self.path.as_posix())


def parse_markdown_with_front_matter(content: str) -> tuple[dict[str, Any], str]:
    """Markdown先頭の YAML front matter を (meta, body) に分離する。

    front matter 形式例:
    ---
    title: "製品X ライセンス更新"
    tags: ["ライセンス", "更新"]
    ---
    # 本文...
    """

    match = FRONT_MATTER_RE.match(content.strip() + "\n")
    if not match:
        return {}, content

    raw_yaml, body = match.group(1), match.group(2)
    meta = yaml.safe_load(raw_yaml) or {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, body


def load_chunks(data_dir: Path) -> list[Chunk]:
    """data_dir配下の .md/.txt を読み、Chunkリストとして返す。"""

    paths = sorted([*data_dir.glob("*.md"), *data_dir.glob("*.txt")])
    chunks: list[Chunk] = []

    for path in paths:
        content = path.read_text(encoding="utf-8", errors="ignore")
        # Markdownはfront matterを剥がしてメタにする。txtはメタ無しとして読む。
        meta, body = parse_markdown_with_front_matter(content) if path.suffix.lower() == ".md" else ({}, content)
        chunks.append(Chunk(path=path, meta=meta, text=body.strip()))

    return chunks


def build_search_corpus(chunks: list[Chunk]) -> list[str]:
    """検索用のテキスト（コーパス）を作る。

    本文だけだと短いクエリでヒットしづらいことがあるので、
    title/product/topic/tags などメタデータも一緒に混ぜた文字列を作っている。
    """

    corpus: list[str] = []
    for c in chunks:
        tags = c.meta.get("tags")
        if isinstance(tags, list):
            tags_text = " ".join(str(x) for x in tags)
        else:
            tags_text = str(tags) if tags else ""

        topic = str(c.meta.get("topic") or "")
        product = str(c.meta.get("product") or "")
        title = c.title

        # メタも一緒に混ぜておくと、短文クエリでも当たりやすい
        blended = "\n".join(
            x
            for x in [
                f"title: {title}",
                f"product: {product}",
                f"topic: {topic}",
                f"tags: {tags_text}",
                c.text,
            ]
            if x.strip()
        )
        corpus.append(blended)
    return corpus


def search(chunks: list[Chunk], query: str, top_k: int) -> list[tuple[Chunk, float]]:
    """chunksからqueryに近い順に上位top_k件を返す。

    - ベクトル化: TF-IDF
    - 特徴量: 文字2〜4gram（日本語の単語分割をしない前提のため）
    - 類似度: コサイン類似度（0〜1に近いほど類似）
    """

    corpus = build_search_corpus(chunks)

    # 日本語は形態素なしだと単語分割が難しいので、文字n-gramで雑に強くする。
    # 例: "ライセンス更新" -> ["ライセ", "イセンス", ...] のような文字片で一致を拾える。
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))

    # matrix: (文書数 x 特徴量数) の疎行列
    matrix = vectorizer.fit_transform(corpus)
    # q_vec: (1 x 特徴量数)
    q_vec = vectorizer.transform([query])

    # q_vec と各チャンクの「角度の近さ」をコサイン類似度で計算
    scores = cosine_similarity(q_vec, matrix)[0]
    ranked = sorted(zip(chunks, scores, strict=True), key=lambda x: x[1], reverse=True)
    return [(c, float(s)) for c, s in ranked[: max(top_k, 1)]]


def format_preview(text: str, max_chars: int = 180) -> str:
    """表示用に本文を1行に潰し、長ければ省略する。"""

    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def main() -> int:
    """CLIエントリーポイント。

    --query を指定すると1回だけ検索して結果表示。
    --query 無しだと対話モード（複数クエリを試せる）。
    """

    parser = argparse.ArgumentParser(description="Tiny RAG retrieval test (no LLM).")
    parser.add_argument("--data-dir", default="data/chunk", help="Directory containing chunk files")
    parser.add_argument("--query", help="Query text")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"data-dir not found: {data_dir}")

    chunks = load_chunks(data_dir)
    if not chunks:
        raise SystemExit(f"No chunks found under: {data_dir}")

    def run_once(q: str) -> None:
        # 検索 → 上位結果を人間が目視できる形で出す
        results = search(chunks, q, args.top_k)
        print(f"Query: {q}")
        print(f"Chunks: {len(chunks)}  TopK: {args.top_k}")
        print("-")
        for i, (c, score) in enumerate(results, start=1):
            print(f"[{i}] score={score:.4f}  title={c.title}")
            print(f"    source_path={c.source_path}")
            print(f"    file={c.path.as_posix()}")
            print(f"    preview={format_preview(c.text)}")
            print("-")

    if args.query:
        run_once(args.query)
        return 0

    print("Interactive mode. Empty input to exit.")
    while True:
        q = input("> ").strip()
        if not q:
            break
        run_once(q)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
