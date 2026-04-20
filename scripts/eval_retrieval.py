from __future__ import annotations

"""Evaluate retrieval quality for TF-IDF / BM25 / Hybrid / Embedding.

Usage:
  python scripts/eval_retrieval.py --chunks data/chunks.jsonl --eval data/eval_sample.csv
  python scripts/eval_retrieval.py --eval data/eval_tuning.csv --modes embedding,hybrid3 --embeddings data/embeddings.json

The eval CSV can use chunk IDs or source paths as the expected target.
Outputs per-question metrics and a summary CSV (`eval_results.csv`).
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_chunks_from_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            records.append(item)
    return records


def build_search_text(rec: dict) -> str:
    tags = rec.get("tags", [])
    if isinstance(tags, list):
        tags_text = " ".join(str(t) for t in tags)
    else:
        tags_text = str(tags) if tags else ""
    return "\n".join(
        [
            f"title: {rec.get('title', '')}",
            f"topic: {rec.get('topic', '')}",
            f"product: {rec.get('product', '')}",
            f"tags: {tags_text}",
            str(rec.get("text", "")),
        ]
    )


class SimpleBM25:
    def __init__(self, tokenized_corpus: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_count = len(tokenized_corpus)
        self.doc_lengths = [len(doc) for doc in tokenized_corpus]
        self.avg_doc_length = (sum(self.doc_lengths) / self.doc_count) if self.doc_count > 0 else 0.0
        self.idf: Dict[str, float] = {}
        self.postings: Dict[str, List[Tuple[int, int]]] = {}

        if self.doc_count == 0:
            return

        doc_freq: Dict[str, int] = {}
        for idx, doc in enumerate(tokenized_corpus):
            tf: Dict[str, int] = {}
            for token in doc:
                if not token:
                    continue
                tf[token] = tf.get(token, 0) + 1
            for token, freq in tf.items():
                self.postings.setdefault(token, []).append((idx, freq))
                doc_freq[token] = doc_freq.get(token, 0) + 1

        for token, df in doc_freq.items():
            self.idf[token] = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0 for _ in range(self.doc_count)]
        if self.doc_count == 0:
            return scores
        for token in set(query_tokens):
            idf = self.idf.get(token)
            if idf is None:
                continue
            for doc_idx, tf in self.postings.get(token, []):
                dl = self.doc_lengths[doc_idx]
                norm = self.k1 * (1.0 - self.b + self.b * (dl / self.avg_doc_length if self.avg_doc_length > 0 else 1.0))
                scores[doc_idx] += idf * ((tf * (self.k1 + 1.0)) / (tf + norm))
        return scores


def _tokenize_for_bm25(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", str(text).lower()).strip()
    tokens = re.findall(r"[0-9a-z_./-]+|[一-龥ぁ-んァ-ン]+", normalized)
    compact = normalized.replace(" ", "")
    for n in (2, 3):
        if len(compact) < n:
            continue
        tokens.extend(compact[i : i + n] for i in range(0, len(compact) - n + 1))
    return tokens


def minmax_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    mn = min(scores)
    mx = max(scores)
    if mx <= mn:
        return [0.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def precision_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    topk = ranked[:k]
    if not topk:
        return 0.0
    return sum(1 for x in topk if x in relevant) / len(topk)


def reciprocal_rank(ranked: List[str], relevant: Set[str]) -> float:
    for i, x in enumerate(ranked, start=1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def average_precision(ranked: List[str], relevant: Set[str]) -> float:
    if not relevant:
        return 0.0
    hit = 0
    sum_prec = 0.0
    for i, x in enumerate(ranked, start=1):
        if x in relevant:
            hit += 1
            sum_prec += hit / i
    return sum_prec / len(relevant)


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def _split_expected_values(raw: Any) -> Set[str]:
    return {_normalize_key(part) for part in str(raw or "").split("|") if _normalize_key(part)}


def _resolve_expected_targets(row: dict) -> Tuple[str, Set[str]]:
    chunk_targets = _split_expected_values(
        row.get("correct_chunk_ids") or row.get("correct_chunk_id") or row.get("expected_chunk_ids") or row.get("expected_chunk_id")
    )
    if chunk_targets:
        return "chunk_id", chunk_targets

    source_targets = _split_expected_values(
        row.get("expected_source_paths") or row.get("expected_source_path") or row.get("correct_source_paths") or row.get("correct_source_path")
    )
    if source_targets:
        return "source_path", source_targets

    return "chunk_id", set()


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _rank_unique_values(scores: List[float], values: List[str], top_k: int) -> List[str]:
    """スコア順を保ったまま、評価対象キーをユニーク化して返す。

    source_path 評価では同一文書の複数チャンクが上位を埋めやすい。
    先に top_k チャンクを切ってから重複除去すると、
    実際より少ない件数しか比較できず評価が歪むため、
    全体順位からユニーク化した後に top_k を適用する。
    """
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked_values = [values[i] for i in ranked_idx]
    return _dedupe_preserve_order(ranked_values)[:top_k]


def evaluate_all(
    records: List[dict],
    eval_rows: List[dict],
    top_k: int = 10,
    modes: List[str] = ("tfidf", "bm25", "hybrid"),
    alphas: List[float] = (0.8,),
    bm25_k1: float = 0.5,
    bm25_b: float = 0.3,
    embeddings_path: str = "",
    ollama_url: str = "http://localhost:11434",
    embed_model: str = "nomic-embed-text",
):
    docs = [build_search_text(r) for r in records]
    ids = [str(r.get("chunk_id") or r.get("title") or i) for i, r in enumerate(records)]
    source_paths = [_normalize_key(r.get("source_path") or r.get("source_uri") or "") for r in records]

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    matrix = vectorizer.fit_transform(docs)

    corpus_tokens = [_tokenize_for_bm25(d) for d in docs]
    bm25 = SimpleBM25(corpus_tokens, k1=bm25_k1, b=bm25_b)

    # Load pre-computed embeddings if available
    emb_matrix = None  # shape: (num_chunks, dim)
    emb_id_to_idx: Dict[str, int] = {}
    need_embedding = any(m in modes for m in ("embedding", "hybrid3"))
    if need_embedding and embeddings_path:
        emb_path = Path(embeddings_path)
        if emb_path.exists():
            with emb_path.open("r", encoding="utf-8") as f:
                emb_data = json.load(f)
            emb_list = emb_data.get("embeddings", [])
            # Build matrix aligned to records order by chunk_id
            chunk_id_to_emb: Dict[str, List[float]] = {}
            for item in emb_list:
                chunk_id_to_emb[item["chunk_id"]] = item["embedding"]
            dim = len(emb_list[0]["embedding"]) if emb_list else 768
            emb_vecs = []
            for i, rec in enumerate(records):
                cid = str(rec.get("chunk_id", str(i)))
                vec = chunk_id_to_emb.get(cid)
                if vec is not None:
                    emb_vecs.append(vec)
                else:
                    emb_vecs.append([0.0] * dim)
            emb_matrix = np.array(emb_vecs, dtype=np.float32)
            # Normalize for cosine similarity
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb_matrix = emb_matrix / norms
            print(f"Loaded {len(emb_list)} embeddings, dim={dim}")
        else:
            print(f"Warning: embeddings file not found: {emb_path}")

    # For embedding queries, use Ollama API
    import httpx

    def _get_query_embedding(text: str) -> List[float]:
        resp = httpx.post(
            f"{ollama_url}/api/embed",
            json={"model": embed_model, "input": text},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        embs = data.get("embeddings", [])
        if embs:
            return embs[0]
        raise ValueError(f"No embedding returned: {data}")

    results_out: List[dict] = []

    for row in eval_rows:
        qid = row.get("id")
        question = row.get("question")
        expected_type, correct_set = _resolve_expected_targets(row)
        # TF-IDF scores
        q_vec = vectorizer.transform([question])
        tfidf_scores = list(cosine_similarity(q_vec, matrix)[0])
        bm25_scores = bm25.score(_tokenize_for_bm25(question))

        for mode in modes:
            if mode == "tfidf":
                final_scores = tfidf_scores
            elif mode == "bm25":
                final_scores = bm25_scores
            elif mode == "embedding":
                if emb_matrix is not None:
                    q_emb = np.array(_get_query_embedding(question), dtype=np.float32)
                    q_norm = np.linalg.norm(q_emb)
                    if q_norm > 0:
                        q_emb = q_emb / q_norm
                    emb_scores = list(emb_matrix @ q_emb)
                    final_scores = emb_scores
                else:
                    continue
            elif mode == "hybrid":
                for alpha in alphas:
                    n1 = minmax_normalize(tfidf_scores)
                    n2 = minmax_normalize(bm25_scores)
                    final = [alpha * a + (1 - alpha) * b for a, b in zip(n1, n2)]
                    ranking_values = source_paths if expected_type == "source_path" else ids
                    ranked_values = _rank_unique_values(final, ranking_values, top_k)
                    p1 = precision_at_k(ranked_values, correct_set, 1)
                    p3 = precision_at_k(ranked_values, correct_set, 3)
                    rr = reciprocal_rank(ranked_values, correct_set)
                    ap = average_precision(ranked_values, correct_set)
                    results_out.append(
                        {
                            "id": qid,
                            "question": question,
                            "mode": mode,
                            "alpha": alpha,
                            "expected_type": expected_type,
                            "P@1": p1,
                            "P@3": p3,
                            "MRR": rr,
                            "MAP": ap,
                        }
                    )
                continue
            elif mode == "hybrid3":
                # TF-IDF + BM25 + Embedding の3方式ブレンド
                if emb_matrix is None:
                    continue
                q_emb = np.array(_get_query_embedding(question), dtype=np.float32)
                q_norm = np.linalg.norm(q_emb)
                if q_norm > 0:
                    q_emb = q_emb / q_norm
                emb_scores = list(emb_matrix @ q_emb)
                for alpha in alphas:
                    # alpha controls embedding weight; keyword weight = 1-alpha
                    # keyword part: existing hybrid (0.8*tfidf + 0.2*bm25)
                    n_tf = minmax_normalize(tfidf_scores)
                    n_bm = minmax_normalize(bm25_scores)
                    n_emb = minmax_normalize(emb_scores)
                    keyword = [0.8 * t + 0.2 * b for t, b in zip(n_tf, n_bm)]
                    n_kw = minmax_normalize(keyword)
                    final = [(1 - alpha) * k + alpha * e for k, e in zip(n_kw, n_emb)]
                    ranking_values = source_paths if expected_type == "source_path" else ids
                    ranked_values = _rank_unique_values(final, ranking_values, top_k)
                    p1 = precision_at_k(ranked_values, correct_set, 1)
                    p3 = precision_at_k(ranked_values, correct_set, 3)
                    rr = reciprocal_rank(ranked_values, correct_set)
                    ap = average_precision(ranked_values, correct_set)
                    results_out.append(
                        {
                            "id": qid,
                            "question": question,
                            "mode": mode,
                            "alpha": alpha,
                            "expected_type": expected_type,
                            "P@1": p1,
                            "P@3": p3,
                            "MRR": rr,
                            "MAP": ap,
                        }
                    )
                continue
            else:
                continue

            ranking_values = source_paths if expected_type == "source_path" else ids
            ranked_values = _rank_unique_values(final_scores, ranking_values, top_k)
            p1 = precision_at_k(ranked_values, correct_set, 1)
            p3 = precision_at_k(ranked_values, correct_set, 3)
            rr = reciprocal_rank(ranked_values, correct_set)
            ap = average_precision(ranked_values, correct_set)
            results_out.append(
                {
                    "id": qid,
                    "question": question,
                    "mode": mode,
                    "alpha": "",
                    "expected_type": expected_type,
                    "P@1": p1,
                    "P@3": p3,
                    "MRR": rr,
                    "MAP": ap,
                }
            )

    return results_out


def read_eval_csv(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval (tfidf/bm25/hybrid)")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--eval", default="data/eval_sample.csv")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--modes", default="tfidf,bm25,hybrid")
    parser.add_argument("--alphas", default="0.5")
    parser.add_argument("--embeddings", default="data/embeddings.json", help="Pre-computed embeddings JSON")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    eval_path = Path(args.eval)
    if not chunks_path.exists():
        print(f"chunks file not found: {chunks_path}")
        return 2
    if not eval_path.exists():
        print(f"eval csv not found: {eval_path}")
        return 2

    records = load_chunks_from_jsonl(chunks_path)
    eval_rows = read_eval_csv(eval_path)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]

    results = evaluate_all(
        records, eval_rows, top_k=args.top_k, modes=modes, alphas=alphas,
        embeddings_path=args.embeddings, ollama_url=args.ollama_url, embed_model=args.embed_model,
    )

    out_path = Path("eval_results.csv")
    keys = ["id", "question", "mode", "alpha", "expected_type", "P@1", "P@3", "MRR", "MAP"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in results:
            w.writerow(r)

    # summary
    summary = {}
    for r in results:
        key = (r["mode"], str(r.get("alpha", "")))
        stat = summary.setdefault(key, {"P@1": [], "P@3": [], "MRR": [], "MAP": []})
        stat["P@1"].append(float(r["P@1"]))
        stat["P@3"].append(float(r["P@3"]))
        stat["MRR"].append(float(r["MRR"]))
        stat["MAP"].append(float(r["MAP"]))

    print("Summary:")
    for (mode, alpha), vals in summary.items():
        n = len(vals["P@1"])
        print(f"mode={mode} alpha={alpha} n={n} P@1={sum(vals['P@1'])/n:.3f} P@3={sum(vals['P@3'])/n:.3f} MRR={sum(vals['MRR'])/n:.3f} MAP={sum(vals['MAP'])/n:.3f}")

    print(f"Wrote per-question results -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
