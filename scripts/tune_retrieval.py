"""Retrieval parameter grid-search tuner.

Usage:
  python scripts/tune_retrieval.py --chunks data/chunks.jsonl --eval data/eval_tuning.csv

Searches over BM25(k1, b), TF-IDF ngram_range, and hybrid alpha.
Outputs a ranked results CSV and prints the best configuration.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Data helpers (shared with eval_retrieval.py)
# ---------------------------------------------------------------------------

def load_chunks_from_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def build_search_text(rec: dict) -> str:
    tags = rec.get("tags", [])
    if isinstance(tags, list):
        tags_text = " ".join(str(t) for t in tags)
    else:
        tags_text = str(tags) if tags else ""
    return "\n".join([
        f"title: {rec.get('title', '')}",
        f"topic: {rec.get('topic', '')}",
        f"product: {rec.get('product', '')}",
        f"tags: {tags_text}",
        str(rec.get("text", "")),
    ])


def read_eval_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def _split_expected_values(raw: Any) -> Set[str]:
    return {_normalize_key(p) for p in str(raw or "").split("|") if _normalize_key(p)}


def _resolve_expected_targets(row: dict) -> Tuple[str, Set[str]]:
    for key in ("correct_chunk_ids", "correct_chunk_id", "expected_chunk_ids", "expected_chunk_id"):
        vals = _split_expected_values(row.get(key))
        if vals:
            return "chunk_id", vals
    for key in ("expected_source_paths", "expected_source_path", "correct_source_paths", "correct_source_path"):
        vals = _split_expected_values(row.get(key))
        if vals:
            return "source_path", vals
    return "chunk_id", set()


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

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
        scores = [0.0] * self.doc_count
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
        tokens.extend(compact[i:i + n] for i in range(len(compact) - n + 1))
    return tokens


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def minmax_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx <= mn:
        return [0.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def precision_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    topk = ranked[:k]
    return sum(1 for x in topk if x in relevant) / len(topk) if topk else 0.0


def reciprocal_rank(ranked: List[str], relevant: Set[str]) -> float:
    for i, x in enumerate(ranked, 1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def average_precision(ranked: List[str], relevant: Set[str]) -> float:
    if not relevant:
        return 0.0
    hit, sum_prec = 0, 0.0
    for i, x in enumerate(ranked, 1):
        if x in relevant:
            hit += 1
            sum_prec += hit / i
    return sum_prec / len(relevant)


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _rank_unique_values(scores: List[float], values: List[str], top_k: int) -> List[str]:
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked_values = [values[i] for i in ranked_idx]
    return _dedupe_preserve_order(ranked_values)[:top_k]


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid_search(
    records: List[dict],
    eval_rows: List[dict],
    top_k: int = 10,
    bm25_k1_values: List[float] = None,
    bm25_b_values: List[float] = None,
    tfidf_ngram_ranges: List[Tuple[int, int]] = None,
    alpha_values: List[float] = None,
) -> List[dict]:
    if bm25_k1_values is None:
        bm25_k1_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    if bm25_b_values is None:
        bm25_b_values = [0.3, 0.5, 0.65, 0.75, 0.85]
    if tfidf_ngram_ranges is None:
        tfidf_ngram_ranges = [(2, 3), (2, 4), (2, 5), (3, 5)]
    if alpha_values is None:
        alpha_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    docs = [build_search_text(r) for r in records]
    source_paths = [_normalize_key(r.get("source_path") or r.get("source_uri") or "") for r in records]
    ids = [str(r.get("chunk_id") or r.get("title") or i) for i, r in enumerate(records)]

    corpus_tokens = [_tokenize_for_bm25(d) for d in docs]

    # Parse expected targets once
    parsed_eval = []
    for row in eval_rows:
        qid = row.get("id")
        question = row.get("question")
        expected_type, correct_set = _resolve_expected_targets(row)
        if not correct_set:
            continue
        parsed_eval.append((qid, question, expected_type, correct_set))

    if not parsed_eval:
        print("ERROR: No valid evaluation rows found.")
        return []

    total_configs = (
        len(tfidf_ngram_ranges)
        + len(bm25_k1_values) * len(bm25_b_values)
        + len(tfidf_ngram_ranges) * len(bm25_k1_values) * len(bm25_b_values) * len(alpha_values)
    )
    print(f"Grid search: {total_configs} configurations x {len(parsed_eval)} questions")

    all_results: List[dict] = []
    config_count = 0

    # --- TF-IDF only ---
    for ngram_range in tfidf_ngram_ranges:
        config_count += 1
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=ngram_range)
        matrix = vectorizer.fit_transform(docs)

        metrics_agg = {"P@1": [], "P@3": [], "MRR": [], "MAP": []}
        per_question = []
        for qid, question, expected_type, correct_set in parsed_eval:
            q_vec = vectorizer.transform([question])
            tfidf_scores = list(cosine_similarity(q_vec, matrix)[0])
            ranking_values = source_paths if expected_type == "source_path" else ids
            ranked = _rank_unique_values(tfidf_scores, ranking_values, top_k)
            p1 = precision_at_k(ranked, correct_set, 1)
            p3 = precision_at_k(ranked, correct_set, 3)
            rr = reciprocal_rank(ranked, correct_set)
            ap = average_precision(ranked, correct_set)
            metrics_agg["P@1"].append(p1)
            metrics_agg["P@3"].append(p3)
            metrics_agg["MRR"].append(rr)
            metrics_agg["MAP"].append(ap)
            per_question.append({"id": qid, "P@1": p1, "MRR": rr})

        n = len(metrics_agg["P@1"])
        result = {
            "mode": "tfidf",
            "ngram_range": f"{ngram_range[0]}-{ngram_range[1]}",
            "bm25_k1": "",
            "bm25_b": "",
            "alpha": "",
            "mean_P@1": sum(metrics_agg["P@1"]) / n,
            "mean_P@3": sum(metrics_agg["P@3"]) / n,
            "mean_MRR": sum(metrics_agg["MRR"]) / n,
            "mean_MAP": sum(metrics_agg["MAP"]) / n,
            "n": n,
        }
        all_results.append(result)
        if config_count % 20 == 0 or config_count == total_configs:
            print(f"  [{config_count}/{total_configs}] ...")

    # --- BM25 only ---
    for k1, b in itertools.product(bm25_k1_values, bm25_b_values):
        config_count += 1
        bm25 = SimpleBM25(corpus_tokens, k1=k1, b=b)

        metrics_agg = {"P@1": [], "P@3": [], "MRR": [], "MAP": []}
        for qid, question, expected_type, correct_set in parsed_eval:
            bm25_scores = bm25.score(_tokenize_for_bm25(question))
            ranking_values = source_paths if expected_type == "source_path" else ids
            ranked = _rank_unique_values(bm25_scores, ranking_values, top_k)
            p1 = precision_at_k(ranked, correct_set, 1)
            p3 = precision_at_k(ranked, correct_set, 3)
            rr = reciprocal_rank(ranked, correct_set)
            ap = average_precision(ranked, correct_set)
            metrics_agg["P@1"].append(p1)
            metrics_agg["P@3"].append(p3)
            metrics_agg["MRR"].append(rr)
            metrics_agg["MAP"].append(ap)

        n = len(metrics_agg["P@1"])
        result = {
            "mode": "bm25",
            "ngram_range": "",
            "bm25_k1": k1,
            "bm25_b": b,
            "alpha": "",
            "mean_P@1": sum(metrics_agg["P@1"]) / n,
            "mean_P@3": sum(metrics_agg["P@3"]) / n,
            "mean_MRR": sum(metrics_agg["MRR"]) / n,
            "mean_MAP": sum(metrics_agg["MAP"]) / n,
            "n": n,
        }
        all_results.append(result)
        if config_count % 20 == 0 or config_count == total_configs:
            print(f"  [{config_count}/{total_configs}] ...")

    # --- Hybrid (TF-IDF + BM25 blended) ---
    for ngram_range in tfidf_ngram_ranges:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=ngram_range)
        matrix = vectorizer.fit_transform(docs)

        # Pre-compute TF-IDF scores for all questions
        tfidf_per_q: List[List[float]] = []
        for _, question, _, _ in parsed_eval:
            q_vec = vectorizer.transform([question])
            tfidf_per_q.append(list(cosine_similarity(q_vec, matrix)[0]))

        for k1, b in itertools.product(bm25_k1_values, bm25_b_values):
            bm25 = SimpleBM25(corpus_tokens, k1=k1, b=b)

            bm25_per_q: List[List[float]] = []
            for _, question, _, _ in parsed_eval:
                bm25_per_q.append(bm25.score(_tokenize_for_bm25(question)))

            for alpha in alpha_values:
                config_count += 1
                metrics_agg = {"P@1": [], "P@3": [], "MRR": [], "MAP": []}

                for qi, (qid, question, expected_type, correct_set) in enumerate(parsed_eval):
                    n1 = minmax_normalize(tfidf_per_q[qi])
                    n2 = minmax_normalize(bm25_per_q[qi])
                    final = [alpha * a + (1 - alpha) * b_val for a, b_val in zip(n1, n2)]
                    ranking_values = source_paths if expected_type == "source_path" else ids
                    ranked = _rank_unique_values(final, ranking_values, top_k)
                    p1 = precision_at_k(ranked, correct_set, 1)
                    p3 = precision_at_k(ranked, correct_set, 3)
                    rr = reciprocal_rank(ranked, correct_set)
                    ap = average_precision(ranked, correct_set)
                    metrics_agg["P@1"].append(p1)
                    metrics_agg["P@3"].append(p3)
                    metrics_agg["MRR"].append(rr)
                    metrics_agg["MAP"].append(ap)

                n = len(metrics_agg["P@1"])
                result = {
                    "mode": "hybrid",
                    "ngram_range": f"{ngram_range[0]}-{ngram_range[1]}",
                    "bm25_k1": k1,
                    "bm25_b": b,
                    "alpha": alpha,
                    "mean_P@1": sum(metrics_agg["P@1"]) / n,
                    "mean_P@3": sum(metrics_agg["P@3"]) / n,
                    "mean_MRR": sum(metrics_agg["MRR"]) / n,
                    "mean_MAP": sum(metrics_agg["MAP"]) / n,
                    "n": n,
                }
                all_results.append(result)
                if config_count % 50 == 0 or config_count == total_configs:
                    print(f"  [{config_count}/{total_configs}] ...")

    return all_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid-search retrieval parameters")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--eval", default="data/eval_tuning.csv")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", default="eval_tuning_results.csv")
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
    print(f"Loaded {len(records)} chunks, {len(eval_rows)} eval questions")

    results = run_grid_search(records, eval_rows, top_k=args.top_k)

    # Sort by mean_MRR descending, then mean_MAP descending
    results.sort(key=lambda r: (-r["mean_MRR"], -r["mean_MAP"], -r["mean_P@1"]))

    out_path = Path(args.out)
    keys = ["mode", "ngram_range", "bm25_k1", "bm25_b", "alpha",
            "mean_P@1", "mean_P@3", "mean_MRR", "mean_MAP", "n"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\n{'='*70}")
    print(f"Results written to {out_path} ({len(results)} configurations)")
    print(f"{'='*70}")

    # Print top 10 configs
    print("\n--- TOP 10 configurations (by MRR) ---\n")
    print(f"{'Rank':<5} {'Mode':<8} {'ngram':<7} {'k1':<5} {'b':<5} {'alpha':<6} {'P@1':<7} {'P@3':<7} {'MRR':<7} {'MAP':<7}")
    print("-" * 70)
    for i, r in enumerate(results[:10], 1):
        ngram = r["ngram_range"] or "-"
        k1 = f"{r['bm25_k1']}" if r["bm25_k1"] != "" else "-"
        b = f"{r['bm25_b']}" if r["bm25_b"] != "" else "-"
        alpha = f"{r['alpha']}" if r["alpha"] != "" else "-"
        print(f"{i:<5} {r['mode']:<8} {ngram:<7} {k1:<5} {b:<5} {alpha:<6} "
              f"{r['mean_P@1']:<7.3f} {r['mean_P@3']:<7.3f} {r['mean_MRR']:<7.3f} {r['mean_MAP']:<7.3f}")

    # Print best per mode
    print("\n--- Best per mode ---\n")
    for mode in ("tfidf", "bm25", "hybrid"):
        mode_results = [r for r in results if r["mode"] == mode]
        if mode_results:
            best = mode_results[0]
            params = []
            if best["ngram_range"]:
                params.append(f"ngram={best['ngram_range']}")
            if best["bm25_k1"] != "":
                params.append(f"k1={best['bm25_k1']}")
            if best["bm25_b"] != "":
                params.append(f"b={best['bm25_b']}")
            if best["alpha"] != "":
                params.append(f"alpha={best['alpha']}")
            print(f"  {mode}: MRR={best['mean_MRR']:.3f}  MAP={best['mean_MAP']:.3f}  P@1={best['mean_P@1']:.3f}  ({', '.join(params)})")

    # Print recommended env vars
    print("\n--- Recommended .env settings ---\n")
    best_overall = results[0]
    mode = best_overall["mode"]
    print(f"  RAG_RETRIEVAL_MODE={mode}")
    if best_overall["ngram_range"]:
        lo, hi = best_overall["ngram_range"].split("-")
        print(f"  # TF-IDF ngram_range: ({lo}, {hi})")
    if best_overall["bm25_k1"] != "":
        print(f"  # BM25 k1={best_overall['bm25_k1']}")
    if best_overall["bm25_b"] != "":
        print(f"  # BM25 b={best_overall['bm25_b']}")
    if best_overall["alpha"] != "":
        print(f"  # Hybrid alpha={best_overall['alpha']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
