"""Rerank 付き検索精度評価（LLM不要・高速版）

hybrid3 の top-10 から Cross-Encoder で rerank → P@1/MRR を計測。
比較用に rerank なし hybrid3 も同時に計測する。
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---- SimpleBM25 (app.py と同じ実装) ----
class SimpleBM25:
    def __init__(self, tokenized_corpus, k1=0.5, b=0.3):
        self.k1, self.b = k1, b
        self.doc_count = len(tokenized_corpus)
        self.doc_lengths = [len(d) for d in tokenized_corpus]
        self.avg_doc_length = (sum(self.doc_lengths) / self.doc_count) if self.doc_count else 0.0
        self.idf, self.postings = {}, {}
        doc_freq = {}
        for idx, doc in enumerate(tokenized_corpus):
            tf = {}
            for t in doc:
                if t:
                    tf[t] = tf.get(t, 0) + 1
            for t, freq in tf.items():
                self.postings.setdefault(t, []).append((idx, freq))
                doc_freq[t] = doc_freq.get(t, 0) + 1
        for t, df in doc_freq.items():
            self.idf[t] = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

    def score(self, query_tokens):
        scores = [0.0] * self.doc_count
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for di, tf in self.postings.get(t, []):
                dl = self.doc_lengths[di]
                norm = self.k1 * (1.0 - self.b + self.b * (dl / self.avg_doc_length if self.avg_doc_length else 1.0))
                scores[di] += idf * ((tf * (self.k1 + 1.0)) / (tf + norm))
        return scores


def tokenize(text):
    n = re.sub(r"\s+", " ", str(text).lower()).strip()
    tokens = re.findall(r"[0-9a-z_./-]+|[一-龥ぁ-んァ-ン]+", n)
    c = n.replace(" ", "")
    for ng in (2, 3):
        if len(c) >= ng:
            tokens.extend(c[i:i + ng] for i in range(len(c) - ng + 1))
    return tokens


def minmax(scores):
    mn, mx = min(scores), max(scores)
    if mx <= mn:
        return [0.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def build_text(rec):
    tags = rec.get("tags", [])
    tags_text = " ".join(str(t) for t in tags) if isinstance(tags, list) else str(tags or "")
    return "\n".join([
        f"title: {rec.get('title', '')}",
        f"topic: {rec.get('topic', '')}",
        f"product: {rec.get('product', '')}",
        f"tags: {tags_text}",
        str(rec.get("text", "")),
    ])


def main():
    chunks_path = Path("data/chunks.jsonl")
    eval_path = Path("data/eval_tuning.csv")
    emb_path = Path("data/embeddings_bge_m3.json")
    ollama_url = "http://localhost:11434"
    embed_model = "bge-m3"
    initial_k = 10

    # Load chunks
    records = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Chunks: {len(records)}")

    # Build search
    docs = [build_text(r) for r in records]
    source_paths = [str(r.get("source_path", "")).strip().replace("\\", "/") for r in records]

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    matrix = vectorizer.fit_transform(docs)

    corpus_tokens = [tokenize(d) for d in docs]
    bm25 = SimpleBM25(corpus_tokens, k1=0.5, b=0.3)

    # Load embeddings
    with emb_path.open("r", encoding="utf-8") as f:
        emb_data = json.load(f)
    emb_list = emb_data.get("embeddings", [])
    chunk_id_to_emb = {item["chunk_id"]: item["embedding"] for item in emb_list}
    dim = len(emb_list[0]["embedding"]) if emb_list else 1024
    emb_vecs = []
    for i, rec in enumerate(records):
        cid = str(rec.get("chunk_id", str(i)))
        vec = chunk_id_to_emb.get(cid, [0.0] * dim)
        emb_vecs.append(vec)
    emb_matrix = np.array(emb_vecs, dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix = emb_matrix / norms
    print(f"Embeddings: {emb_matrix.shape}")

    # Load reranker
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
        print("Reranker: loaded")
    except ImportError:
        print("ERROR: sentence-transformers not installed")
        sys.exit(1)

    # Load eval set
    eval_rows = list(csv.DictReader(eval_path.open("r", encoding="utf-8")))
    print(f"Eval questions: {len(eval_rows)}")

    # Metrics
    baseline_hit1, baseline_rr = 0, 0.0
    rerank_hit1, rerank_rr = 0, 0.0
    total = 0

    for row in eval_rows:
        qid = row["id"]
        question = row["question"]
        expected_set = {
            p.strip().replace("\\", "/")
            for p in row.get("expected_source_path", "").split("|")
            if p.strip()
        }
        if not expected_set:
            continue
        total += 1

        # TF-IDF scores
        q_vec = vectorizer.transform([question])
        tfidf_scores = list(cosine_similarity(q_vec, matrix)[0])

        # BM25 scores
        bm25_scores = bm25.score(tokenize(question))

        # Embedding scores
        q_emb_resp = httpx.post(
            f"{ollama_url}/api/embed",
            json={"model": embed_model, "input": question},
            timeout=60,
        )
        q_emb_resp.raise_for_status()
        q_emb = np.array(q_emb_resp.json()["embeddings"][0], dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm
        emb_scores = list(emb_matrix @ q_emb)

        # hybrid3: 0.8*tfidf + 0.2*bm25 (keyword), then 0.3*keyword + 0.7*embedding
        n_tf = minmax(tfidf_scores)
        n_bm = minmax(bm25_scores)
        n_emb = minmax(emb_scores)
        keyword = [0.8 * t + 0.2 * b for t, b in zip(n_tf, n_bm)]
        n_kw = minmax(keyword)
        final = [(1 - 0.7) * k + 0.7 * e for k, e in zip(n_kw, n_emb)]

        # Baseline: hybrid3 top-5 (no rerank)
        ranked_idx = sorted(range(len(final)), key=lambda i: final[i], reverse=True)
        seen = set()
        baseline_ranked = []
        for idx in ranked_idx:
            sp = source_paths[idx]
            if sp not in seen:
                seen.add(sp)
                baseline_ranked.append(sp)
            if len(baseline_ranked) >= 5:
                break

        if baseline_ranked and baseline_ranked[0] in expected_set:
            baseline_hit1 += 1
        for i, p in enumerate(baseline_ranked, 1):
            if p in expected_set:
                baseline_rr += 1.0 / i
                break

        # Rerank: take top initial_k, rerank with CrossEncoder
        # app.py は rec.get("text", "") = rawテキストのみを reranker に渡す
        top_indices = ranked_idx[:initial_k]
        pairs = [(question, str(records[idx].get("text", ""))) for idx in top_indices]
        rerank_scores = reranker.predict(pairs)

        reranked_order = sorted(range(len(top_indices)), key=lambda i: rerank_scores[i], reverse=True)
        seen2 = set()
        rerank_ranked = []
        for ri in reranked_order:
            sp = source_paths[top_indices[ri]]
            if sp not in seen2:
                seen2.add(sp)
                rerank_ranked.append(sp)
            if len(rerank_ranked) >= 5:
                break

        if rerank_ranked and rerank_ranked[0] in expected_set:
            rerank_hit1 += 1
        for i, p in enumerate(rerank_ranked, 1):
            if p in expected_set:
                rerank_rr += 1.0 / i
                break

        # Show misses for rerank
        if not (rerank_ranked and rerank_ranked[0] in expected_set):
            b_status = "HIT" if (baseline_ranked and baseline_ranked[0] in expected_set) else "MISS"
            print(f"  Q{qid} rerank MISS (baseline={b_status}): expected={row.get('expected_source_path','')} got={rerank_ranked[:3]}")

        if total % 20 == 0:
            print(f"  ... {total} questions processed")

    print(f"\n{'='*60}")
    print(f"Total questions: {total}")
    print(f"")
    print(f"--- Baseline (hybrid3, no rerank) ---")
    print(f"  P@1 = {baseline_hit1/total:.4f} ({baseline_hit1}/{total})")
    print(f"  MRR = {baseline_rr/total:.4f}")
    print(f"")
    print(f"--- Rerank (hybrid3 + bge-reranker-v2-m3, initial_k={initial_k}) ---")
    print(f"  P@1 = {rerank_hit1/total:.4f} ({rerank_hit1}/{total})")
    print(f"  MRR = {rerank_rr/total:.4f}")
    print(f"")
    print(f"--- Delta ---")
    print(f"  P@1: {(rerank_hit1-baseline_hit1)/total:+.4f}")
    print(f"  MRR: {(rerank_rr-baseline_rr)/total:+.4f}")


if __name__ == "__main__":
    main()
