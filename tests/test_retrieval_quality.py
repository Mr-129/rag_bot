"""検索品質テスト。

eval_tuning.csv の質問を使い、検索結果の上位に期待するドキュメントが含まれるかテストする。
"""
import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app


def _load_eval_tuning() -> list[dict]:
    path = ROOT / "data" / "eval_tuning.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def rag_payload():
    jsonl_path = ROOT / "data" / "chunks.jsonl"
    if not jsonl_path.exists():
        pytest.skip("chunks.jsonl not found")
    fake_joblib = ROOT / "data" / "nonexistent.joblib"
    return app.load_index(fake_joblib)


class TestRetrievalQuality:
    """eval_tuning.csv を使った検索品質テスト"""

    @pytest.fixture(autouse=True)
    def setup(self, rag_payload):
        self.payload = rag_payload
        self.eval_rows = _load_eval_tuning()
        if not self.eval_rows:
            pytest.skip("eval_tuning.csv not available")

    def _check_hit(self, query: str, expected_source: str, mode: str, top_k: int = 5) -> bool:
        """expected_source が検索結果の top_k に含まれるかチェック"""
        results = app.retrieve_topk(
            self.payload,
            ROOT / "data" / "nonexistent.joblib",
            query=query,
            top_k=top_k,
            retrieval_mode=mode,
        )
        sources = [app._pick_source_path(r) for r in results]
        return any(expected_source in s for s in sources)

    def test_hybrid_recall_at_5(self):
        """hybrid モードの Recall@5 が 50% 以上であること"""
        hits = 0
        total = 0
        for row in self.eval_rows:
            query = row.get("question") or row.get("query", "")
            expected = row.get("expected_source_path", row.get("expected_source", "")).strip()
            if not query or not expected:
                continue
            total += 1
            if self._check_hit(query, expected, mode="hybrid", top_k=5):
                hits += 1

        if total == 0:
            pytest.skip("No valid eval rows")
        recall = hits / total
        assert recall >= 0.5, f"hybrid Recall@5 = {recall:.2%} ({hits}/{total}), expected >= 50%"

    def test_tfidf_recall_at_5(self):
        """tfidf モードの Recall@5 が 40% 以上であること"""
        hits = 0
        total = 0
        for row in self.eval_rows:
            query = row.get("question") or row.get("query", "")
            expected = row.get("expected_source_path", row.get("expected_source", "")).strip()
            if not query or not expected:
                continue
            total += 1
            if self._check_hit(query, expected, mode="tfidf", top_k=5):
                hits += 1

        if total == 0:
            pytest.skip("No valid eval rows")
        recall = hits / total
        assert recall >= 0.4, f"tfidf Recall@5 = {recall:.2%} ({hits}/{total}), expected >= 40%"

    def test_bm25_recall_at_5(self):
        """bm25 モードの Recall@5 が 40% 以上であること"""
        hits = 0
        total = 0
        for row in self.eval_rows:
            query = row.get("question") or row.get("query", "")
            expected = row.get("expected_source_path", row.get("expected_source", "")).strip()
            if not query or not expected:
                continue
            total += 1
            if self._check_hit(query, expected, mode="bm25", top_k=5):
                hits += 1

        if total == 0:
            pytest.skip("No valid eval rows")
        recall = hits / total
        assert recall >= 0.4, f"bm25 Recall@5 = {recall:.2%} ({hits}/{total}), expected >= 40%"
