"""app.py のユニットテスト。

サーバ起動不要のロジック単体テスト。
BM25、TF-IDF、ハイブリッド検索、クエリ展開、ヘルパー関数などをテストする。
"""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# app.py はサーバ起動しないと一部機能が動かないが、
# 関数単位でインポートしてテストする
import app


# ============================================================
# SimpleBM25
# ============================================================

class TestSimpleBM25:
    def test_empty_corpus(self):
        bm25 = app.SimpleBM25([], k1=0.5, b=0.3)
        scores = bm25.score(["テスト"])
        assert scores == []

    def test_single_doc(self):
        corpus = [["VPN", "接続", "トラブル"]]
        bm25 = app.SimpleBM25(corpus)
        scores = bm25.score(["VPN"])
        assert len(scores) == 1
        assert scores[0] > 0

    def test_unknown_token_zero(self):
        corpus = [["VPN", "接続"]]
        bm25 = app.SimpleBM25(corpus)
        scores = bm25.score(["存在しないトークン"])
        assert scores == [0.0]

    def test_relevant_doc_scores_higher(self):
        corpus = [
            ["VPN", "接続", "トラブル", "ネットワーク"],
            ["Excel", "関数", "VLOOKUP", "集計"],
            ["VPN", "設定", "接続"],
        ]
        bm25 = app.SimpleBM25(corpus)
        scores = bm25.score(["VPN", "接続"])
        # VPN+接続を含む文書(0, 2)がExcel文書(1)より高スコア
        assert scores[0] > scores[1]
        assert scores[2] > scores[1]

    def test_idf_calculation(self):
        corpus = [["a", "b"], ["a", "c"], ["a", "d"]]
        bm25 = app.SimpleBM25(corpus)
        # "a" は全文書に出現 → IDF低い、"b" は1文書のみ → IDF高い
        assert bm25.idf["b"] > bm25.idf["a"]


# ============================================================
# _minmax_normalize
# ============================================================

class TestMinmaxNormalize:
    def test_normal(self):
        result = app._minmax_normalize([1.0, 2.0, 3.0])
        assert result[0] == 0.0
        assert result[2] == 1.0
        assert abs(result[1] - 0.5) < 1e-6

    def test_all_same(self):
        result = app._minmax_normalize([5.0, 5.0, 5.0])
        assert result == [0.0, 0.0, 0.0]

    def test_empty(self):
        result = app._minmax_normalize([])
        assert result == []


# ============================================================
# _build_search_text
# ============================================================

class TestBuildSearchText:
    def test_contains_metadata(self):
        rec = {
            "title": "VPN接続トラブル",
            "topic": "VPN",
            "product": "社内ネットワーク",
            "tags": ["VPN", "接続"],
            "text": "本文テキスト",
        }
        result = app._build_search_text(rec)
        assert "VPN接続トラブル" in result
        assert "社内ネットワーク" in result
        assert "本文テキスト" in result

    def test_tags_as_string(self):
        rec = {"title": "T", "tags": "タグ文字列", "text": "本文"}
        result = app._build_search_text(rec)
        assert "タグ文字列" in result

    def test_missing_fields(self):
        rec = {"text": "本文のみ"}
        result = app._build_search_text(rec)
        assert "本文のみ" in result


# ============================================================
# _tokenize_for_bm25
# ============================================================

class TestTokenizeForBM25:
    def test_japanese(self):
        tokens = app._tokenize_for_bm25("VPN接続トラブル")
        assert "vpn" in tokens  # 英字は小文字化
        assert any("接続" in t for t in tokens)  # 日本語部分が含まれる

    def test_ngrams_generated(self):
        tokens = app._tokenize_for_bm25("テスト")
        # 2-gram/3-gramが生成される
        assert any(len(t) == 2 for t in tokens)
        assert any(len(t) == 3 for t in tokens)


# ============================================================
# _build_query_variants (同義語展開)
# ============================================================

class TestBuildQueryVariants:
    def test_no_synonym_hit(self):
        variants = app._build_query_variants("独自の質問テキスト")
        assert len(variants) >= 1
        assert variants[0][0] == "独自の質問テキスト"
        assert variants[0][1] == 1.0

    def test_synonym_expansion(self):
        # 「エラー」は synonym group に含まれるはず
        variants = app._build_query_variants("エラーが発生した")
        if len(variants) > 1:
            # 展開された場合、「不具合」「障害」等が含まれる
            expanded = variants[1][0]
            assert "エラーが発生した" in expanded
            assert variants[1][1] < 1.0  # 展開クエリは低い重み


# ============================================================
# ヘルパー関数
# ============================================================

class TestHelperFunctions:
    def test_pick_source_path(self):
        assert app._pick_source_path({"source_path": "data/chunk/test.md"}) == "data/chunk/test.md"
        assert app._pick_source_path({"source_uri": "data/chunk/test2.md"}) == "data/chunk/test2.md"
        assert app._pick_source_path({}) == ""

    def test_pick_source_path_normalizes_backslash(self):
        result = app._pick_source_path({"source_path": "data\\chunk\\test.md"})
        assert "\\" not in result

    def test_pick_doc_type(self):
        assert app._pick_doc_type({"doc_type": "Troubleshooting"}) == "troubleshooting"
        assert app._pick_doc_type({"doc_type": "HOWTO "}) == "howto"
        assert app._pick_doc_type({}) == ""

    def test_is_allowed_source(self):
        assert app._is_allowed_source({"source_path": "data/chunk/VPN.md"})
        assert not app._is_allowed_source({"source_path": "malicious/path.md"})

    def test_choose_answer_style_empty(self):
        assert app._choose_answer_style([]) == "default"

    def test_choose_answer_style_troubleshooting(self):
        items = [{"doc_type": "troubleshooting", "score": 0.9}]
        assert app._choose_answer_style(items) == "troubleshooting"

    def test_normalize_related_ids_string(self):
        assert app._normalize_related_ids("abc") == ["abc"]
        assert app._normalize_related_ids("  ") == []

    def test_normalize_related_ids_list(self):
        assert app._normalize_related_ids(["a", "b"]) == ["a", "b"]
        assert app._normalize_related_ids([" ", "x"]) == ["x"]

    def test_to_int(self):
        assert app._to_int("42") == 42
        assert app._to_int("abc", default=0) == 0
        assert app._to_int(None, default=-1) == -1


# ============================================================
# load_index (chunks.jsonl からの構築)
# ============================================================

class TestLoadIndex:
    def test_load_from_jsonl(self):
        """chunks.jsonl からインデックスを構築できるか"""
        jsonl_path = ROOT / "data" / "chunks.jsonl"
        if not jsonl_path.exists():
            pytest.skip("chunks.jsonl not found")

        # 存在しないjoblibパスを指定 → chunks.jsonlからfallback構築
        fake_joblib = ROOT / "data" / "nonexistent.joblib"
        payload = app.load_index(fake_joblib)

        assert "records" in payload
        assert "vectorizer" in payload
        assert "matrix" in payload
        assert "bm25" in payload
        assert len(payload["records"]) > 0

    def test_records_have_required_fields(self):
        jsonl_path = ROOT / "data" / "chunks.jsonl"
        if not jsonl_path.exists():
            pytest.skip("chunks.jsonl not found")

        fake_joblib = ROOT / "data" / "nonexistent.joblib"
        payload = app.load_index(fake_joblib)
        for rec in payload["records"]:
            assert "chunk_id" in rec
            assert "title" in rec
            assert "source_path" in rec
            assert "text" in rec


# ============================================================
# retrieve_topk (検索パイプライン結合テスト)
# ============================================================

class TestRetrieveTopk:
    @pytest.fixture(autouse=True)
    def setup_payload(self):
        jsonl_path = ROOT / "data" / "chunks.jsonl"
        if not jsonl_path.exists():
            pytest.skip("chunks.jsonl not found")
        fake_joblib = ROOT / "data" / "nonexistent.joblib"
        self.payload = app.load_index(fake_joblib)

    def test_tfidf_search(self):
        results = app.retrieve_topk(
            self.payload,
            ROOT / "data" / "nonexistent.joblib",
            query="VPN接続できない",
            top_k=3,
            retrieval_mode="tfidf",
        )
        assert len(results) == 3
        for r in results:
            assert "score" in r
            assert "text" in r

    def test_bm25_search(self):
        results = app.retrieve_topk(
            self.payload,
            ROOT / "data" / "nonexistent.joblib",
            query="パスワードリセット",
            top_k=2,
            retrieval_mode="bm25",
        )
        assert len(results) == 2

    def test_hybrid_search(self):
        results = app.retrieve_topk(
            self.payload,
            ROOT / "data" / "nonexistent.joblib",
            query="プリンターから印刷できない",
            top_k=3,
            retrieval_mode="hybrid",
        )
        assert len(results) == 3
        # スコア降順であること
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_results_have_score(self):
        results = app.retrieve_topk(
            self.payload,
            ROOT / "data" / "nonexistent.joblib",
            query="OneDrive同期",
            top_k=2,
            retrieval_mode="tfidf",
        )
        for r in results:
            assert isinstance(r["score"], float)
            assert r["score"] >= 0

    def test_vpn_query_finds_vpn_doc(self):
        results = app.retrieve_topk(
            self.payload,
            ROOT / "data" / "nonexistent.joblib",
            query="VPN接続がタイムアウトする",
            top_k=3,
            retrieval_mode="hybrid",
        )
        sources = [app._pick_source_path(r) for r in results]
        assert any("VPN" in s for s in sources), f"VPN doc not in top 3: {sources}"

    def test_password_query_finds_password_doc(self):
        results = app.retrieve_topk(
            self.payload,
            ROOT / "data" / "nonexistent.joblib",
            query="パスワードを忘れてログインできない",
            top_k=3,
            retrieval_mode="hybrid",
        )
        sources = [app._pick_source_path(r) for r in results]
        assert any("パスワード" in s for s in sources), f"Password doc not in top 3: {sources}"

    def test_printer_query_finds_printer_doc(self):
        results = app.retrieve_topk(
            self.payload,
            ROOT / "data" / "nonexistent.joblib",
            query="印刷ジョブが詰まって進まない",
            top_k=3,
            retrieval_mode="hybrid",
        )
        sources = [app._pick_source_path(r) for r in results]
        assert any("プリンター" in s for s in sources), f"Printer doc not in top 3: {sources}"
