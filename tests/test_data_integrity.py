"""データ整合性テスト。

chunks.jsonl、チャンクファイル、eval データの整合性を検証する。
"""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


class TestChunksJsonl:
    """chunks.jsonl の整合性テスト"""

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.jsonl_path = ROOT / "data" / "chunks.jsonl"
        if not self.jsonl_path.exists():
            pytest.skip("chunks.jsonl not found")
        self.records = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

    def test_not_empty(self):
        assert len(self.records) > 0, "chunks.jsonl is empty"

    def test_all_records_are_dicts(self):
        for i, rec in enumerate(self.records):
            assert isinstance(rec, dict), f"Record {i} is not a dict: {type(rec)}"

    def test_required_fields_present(self):
        required = {"chunk_id", "title", "source_path", "text"}
        for i, rec in enumerate(self.records):
            missing = required - set(rec.keys())
            assert not missing, f"Record {i} missing fields: {missing}"

    def test_no_empty_text(self):
        for i, rec in enumerate(self.records):
            text = rec.get("text", "")
            assert text.strip(), f"Record {i} has empty text"

    def test_chunk_ids_unique(self):
        ids = [rec["chunk_id"] for rec in self.records]
        duplicates = [cid for cid in set(ids) if ids.count(cid) > 1]
        assert not duplicates, f"Duplicate chunk_ids: {duplicates}"

    def test_source_paths_exist(self):
        """source_path のファイルが存在するか確認"""
        for i, rec in enumerate(self.records):
            source = rec.get("source_path", "")
            if not source:
                continue
            full_path = ROOT / source
            assert full_path.exists(), f"Record {i}: source_path not found: {source}"

    def test_no_backslash_in_source_path(self):
        for i, rec in enumerate(self.records):
            source = rec.get("source_path", "")
            assert "\\" not in source, f"Record {i} has backslash in source_path: {source}"


class TestChunkFiles:
    """data/chunk/ ディレクトリのテスト"""

    def test_chunk_dir_exists(self):
        chunk_dir = ROOT / "data" / "chunk"
        assert chunk_dir.is_dir(), "data/chunk directory not found"

    def test_all_md_files_have_front_matter(self):
        chunk_dir = ROOT / "data" / "chunk"
        if not chunk_dir.exists():
            pytest.skip("data/chunk not found")
        for md_file in sorted(chunk_dir.glob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            assert content.strip().startswith("---"), (
                f"{md_file.name} does not start with YAML front matter"
            )

    def test_all_md_files_have_title(self):
        """front matter に title フィールドがあるか"""
        chunk_dir = ROOT / "data" / "chunk"
        if not chunk_dir.exists():
            pytest.skip("data/chunk not found")

        sys.path.insert(0, str(ROOT / "scripts"))
        from rag_build_jsonl import parse_markdown_with_front_matter

        for md_file in sorted(chunk_dir.glob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            meta, _ = parse_markdown_with_front_matter(content)
            assert meta.get("title"), f"{md_file.name} missing 'title' in front matter"

    def test_all_chunk_files_referenced_in_jsonl(self):
        """chunk/ 配下の全ファイルが chunks.jsonl に参照されているか"""
        chunk_dir = ROOT / "data" / "chunk"
        jsonl_path = ROOT / "data" / "chunks.jsonl"
        if not chunk_dir.exists() or not jsonl_path.exists():
            pytest.skip("Required data not found")

        md_files = {f.name for f in chunk_dir.glob("*.md")}

        sources = set()
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                source = rec.get("source_path", "")
                if "/" in source:
                    sources.add(source.split("/")[-1])

        not_referenced = md_files - sources
        assert not not_referenced, f"Chunk files not in chunks.jsonl: {not_referenced}"


class TestEvalData:
    """eval_sample.csv / eval_tuning.csv のテスト"""

    def _load_csv(self, path: Path) -> list[dict]:
        if not path.exists():
            pytest.skip(f"{path.name} not found")
        with path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_eval_sample_not_empty(self):
        rows = self._load_csv(ROOT / "data" / "eval_sample.csv")
        assert len(rows) > 0

    def test_eval_sample_has_required_cols(self):
        rows = self._load_csv(ROOT / "data" / "eval_sample.csv")
        for row in rows:
            assert "question" in row or "query" in row, f"Missing question/query: {row.keys()}"

    def test_eval_tuning_not_empty(self):
        rows = self._load_csv(ROOT / "data" / "eval_tuning.csv")
        assert len(rows) > 0

    def test_eval_tuning_source_paths_exist(self):
        """eval_tuning.csv の expected_source パスが存在するか"""
        rows = self._load_csv(ROOT / "data" / "eval_tuning.csv")
        for i, row in enumerate(rows):
            source = row.get("expected_source", "").strip()
            if not source:
                continue
            full = ROOT / source
            assert full.exists(), f"eval_tuning row {i}: source not found: {source}"


class TestPromptFiles:
    """data/prompts/ のテスト"""

    def test_prompts_dir_exists(self):
        prompts_dir = ROOT / "data" / "prompts"
        assert prompts_dir.is_dir(), "data/prompts directory not found"

    def test_expected_prompt_files_exist(self):
        expected = ["general.txt", "strict.txt", "hybrid.txt"]
        prompts_dir = ROOT / "data" / "prompts"
        if not prompts_dir.exists():
            pytest.skip("data/prompts not found")
        for name in expected:
            assert (prompts_dir / name).exists(), f"Missing prompt file: {name}"

    def test_prompt_files_not_empty(self):
        prompts_dir = ROOT / "data" / "prompts"
        if not prompts_dir.exists():
            pytest.skip("data/prompts not found")
        for txt_file in prompts_dir.glob("*.txt"):
            content = txt_file.read_text(encoding="utf-8")
            assert content.strip(), f"Prompt file is empty: {txt_file.name}"
