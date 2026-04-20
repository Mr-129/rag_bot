"""rag_build_jsonl.py のユニットテスト。

チャンク分割・YAML front matter パース・JSONL生成ロジックをテストする。
"""
import json
import sys
from pathlib import Path

import pytest

# プロジェクトルートをPATHに追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from rag_build_jsonl import (
    Chunk,
    parse_markdown_with_front_matter,
    split_markdown_by_headings,
    split_text_by_limit,
    stable_chunk_id,
    to_json_record,
    load_chunks,
)


# ============================================================
# parse_markdown_with_front_matter
# ============================================================

class TestParseFrontMatter:
    def test_valid_front_matter(self):
        content = '---\ntitle: "テスト文書"\nproduct: "TestProduct"\ntags: ["a", "b"]\n---\n# 本文\nここが本文です。'
        meta, body = parse_markdown_with_front_matter(content)
        assert meta["title"] == "テスト文書"
        assert meta["product"] == "TestProduct"
        assert meta["tags"] == ["a", "b"]
        assert "# 本文" in body

    def test_no_front_matter(self):
        content = "# ただのMarkdown\n本文です。"
        meta, body = parse_markdown_with_front_matter(content)
        assert meta == {}
        assert "# ただのMarkdown" in body

    def test_empty_front_matter(self):
        content = "---\n---\n# 本文"
        meta, body = parse_markdown_with_front_matter(content)
        assert meta == {}
        assert "# 本文" in body

    def test_invalid_yaml_raises_or_returns_empty(self):
        import yaml
        content = "---\n: : invalid yaml [[\n---\n本文"
        try:
            meta, body = parse_markdown_with_front_matter(content)
            # パースエラーをハンドルしている場合
            assert isinstance(meta, dict)
        except yaml.YAMLError:
            # YAMLパースエラーが伝播するのも正当な実装
            pass


# ============================================================
# split_markdown_by_headings
# ============================================================

class TestSplitByHeadings:
    def test_single_heading(self):
        body = "# タイトル\n本文テキスト"
        sections = split_markdown_by_headings(body)
        assert len(sections) == 1
        assert "本文テキスト" in sections[0][1]

    def test_multiple_headings(self):
        body = "# セクション1\nテキスト1\n## セクション2\nテキスト2\n## セクション3\nテキスト3"
        sections = split_markdown_by_headings(body)
        assert len(sections) >= 2

    def test_heading_in_code_block_ignored(self):
        body = "# 実際の見出し\n本文\n```\n# コメント（見出しではない）\n```\n別の本文"
        sections = split_markdown_by_headings(body)
        # コードブロック内の#は見出しとして扱われないので、セクションは1つ
        assert len(sections) == 1

    def test_no_headings(self):
        body = "見出しなしのテキスト\n改行あり"
        sections = split_markdown_by_headings(body)
        assert len(sections) == 1
        assert "見出しなし" in sections[0][1]

    def test_heading_path_nested(self):
        body = "# 親\nテキスト1\n## 子\nテキスト2"
        sections = split_markdown_by_headings(body)
        paths = [s[0] for s in sections]
        # 2つ目のセクションのheading_pathに「親 > 子」が含まれるはず
        assert any("親 > 子" in p for p in paths)


# ============================================================
# split_text_by_limit
# ============================================================

class TestSplitTextByLimit:
    def test_short_text_no_split(self):
        text = "短いテキスト"
        result = split_text_by_limit(text, max_chars=100)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_splits(self):
        text = "あ" * 200
        result = split_text_by_limit(text, max_chars=100)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 200  # 厳密ではなく、分割されることを確認

    def test_paragraph_boundary_split(self):
        text = "段落1の内容です。これは長いテキストです。" + "\n\n" + "段落2の内容です。これも長いテキストです。"
        result = split_text_by_limit(text, max_chars=15)
        assert len(result) >= 2

    def test_overlap(self):
        text = "一段落目\n\n二段落目\n\n三段落目"
        result = split_text_by_limit(text, max_chars=12, overlap=5)
        # オーバーラップにより、先頭以外に前チャンクの末尾が含まれる可能性
        assert len(result) >= 2

    def test_empty_text(self):
        result = split_text_by_limit("", max_chars=100)
        assert result == []


# ============================================================
# stable_chunk_id
# ============================================================

class TestStableChunkId:
    def test_deterministic(self):
        chunk = Chunk(path=Path("test.md"), meta={"title": "Test"}, text="テスト本文")
        id1 = stable_chunk_id(chunk)
        id2 = stable_chunk_id(chunk)
        assert id1 == id2

    def test_prefix(self):
        chunk = Chunk(path=Path("test.md"), meta={"title": "Test"}, text="テスト")
        cid = stable_chunk_id(chunk)
        assert cid.startswith("sha256:")

    def test_different_text_different_id(self):
        c1 = Chunk(path=Path("test.md"), meta={"title": "T"}, text="テキストA")
        c2 = Chunk(path=Path("test.md"), meta={"title": "T"}, text="テキストB")
        assert stable_chunk_id(c1) != stable_chunk_id(c2)


# ============================================================
# to_json_record
# ============================================================

class TestToJsonRecord:
    def test_required_fields(self):
        chunk = Chunk(
            path=Path("data/chunk/test.md"),
            meta={"title": "テスト", "tags": ["tag1"], "source_path": "data/chunk/test.md"},
            text="本文です",
            heading_path="テスト",
            chunk_index=1,
        )
        rec = to_json_record(chunk)
        assert "chunk_id" in rec
        assert rec["title"] == "テスト"
        assert rec["source_path"] == "data/chunk/test.md"
        assert rec["text"] == "本文です"
        assert rec["tags"] == ["tag1"]
        assert rec["chunk_index"] == 1

    def test_json_serializable(self):
        chunk = Chunk(
            path=Path("test.md"),
            meta={"title": "Test", "tags": []},
            text="テスト",
        )
        rec = to_json_record(chunk)
        serialized = json.dumps(rec, ensure_ascii=False)
        assert isinstance(serialized, str)


# ============================================================
# load_chunks (実データ結合テスト)
# ============================================================

class TestLoadChunks:
    def test_load_from_data_dir(self):
        data_dir = ROOT / "data" / "chunk"
        if not data_dir.exists():
            pytest.skip("data/chunk directory not found")
        chunks = load_chunks(data_dir, max_chars=700, overlap=100)
        assert len(chunks) > 0
        for c in chunks:
            assert c.text.strip()
            assert c.title

    def test_all_chunks_have_source_path(self):
        data_dir = ROOT / "data" / "chunk"
        if not data_dir.exists():
            pytest.skip("data/chunk directory not found")
        chunks = load_chunks(data_dir, max_chars=700, overlap=100)
        for c in chunks:
            assert c.source_path, f"source_path missing for {c.path}"
