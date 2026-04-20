from __future__ import annotations

"""Build JSONL from Markdown/TXT chunks (minimal, learning-first).

- 入力: --data-dir 配下の .md / .txt
- 出力: 1行=1チャンクのJSONL

前提
- .md は先頭の YAML front matter（--- ... ---）があればメタデータとして読む
- .txt はメタ無し（ファイル名などから最低限を補う）

出力フィールド（最小）
- chunk_id: 安定ID（sha256）
- title: タイトル
- source_path: 出典（front matterがあればそれを優先）
- tags: タグ（あれば）
- text: 本文（front matter除外後）

メタが存在する場合は、product/topic/doc_type/os/created も一緒に出力します。
"""

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
DEFAULT_MAX_CHARS = 700


@dataclass(frozen=True)
class Chunk:
    """検索用に扱う1チャンクのデータ構造。"""

    path: Path
    meta: dict[str, Any]
    text: str
    heading_path: str = ""
    chunk_index: int = 1

    @property
    def title(self) -> str:
        return str(self.meta.get("title") or self.path.stem)

    @property
    def source_path(self) -> str:
        raw = self.meta.get("source_path")
        if isinstance(raw, str) and raw.strip():
            return raw.strip().replace("\\", "/")
        try:
            return self.path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return self.path.as_posix()


def parse_markdown_with_front_matter(content: str) -> tuple[dict[str, Any], str]:
    """Markdown先頭の front matter を (meta, body) に分離する。"""
    match = FRONT_MATTER_RE.match(content.strip() + "\n")
    if not match:
        return {}, content

    raw_yaml, body = match.group(1), match.group(2)
    meta = yaml.safe_load(raw_yaml) or {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, body


def split_markdown_by_headings(body: str) -> list[tuple[str, str]]:
    """Markdown本文を見出し単位に分割する。

    注意:
    - コードブロック内（``` / ~~~）の # は見出しとして扱わない
    - 見出し1行だけの空チャンクは除外する
    """
    lines = body.splitlines()
    sections: list[tuple[str, str]] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_heading_path = ""
    in_fenced_code = False

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        if not text:
            return
        text_lines = text.splitlines()
        # 「見出し1行だけ」のチャンクを作らない（例: "# タイトル" のみ）
        if len(text_lines) == 1 and HEADING_RE.match(text_lines[0]):
            return
        if text:
            sections.append((current_heading_path, text))

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fenced_code = not in_fenced_code

        match = HEADING_RE.match(line) if not in_fenced_code else None
        if match:
            flush()
            current_lines = []

            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            current_heading_path = " > ".join(heading_stack)

        current_lines.append(line)

    flush()

    if not sections:
        raw = body.strip()
        return [("", raw)] if raw else []
    return sections


def split_text_by_limit(text: str, max_chars: int, overlap: int = 0) -> list[str]:
    """文字数上限で再分割する。

    優先:
    1) 段落境界（空行）
    2) 長すぎる場合は改行位置または強制分割

    overlap > 0 の場合、前チャンクの末尾 overlap 文字を次チャンクの先頭に重複させる。
    これにより分割境界での情報損失を防ぐ。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    out: list[str] = []
    current = ""

    def push_hard_split(long_text: str) -> None:
        rest = long_text.strip()
        while len(rest) > max_chars:
            cut = rest.rfind("\n", 0, max_chars + 1)
            if cut < int(max_chars * 0.5):
                cut = max_chars
            part = rest[:cut].strip()
            if part:
                out.append(part)
            rest = rest[cut:].strip()
        if rest:
            out.append(rest)

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                out.append(current)
                current = ""
            push_hard_split(paragraph)
            continue

        if not current:
            current = paragraph
            continue

        candidate = f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            out.append(current)
            current = paragraph

    if current:
        out.append(current)

    # オーバーラップ: 前チャンク末尾を次チャンク先頭に付加
    if overlap > 0 and len(out) > 1:
        overlapped = [out[0]]
        for i in range(1, len(out)):
            prev = out[i - 1]
            tail = prev[-overlap:]
            # 行境界に揃えて中途半端な切断を避ける
            nl = tail.find("\n")
            if nl >= 0:
                tail = tail[nl + 1 :]
            tail = tail.strip()
            if tail:
                overlapped.append(tail + "\n\n" + out[i])
            else:
                overlapped.append(out[i])
        return overlapped

    return out


DEFAULT_OVERLAP = 100


def load_chunks(data_dir: Path, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP) -> list[Chunk]:
    """data_dir配下の .md/.txt を読み込み、チャンク化して返す。"""
    paths = sorted([*data_dir.glob("*.md"), *data_dir.glob("*.txt")])
    chunks: list[Chunk] = []

    for path in paths:
        content = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() == ".md":
            meta, body = parse_markdown_with_front_matter(content)
            units = split_markdown_by_headings(body)
        else:
            meta, body = {}, content
            raw = body.strip()
            units = [("", raw)] if raw else []

        chunk_index = 1
        for heading_path, section_text in units:
            for split_text in split_text_by_limit(section_text, max_chars=max_chars, overlap=overlap):
                chunks.append(
                    Chunk(
                        path=path,
                        meta=meta,
                        text=split_text,
                        heading_path=heading_path,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

    return chunks


def stable_chunk_id(chunk: Chunk) -> str:
    # 学習優先の最小構成: 内容が変わればIDも変わる（差分検知がしやすい）
    # ※将来的に「source_path + chunk_index」で安定化したくなったらここを変える
    payload = "\n".join([chunk.source_path, chunk.heading_path, chunk.title, chunk.text]).encode("utf-8", errors="ignore")
    digest = hashlib.sha256(payload).hexdigest()
    return f"sha256:{digest}"


def to_json_record(chunk: Chunk) -> dict[str, Any]:
    """ChunkをJSONL出力用dictへ変換する。"""
    tags = chunk.meta.get("tags") if "tags" in chunk.meta else []
    if isinstance(tags, str):
        tags = [tags]
    elif not isinstance(tags, list):
        tags = []

    related_ids_raw = chunk.meta.get("related_ids") if "related_ids" in chunk.meta else []
    if isinstance(related_ids_raw, str):
        related_ids = [related_ids_raw]
    elif isinstance(related_ids_raw, list):
        related_ids = [str(value) for value in related_ids_raw if str(value).strip()]
    else:
        related_ids = []

    record: dict[str, Any] = {
        "chunk_id": stable_chunk_id(chunk),
        "title": chunk.title,
        "source_path": chunk.source_path,
        "tags": tags,
        "text": chunk.text,
    }
    if related_ids:
        record["related_ids"] = related_ids
    if chunk.heading_path:
        record["heading_path"] = chunk.heading_path
    record["chunk_index"] = chunk.chunk_index

    created_value = chunk.meta.get("created_at") or chunk.meta.get("created")
    if created_value is not None:
        # 新運用は created_at を正本とし、旧利用側向けに created も残す。
        record["created_at"] = created_value
        record["created"] = created_value

    updated_value = chunk.meta.get("updated_at") or chunk.meta.get("updated")
    if updated_value is not None:
        record["updated_at"] = updated_value

    # 任意メタ（あれば）
    for key in ["product", "topic", "doc_type", "os"]:
        if key in chunk.meta:
            record[key] = chunk.meta.get(key)

    return record


def main() -> int:
    """CLIエントリーポイント。

    例:
    python rag_build_jsonl.py --data-dir data/chunk --out data/chunks.jsonl --max-chars 700
    """
    parser = argparse.ArgumentParser(description="Build JSONL from chunk files.")
    parser.add_argument("--data-dir", default="data/chunk", help="Directory containing chunk files")
    parser.add_argument("--out", default="data/chunks.jsonl", help="Output JSONL path")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Max characters per chunk")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Overlap characters between consecutive chunks (default: 100)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"data-dir not found: {data_dir}")

    if args.max_chars <= 0:
        raise SystemExit("--max-chars must be > 0")

    if args.overlap < 0:
        raise SystemExit("--overlap must be >= 0")

    chunks = load_chunks(data_dir, max_chars=args.max_chars, overlap=args.overlap)
    if not chunks:
        raise SystemExit(f"No chunks found under: {data_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for chunk in chunks:
            record = to_json_record(chunk)
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    print(f"Wrote {len(chunks)} records -> {out_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
