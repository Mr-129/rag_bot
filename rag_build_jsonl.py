"""Compatibility wrapper: call the real script under `scripts/`.

このファイルは互換性維持のための薄いラッパーです。
古い運用で `python rag_build_jsonl.py` を使っている場合に、
内部で `scripts/rag_build_jsonl.py` を実行します。
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "rag_build_jsonl.py"
    runpy.run_path(str(target), run_name="__main__")
