"""重建 BM25 索引：uv run python scripts/build_index.py

讀 data/chunks.jsonl（或 --chunks 指定），依 preset 建索引，寫到 index/{preset}.json。
search() 啟動時若 index/ 裡的索引和目前 chunks、詞庫一致就直接載入，否則現場重建。
"""

import argparse
import time

from regrag.ingest import load_chunks
from regrag.retrieval.bm25 import DEFAULT_INDEX_DIR
from regrag.retrieval.presets import PRESET_NAMES, build, save_index


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    ap.add_argument("--preset", nargs="+", default=list(PRESET_NAMES), choices=PRESET_NAMES)
    args = ap.parse_args(argv)

    chunks = load_chunks(args.chunks)
    for name in args.preset:
        start = time.perf_counter()
        path = save_index(build(name, chunks), name, args.index_dir)
        ms = (time.perf_counter() - start) * 1000
        print(f"{name}: {len(chunks)} chunks → {path}（{ms:.0f} ms）")


if __name__ == "__main__":
    main()
