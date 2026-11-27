"""對 dev set 跑檢索，輸出 TREC run 檔：uv run python scripts/run_retrieval.py

預設跑 bm25（正式版）與 baseline（單字切分、無詞庫），分別寫到
eval/runs/{preset}.trec。算指標用 ④ 的評估腳本，例如：
    uv run python -m regrag.evaluation.run_eval --run eval/runs/bm25.trec
"""

import argparse

from regrag.ingest import load_chunks
from regrag.retrieval.presets import PRESET_NAMES, build
from regrag.retrieval.runfile import load_queries, run_queries, write_run


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--queries", default="eval/dev.jsonl")
    ap.add_argument("--out-dir", default="eval/runs")
    ap.add_argument("--preset", nargs="+", default=list(PRESET_NAMES), choices=PRESET_NAMES)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--include-unanswerable", action="store_true", help="無解題也輸出（預設不輸出）"
    )
    args = ap.parse_args(argv)

    chunks = load_chunks(args.chunks)
    queries = load_queries(args.queries)
    for name in args.preset:
        lines = run_queries(build(name, chunks), queries, name, args.k, args.include_unanswerable)
        path = write_run(lines, f"{args.out_dir}/{name}.trec")
        print(f"{name}: {len(lines)} 行 → {path}")


if __name__ == "__main__":
    main()
