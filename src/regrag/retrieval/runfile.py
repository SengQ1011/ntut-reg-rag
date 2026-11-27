"""② 對 dev set 跑檢索，輸出 TREC run 檔給 ④ 的評估腳本（eval/runs/*.trec）。

TREC run 格式，一行一筆：qid Q0 chunk_id rank score tag
指標（Recall@5、MRR、nDCG@10）由 ④ 的 run_eval 計算，這裡只負責產生 run 檔。
"""

import json
from collections.abc import Iterable
from pathlib import Path

from regrag.retrieval.bm25 import BM25Retriever
from regrag.schemas import DevQuery, RetrievalResult


def load_queries(path: str | Path) -> list[DevQuery]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [DevQuery.model_validate(json.loads(line)) for line in lines if line.strip()]


def trec_lines(qid: str, result: RetrievalResult, tag: str) -> list[str]:
    if any(c.isspace() for c in tag):
        raise ValueError("run tag 不可含空白")
    return [f"{qid} Q0 {h.chunk_id} {h.rank} {h.score:.6f} {tag}" for h in result.hits]


def run_queries(
    retriever: BM25Retriever,
    queries: Iterable[DevQuery],
    tag: str,
    k: int = 10,
    include_unanswerable: bool = False,
) -> list[str]:
    """無解題預設不輸出：它們沒有 qrels（見 eval/guidelines.md），拒答率另外算。"""
    lines: list[str] = []
    for q in queries:
        if q.answerable or include_unanswerable:
            lines.extend(trec_lines(q.qid, retriever.search(q.query, k), tag))
    return lines


def write_run(lines: list[str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path
