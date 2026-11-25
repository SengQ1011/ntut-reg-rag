"""generate：檢索結果 → prompt → LLM → Answer。"""

import logging
import time

from regrag.llm.client import LLMClient
from regrag.llm.parsing import LLMOutputError, parse_llm_output
from regrag.llm.prompt import build_prompt
from regrag.schemas import Answer, Chunk, Citation, Hit, RetrievalResult

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = "目前收錄的校規中查無相關規定，建議直接洽詢承辦單位。"


def generate(
    retrieval: RetrievalResult,
    llm: LLMClient,
    k: int = 5,
    started_at: float | None = None,
) -> Answer:
    """取檢索結果前 k 筆給 LLM 回答，組成通過合約驗證的 Answer。

    問題一律用 retrieval.query（② 原樣回傳呼叫端的查詢），不另外收 query，
    避免 prompt 用的問題和 Answer.retrieval 記下的不一致。

    - 沒有任何命中：直接拒答，不呼叫 LLM。
    - LLM 回 refused：拒答，忽略它附的引用。
    - 不合法的引用編號丟掉並記 warning；一筆合法引用都沒有 → LLMOutputError。
    - LLM 輸出解析失敗 → LLMOutputError（系統錯誤，不可當成拒答）。

    started_at：answer() 開始時的 time.perf_counter()。合約規定 Answer.latency_ms
    量 answer() 端到端，所以由 answer() 傳入；沒傳就從本函式開始算。
    """
    if k < 1:
        raise ValueError(f"k 必須 >= 1，收到 {k}")
    start = time.perf_counter() if started_at is None else started_at
    query = retrieval.query
    hits = retrieval.hits[:k]
    if not hits:
        return _refusal(retrieval, llm.model, start)

    raw = llm.complete(build_prompt(query, hits))
    output = parse_llm_output(raw)
    if output.refused:
        return _refusal(retrieval, llm.model, start)

    citations = _resolve_citations(output.citations, hits)
    if not citations:
        raise LLMOutputError("LLM 回答了，但沒有任何合法的引用編號", raw)
    return Answer(
        query=query,
        answer=output.answer,
        citations=citations,
        refused=False,
        model=llm.model,
        retrieval=retrieval,
        latency_ms=_elapsed_ms(start),
    )


def _refusal(retrieval: RetrievalResult, model: str, start: float) -> Answer:
    return Answer(
        query=retrieval.query,
        answer=REFUSAL_MESSAGE,
        citations=[],
        refused=True,
        model=model,
        retrieval=retrieval,
        latency_ms=_elapsed_ms(start),
    )


def _resolve_citations(numbers: list[int], hits: list[Hit]) -> list[Citation]:
    """編號 1..len(hits) 對回 hit.chunk；不合法的丟掉並記錄，重複的只留第一次。"""
    invalid = [n for n in numbers if not 1 <= n <= len(hits)]
    if invalid:
        logger.warning("LLM 引用了不合法的編號 %s（合法範圍 1–%d），已丟掉", invalid, len(hits))
    valid = dict.fromkeys(n for n in numbers if 1 <= n <= len(hits))
    return [_to_citation(hits[n - 1].chunk) for n in valid]


def _to_citation(chunk: Chunk) -> Citation:
    return Citation(
        chunk_id=chunk.chunk_id,
        doc_title=chunk.doc_title,
        article_no=chunk.article_no,
        source_url=chunk.source_url,
    )


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
