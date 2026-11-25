"""generate：檢索結果 → prompt → LLM → Answer。"""

import logging
import time

from regrag.llm.client import LLMClient
from regrag.llm.parsing import LLMOutputError, parse_llm_output
from regrag.llm.prompt import build_prompt
from regrag.schemas import Answer, Chunk, Citation, Hit, RetrievalResult

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = "目前收錄的校規中查無相關規定，建議直接洽詢承辦單位。"


def generate(retrieval: RetrievalResult, llm: LLMClient, started_at: float) -> Answer:
    """把檢索結果全部交給 LLM 回答，組成通過合約驗證的 Answer。

    取幾筆由 answer() 呼叫 search(k=...) 決定（合約保證 hits 最多 k 筆），這裡不再切。
    問題一律用 retrieval.query（② 原樣回傳呼叫端的查詢），避免 prompt 和 Answer 記的不一致。

    - 沒有任何命中：直接拒答，不呼叫 LLM。
    - LLM 回 refused：拒答，忽略它附的引用。
    - 不合法的引用編號丟掉並記 warning；一筆合法引用都沒有 → LLMOutputError。
    - LLM 輸出解析失敗 → LLMOutputError（系統錯誤，不可當成拒答）。

    started_at：answer() 開始時的 time.perf_counter()。合約規定 latency_ms 量 answer() 端到端。
    """
    hits = retrieval.hits
    if not hits:
        return _build_answer(retrieval, llm.model, started_at, REFUSAL_MESSAGE, [])

    raw = llm.complete(build_prompt(retrieval.query, hits))
    output = parse_llm_output(raw)
    if output.refused:
        return _build_answer(retrieval, llm.model, started_at, REFUSAL_MESSAGE, [])

    citations = _resolve_citations(output.citations, hits)
    if not citations:
        raise LLMOutputError("LLM 回答了，但沒有任何合法的引用編號", raw)
    return _build_answer(retrieval, llm.model, started_at, output.answer, citations)


def _build_answer(
    retrieval: RetrievalResult,
    model: str,
    started_at: float,
    text: str,
    citations: list[Citation],
) -> Answer:
    """沒有引用就是拒答，和合約「refused 若且唯若 citations 為空」一致。"""
    return Answer(
        query=retrieval.query,
        answer=text,
        citations=citations,
        refused=not citations,
        model=model,
        retrieval=retrieval,
        latency_ms=(time.perf_counter() - started_at) * 1000,
    )


def _resolve_citations(numbers: list[int], hits: list[Hit]) -> list[Citation]:
    """編號 1..len(hits) 對回 hit.chunk；不合法的丟掉並記錄，重複的只留第一次。"""
    allowed = range(1, len(hits) + 1)
    invalid = [n for n in numbers if n not in allowed]
    if invalid:
        logger.warning("LLM 引用了不合法的編號 %s（合法範圍 1–%d），已丟掉", invalid, len(hits))
    valid = dict.fromkeys(n for n in numbers if n in allowed)
    return [_to_citation(hits[n - 1].chunk) for n in valid]


def _to_citation(chunk: Chunk) -> Citation:
    return Citation(
        chunk_id=chunk.chunk_id,
        doc_title=chunk.doc_title,
        article_no=chunk.article_no,
        source_url=chunk.source_url,
    )
