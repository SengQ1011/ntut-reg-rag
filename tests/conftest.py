"""共用的測試資料：用 fixtures/chunks.jsonl 的假條文組出假的檢索結果。"""

from collections.abc import Callable
from pathlib import Path

import pytest

from regrag.ingest import load_chunks
from regrag.schemas import Chunk, Hit, RetrievalResult

FIX = Path(__file__).parent / "fixtures"

MakeRetrieval = Callable[[str, list[Chunk]], RetrievalResult]


def _build_retrieval(query: str, chunks: list[Chunk]) -> RetrievalResult:
    """依 chunks 的順序組出 RetrievalResult，rank 從 1 開始、分數遞減。"""
    hits = [
        Hit(chunk_id=c.chunk_id, score=float(len(chunks) - i), rank=i + 1, chunk=c)
        for i, c in enumerate(chunks)
    ]
    return RetrievalResult(query=query, retriever="bm25", hits=hits, latency_ms=1.0)


@pytest.fixture
def make_retrieval() -> MakeRetrieval:
    """工廠 fixture：測試需要自訂 chunks 時用它組 RetrievalResult（不要直接 import conftest）。"""
    return _build_retrieval


@pytest.fixture
def fake_chunks() -> list[Chunk]:
    return load_chunks(FIX / "chunks.jsonl")


@pytest.fixture
def retrieval(fake_chunks: list[Chunk]) -> RetrievalResult:
    """兩筆命中：[1] fake-xueze#art1、[2] fake-xuanke#art3。"""
    return _build_retrieval("成績不及格會怎樣？", fake_chunks)


@pytest.fixture
def empty_retrieval() -> RetrievalResult:
    return _build_retrieval("宿舍可以養貓嗎？", [])
