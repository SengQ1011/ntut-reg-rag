"""確認假資料符合合約，合約改壞時 CI 會紅燈。"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from regrag.ingest import load_chunks
from regrag.schemas import Answer, Chunk, Citation, DevQuery, Hit, RetrievalResult

FIX = Path(__file__).parent / "fixtures"


def test_fixture_chunks_validate():
    chunks = load_chunks(FIX / "chunks.jsonl")
    assert len(chunks) == 2
    assert all(c.doc_id.startswith("fake-") for c in chunks)


def test_chunk_id_format():
    for c in load_chunks(FIX / "chunks.jsonl"):
        assert c.chunk_id.startswith(f"{c.doc_id}#art")


def test_fixture_dev_queries_validate():
    lines = (FIX / "dev.jsonl").read_text(encoding="utf-8").splitlines()
    qs = [DevQuery.model_validate(json.loads(line)) for line in lines if line.strip()]
    assert {q.qid for q in qs} == {"dev-001", "dev-002"}


def test_qrels_point_to_existing_chunks():
    chunk_ids = {c.chunk_id for c in load_chunks(FIX / "chunks.jsonl")}
    for line in (FIX / "dev.qrels").read_text(encoding="utf-8").splitlines():
        qid, _, chunk_id, grade = line.split()
        assert chunk_id in chunk_ids
        assert int(grade) >= 0


def test_bad_chunk_rejected():
    with pytest.raises(ValidationError):
        Chunk.model_validate({"chunk_id": "x"})


def test_dev_unanswerable_has_no_qrels():
    lines = (FIX / "dev.jsonl").read_text(encoding="utf-8").splitlines()
    qs = [DevQuery.model_validate(json.loads(line)) for line in lines if line.strip()]
    unanswerable = {q.qid for q in qs if not q.answerable}
    qrel_qids = {
        line.split()[0] for line in (FIX / "dev.qrels").read_text(encoding="utf-8").splitlines()
    }
    assert unanswerable.isdisjoint(qrel_qids)


def test_dev_qtype_must_match_answerable():
    with pytest.raises(ValidationError):
        DevQuery(qid="x", query="q", author="a", source="self", qtype="direct", answerable=False)
    with pytest.raises(ValidationError):
        DevQuery(
            qid="x", query="q", author="a", source="self", qtype="unanswerable", answerable=True
        )


def test_unknown_field_rejected():
    """拼錯的欄位要報錯，不可默默丟掉。"""
    with pytest.raises(ValidationError):
        DevQuery(
            qid="x",
            query="q",
            author="a",
            source="self",
            qtype="direct",
            answerable=True,
            notes="拼錯",
        )


def _retrieval() -> RetrievalResult:
    chunk = load_chunks(FIX / "chunks.jsonl")[0]
    hit = Hit(chunk_id=chunk.chunk_id, score=1.0, rank=1, chunk=chunk)
    return RetrievalResult(query="q", retriever="bm25", hits=[hit], latency_ms=1.0)


def _citation() -> Citation:
    c = _retrieval().hits[0].chunk
    return Citation(
        chunk_id=c.chunk_id, doc_title=c.doc_title, article_no=c.article_no, source_url=c.source_url
    )


def _answer(refused: bool, citations: list[Citation]) -> Answer:
    return Answer(
        query="q",
        answer="a",
        citations=citations,
        refused=refused,
        model="fake",
        retrieval=_retrieval(),
        latency_ms=2.0,
    )


def test_answer_valid_cases():
    assert _answer(refused=False, citations=[_citation()]).citations
    assert _answer(refused=True, citations=[]).refused


def test_refused_answer_must_have_no_citations():
    with pytest.raises(ValidationError):
        _answer(refused=True, citations=[_citation()])


def test_non_refused_answer_must_cite():
    with pytest.raises(ValidationError):
        _answer(refused=False, citations=[])
