"""確認假資料符合合約，合約改壞時 CI 會紅燈。"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from regrag.ingest import load_chunks
from regrag.schemas import Chunk, DevQuery

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
