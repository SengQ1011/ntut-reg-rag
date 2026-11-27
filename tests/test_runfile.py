"""② TREC run 檔輸出與兩個腳本。只用 tests/fixtures 的假資料，輸出寫到 tmp_path。"""

import importlib.util
import json
from pathlib import Path

import pytest

from regrag.ingest import load_chunks
from regrag.retrieval import presets, runfile

ROOT = Path(__file__).parent.parent
FIX = Path(__file__).parent / "fixtures"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def retriever():
    return presets.build("bm25", load_chunks(FIX / "chunks.jsonl"))


@pytest.fixture
def queries():
    return runfile.load_queries(FIX / "dev.jsonl")


def test_trec_line_format(retriever):
    lines = runfile.trec_lines("dev-x", retriever.search("加退選", k=1), "bm25")
    qid, q0, chunk_id, rank, score, tag = lines[0].split()
    assert (qid, q0, chunk_id, rank, tag) == ("dev-x", "Q0", "fake-xuanke#art3", "1", "bm25")
    assert float(score) > 0


def test_tag_with_space_rejected(retriever):
    with pytest.raises(ValueError):
        runfile.trec_lines("q", retriever.search("學生"), "bm 25")


def test_unanswerable_skipped_by_default(retriever, queries):
    qids = {line.split()[0] for line in runfile.run_queries(retriever, queries, "t")}
    assert qids == {"dev-001"}  # dev-002 是無解題


def test_unanswerable_included_on_request(retriever, queries):
    lines = runfile.run_queries(retriever, queries, "t", include_unanswerable=True)
    # 無解題若一條都沒命中就不會有行；有命中就會輸出
    assert {line.split()[0] for line in lines} >= {"dev-001"}


def test_ranks_are_consecutive(retriever, queries):
    lines = runfile.run_queries(retriever, queries, "t", k=10)
    assert [int(line.split()[3]) for line in lines] == list(range(1, len(lines) + 1))


def test_build_index_script(tmp_path):
    _script("build_index").main(
        ["--chunks", str(FIX / "chunks.jsonl"), "--index-dir", str(tmp_path)]
    )
    for name in presets.PRESET_NAMES:
        data = json.loads((tmp_path / f"{name}.json").read_text(encoding="utf-8"))
        assert data["preset"] == name
        assert data["chunk_ids"] == ["fake-xueze#art1", "fake-xuanke#art3"]


def test_run_retrieval_script(tmp_path):
    _script("run_retrieval").main(
        [
            "--chunks",
            str(FIX / "chunks.jsonl"),
            "--queries",
            str(FIX / "dev.jsonl"),
            "--out-dir",
            str(tmp_path),
        ]
    )
    for name in presets.PRESET_NAMES:
        lines = (tmp_path / f"{name}.trec").read_text(encoding="utf-8").splitlines()
        assert lines and all(line.split()[0] == "dev-001" for line in lines)
        assert all(line.endswith(f" {name}") for line in lines)
