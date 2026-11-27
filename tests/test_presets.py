"""② preset 與索引存取。只用 tests/fixtures 的假資料，索引寫到 tmp_path。"""

from pathlib import Path

import pytest

from regrag.ingest import load_chunks
from regrag.retrieval import bm25, presets
from regrag.retrieval.bm25 import char_tokenize

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def chunks():
    return load_chunks(FIX / "chunks.jsonl")


def _ids(retriever, query):
    return [(h.chunk_id, round(h.score, 9)) for h in retriever.search(query, k=10).hits]


def test_baseline_is_char_without_lexicon(chunks):
    r = presets.build("baseline", chunks)
    assert r.tokenizer is char_tokenize
    assert r.query_expander is None


def test_bm25_preset_uses_userdict(chunks):
    r = presets.build("bm25", chunks)
    assert "加退選" in r.tokenizer("加退選")  # 校務詞整個成為一個詞
    assert r.query_expander is not None
    assert "不及格" in r.query_expander("被當")


def test_unknown_preset_rejected(chunks):
    with pytest.raises(ValueError):
        presets.build("dense", chunks)


@pytest.mark.parametrize("name", presets.PRESET_NAMES)
def test_save_load_roundtrip(chunks, tmp_path, name):
    built = presets.build(name, chunks)
    presets.save_index(built, name, tmp_path)
    index = presets.load_index(name, chunks, tmp_path)
    assert index == built.index
    loaded = presets.load_or_build(name, chunks, tmp_path)
    for q in ["加退選", "被當一半", "學生"]:
        assert _ids(loaded, q) == _ids(built, q)


def test_stale_index_not_loaded(chunks, tmp_path):
    presets.save_index(presets.build("bm25", chunks), "bm25", tmp_path)
    changed = [chunks[0].model_copy(update={"text": "改過的條文"}), chunks[1]]
    assert presets.load_index("bm25", changed, tmp_path) is None
    assert presets.load_index("bm25", chunks, tmp_path, k1=2.0) is None


def test_missing_or_corrupt_index_not_loaded(chunks, tmp_path):
    assert presets.load_index("bm25", chunks, tmp_path) is None
    presets.index_path(tmp_path, "bm25").write_text("{壞掉", encoding="utf-8")
    assert presets.load_index("bm25", chunks, tmp_path) is None
    # 壞檔時 load_or_build 仍能現場重建
    assert presets.load_or_build("bm25", chunks, tmp_path).search("加退選").hits


def test_index_size_mismatch_rejected(chunks):
    other = presets.build("bm25", chunks[:1]).index
    with pytest.raises(ValueError):
        bm25.BM25Retriever(chunks, index=other)


def test_default_search_uses_bm25_preset(monkeypatch, tmp_path):
    monkeypatch.setenv(bm25.CHUNKS_ENV, str(FIX / "chunks.jsonl"))
    monkeypatch.setenv(bm25.INDEX_ENV, str(tmp_path))
    bm25.set_retriever(None)
    try:
        result = bm25.search("被當", k=3)  # 只有靠口語擴展才找得到
        assert [h.chunk_id for h in result.hits] == ["fake-xueze#art1"]
        assert result.retriever == "bm25"
    finally:
        bm25.set_retriever(None)
