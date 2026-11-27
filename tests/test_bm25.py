"""② 斷詞、倒排索引、BM25、檢索器與 search()。只用 tests/fixtures 的假資料。

tests/fixtures/chunks.jsonl 只有兩條：
- fake-xueze#art1「學生學期學業成績不及格學分達修習學分二分之一者，應予輔導。」
- fake-xuanke#art3「學生加退選應於規定期間內於線上辦理。」
"""

import json
import math
from pathlib import Path

import pytest

from regrag.ingest import load_chunks
from regrag.retrieval import bm25
from regrag.retrieval.bm25 import BM25Retriever, bigram_tokenize, build_index, char_tokenize

FIX = Path(__file__).parent / "fixtures"
CHUNKS = FIX / "chunks.jsonl"


@pytest.fixture
def chunks():
    return load_chunks(CHUNKS)


@pytest.fixture
def retriever(chunks):
    return BM25Retriever(chunks)


# ---- 斷詞 ----


def test_normalize_fullwidth_and_case():
    assert bm25.normalize("ＴＯＥＩＣ５５０") == "toeic550"


def test_char_tokenize():
    assert char_tokenize("休學，ＴＯＥＩＣ550分！") == ["休", "學", "toeic", "550", "分"]


def test_bigram_tokenize():
    assert bigram_tokenize("休學期，ok") == ["休", "學", "期", "休學", "學期", "ok"]


def test_tokenize_punctuation_only():
    assert char_tokenize("，。！？ ") == []
    assert bigram_tokenize("，。！？ ") == []


# ---- 倒排索引 ----


def test_build_index():
    idx = build_index([["a", "b", "a"], ["b"], []])
    assert idx.postings == {"a": [(0, 2)], "b": [(0, 1), (1, 1)]}
    assert idx.doc_lens == [3, 1, 0]
    assert idx.n_docs == 3
    assert idx.avgdl == pytest.approx(4 / 3)


def test_empty_index():
    idx = build_index([])
    assert idx.avgdl == 0.0
    assert bm25.score(idx, ["a"]) == {}


# ---- BM25 ----


def test_idf_always_positive():
    assert bm25.idf(10, 10) > 0  # df = N 時課本公式會是負數
    assert bm25.idf(10, 1) > bm25.idf(10, 5)


def test_bm25_hand_computed():
    idx = build_index([["a", "b", "a"], ["b"], []])
    k1, b, avgdl = 1.5, 0.75, 4 / 3
    idf_a = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1)
    idf_b = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1)

    def term(w, tf, dl):
        return w * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

    scores = bm25.score(idx, ["a", "b"], k1, b)
    assert scores[0] == pytest.approx(term(idf_a, 2, 3) + term(idf_b, 1, 3))
    assert scores[1] == pytest.approx(term(idf_b, 1, 1))
    assert 2 not in scores  # 沒命中的 doc 不出現


def _reference_bm25(docs: list[list[str]], query: list[str], k1: float, b: float) -> list[float]:
    """不經倒排索引、直接照公式逐篇算，用來對照。"""
    n = len(docs)
    avgdl = sum(map(len, docs)) / n
    out = []
    for d in docs:
        s = 0.0
        for t in query:
            tf = d.count(t)
            if tf:
                df = sum(t in x for x in docs)
                w = math.log((n - df + 0.5) / (df + 0.5) + 1)
                s += w * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))
        out.append(s)
    return out


def _queries() -> list[str]:
    lines = CHUNKS.with_name("dev.jsonl").read_text(encoding="utf-8").splitlines()
    dev = [json.loads(line)["query"] for line in lines if line.strip()]
    return dev + ["學生加退選", "學分 學分", "輔導"]


@pytest.mark.parametrize("k1,b", [(1.5, 0.75), (1.2, 0.5), (2.0, 1.0)])
@pytest.mark.parametrize("tokenizer", [char_tokenize, bigram_tokenize])
@pytest.mark.parametrize("query", _queries())
def test_bm25_matches_reference(chunks, query, tokenizer, k1, b):
    docs = [tokenizer(c.text) for c in chunks]
    q = tokenizer(query)
    ref = _reference_bm25(docs, q, k1, b)
    ours = bm25.score(build_index(docs), q, k1, b)
    for i, expected in enumerate(ref):
        assert ours.get(i, 0.0) == pytest.approx(expected, abs=1e-6)
    ref_top = sorted((i for i, s in enumerate(ref) if s > 0), key=lambda i: (-ref[i], i))
    assert [i for i, _ in bm25.top_k(ours, 10)] == ref_top[:10]


def test_repeated_query_term_counts_twice():
    idx = build_index([["a"], ["b"]])
    assert bm25.score(idx, ["a", "a"])[0] == pytest.approx(2 * bm25.score(idx, ["a"])[0])


def test_top_k_tie_break_by_doc_order():
    assert bm25.top_k({2: 1.0, 0: 1.0, 1: 3.0}, 2) == [(1, 3.0), (0, 1.0)]


# ---- 檢索器 ----


def test_search_result_contract(retriever, chunks):
    result = retriever.search("學生加退選", k=2)
    assert result.query == "學生加退選"
    assert result.retriever == "bm25"
    assert len(result.hits) == 2  # 兩條都有「學生」
    assert result.latency_ms >= 0
    by_id = {c.chunk_id: c for c in chunks}
    for pos, hit in enumerate(result.hits):
        assert hit.rank == pos + 1
        assert hit.chunk == by_id[hit.chunk_id]  # 帶出的是建索引用的同一份 chunk
    assert result.hits[0].chunk_id == "fake-xuanke#art3"
    assert result.hits[0].score > result.hits[1].score


def test_k_limits_hits(retriever):
    assert len(retriever.search("學生", k=1).hits) == 1


def test_only_matching_chunks_returned(retriever):
    # 「輔導」只出現在一條，k 再大也只回一筆
    assert [h.chunk_id for h in retriever.search("輔導", k=10).hits] == ["fake-xueze#art1"]


def test_no_match_returns_empty(retriever):
    assert retriever.search("ＸＹＺ？", k=5).hits == []


def test_title_not_indexed(retriever):
    # 「測試」只出現在 doc_title，不在條文內容
    assert retriever.search("測試", k=5).hits == []


def test_index_does_not_change_chunk_text(chunks):
    before = [c.text for c in chunks]
    BM25Retriever(chunks)
    assert [c.text for c in chunks] == before


def test_bad_k_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search("學生", k=0)


def test_duplicate_chunk_id_rejected(chunks):
    with pytest.raises(ValueError):
        BM25Retriever([chunks[0], chunks[0]])


# ---- 對外 search() ----


def test_module_search_uses_injected_retriever(chunks):
    bm25.set_retriever(BM25Retriever(chunks, char_tokenize))
    try:
        assert bm25.search("加退選", k=2).hits[0].chunk_id == "fake-xuanke#art3"
    finally:
        bm25.set_retriever(None)


def test_default_search_loads_chunks_from_env(monkeypatch):
    monkeypatch.setenv(bm25.CHUNKS_ENV, str(CHUNKS))
    bm25.set_retriever(None)
    try:
        result = bm25.search("加退選", k=3)
        assert result.hits[0].chunk_id == "fake-xuanke#art3"
        assert all(h.chunk.doc_id.startswith("fake-") for h in result.hits)
    finally:
        bm25.set_retriever(None)
