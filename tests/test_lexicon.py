"""② 校務詞庫：詞典比對、口語擴展。只用 tests/fixtures 的假資料。"""

from pathlib import Path

import pytest

from regrag.ingest import load_chunks
from regrag.retrieval import lexicon as lx
from regrag.retrieval.bm25 import BM25Retriever, bigram_tokenize

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def lex():
    return lx.load_lexicon(FIX / "userdict.txt")


@pytest.fixture
def chunks():
    return load_chunks(FIX / "chunks.jsonl")


# ---- 讀檔 ----


def test_load_terms_and_synonyms(lex):
    # 兩字的「輔導」不收；校規用詞「二分之一」「不及格」(>=3 字) 也成為校務詞
    assert lex.terms == {"加退選", "二分之一", "不及格"}
    assert lex.synonyms == {"被當": ("不及格",), "一半": ("二分之一",)}
    assert lex.max_len == 4


def test_parse_normalizes_and_merges():
    lex = lx.parse_lexicon(["ＴＯＥＩＣ門檻", "當兵 => 兵役", "當兵 => 服役"])
    assert "toeic門檻" in lex.terms
    assert lex.synonyms["當兵"] == ("兵役", "服役")


@pytest.mark.parametrize("line", ["被當 =>", "=> 不及格", "加退選 停修"])
def test_bad_line_rejected(line):
    with pytest.raises(ValueError):
        lx.parse_lexicon([line])


def test_comments_and_blank_lines_ignored():
    assert lx.parse_lexicon(["# 註解", "", "   "]) == lx.Lexicon()


def test_default_userdict_is_valid():
    lex = lx.load_lexicon()
    assert lex.terms
    assert all(len(t) >= lx.MIN_TERM_LEN for t in lex.terms)
    assert lex.synonyms


# ---- 詞典比對 ----


def test_find_terms_counts_every_occurrence(lex):
    assert lx.find_terms("加退選與加退選", lex) == ["加退選", "加退選"]


def test_find_terms_overlapping(lex):
    assert lx.find_terms("不及格學分達二分之一", lex) == ["不及格", "二分之一"]


def test_find_terms_normalizes(lex):
    assert lx.find_terms("ＡＢ加退選", lex) == ["加退選"]


def test_find_terms_empty_lexicon():
    assert lx.find_terms("加退選", lx.Lexicon()) == []


def test_tokenizer_adds_terms_after_base(lex):
    tok = lx.make_tokenizer(lex)
    base = bigram_tokenize("線上加退選")
    assert tok("線上加退選") == base + ["加退選"]


# ---- 口語擴展 ----


def test_expander_appends_formal_terms(lex):
    expand = lx.make_expander(lex)
    assert expand("被當一半會怎樣") == "被當一半會怎樣 不及格 二分之一"


def test_expander_leaves_other_queries(lex):
    assert lx.make_expander(lex)("加退選期間") == "加退選期間"


def test_expansion_finds_chunk_bigram_misses(lex, chunks):
    """「被當」在條文裡不存在；擴展成「不及格」才命中正確條文。"""
    plain = BM25Retriever(chunks, lx.make_tokenizer(lex))
    expanded = BM25Retriever(chunks, lx.make_tokenizer(lex), query_expander=lx.make_expander(lex))
    assert plain.search("被當", k=5).hits == []
    assert [h.chunk_id for h in expanded.search("被當", k=5).hits] == ["fake-xueze#art1"]


def test_expansion_only_on_query_side(lex, chunks):
    before = [c.text for c in chunks]
    r = BM25Retriever(chunks, lx.make_tokenizer(lex), query_expander=lx.make_expander(lex))
    r.search("被當", k=5)
    assert [c.text for c in r.chunks] == before
    # 索引裡沒有被擴展出來的口語詞
    assert "被當" not in r.index.postings
