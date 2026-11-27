"""② 自寫 BM25 vs rank_bm25 對照（kickoff 驗證表：top-10 排序一致、分數誤差 1e-6 內）。

rank_bm25 只出現在 tests/ 當對照，不可拿來替代自寫的檢索（見 CLAUDE.md）。

IDF 對齊：rank_bm25 的 BM25Okapi 用課本 IDF ln((N−df+0.5)/(df+0.5))，
df > N/2 時是負數，再用 epsilon × 平均 IDF 當下限（見其 _calc_idf 原始碼）。
我們用 ln((N−df+0.5)/(df+0.5) + 1)，恆為正。兩者公式不同，所以這裡覆寫 _calc_idf
改成我們的版本，讓兩邊只差在「倒排索引 vs 逐篇計算」的實作，其餘公式完全相同。
"""

import math
from pathlib import Path

import pytest
from rank_bm25 import BM25Okapi

from regrag.ingest import load_chunks
from regrag.retrieval import bm25
from regrag.retrieval.bm25 import bigram_tokenize, build_index, char_tokenize
from regrag.retrieval.lexicon import load_lexicon, make_tokenizer

FIX = Path(__file__).parent / "fixtures"

# 假條文：只在這個測試裡用，模擬不同長度與重複詞，讓 df、tf、長度正規化都有變化
FAKE_TEXTS = [
    "學生學期學業成績不及格學分達修習學分二分之一者，應予退學。",
    "學生學期學業成績不及格學分連續兩學期達修習學分三分之一者，應予退學。",
    "學生加退選應於規定期間內於線上辦理，逾期不予受理。",
    "學生於學期中得申請停修，停修以一科為限。",
    "學士班修業年限為四年，未修滿應修學分者得延長修業年限，最多二年。",
    "學生因病或其他事故得申請休學，休學累計以二學年為限。",
    "學生應徵服兵役者，得申請保留學籍。",
    "本系學生須通過英文畢業門檻，多益成績達550分以上。",
    "學分學分學分",  # 極短、全部重複
]
# 不放空條文：b=1 時 rank_bm25 對空文件會算出 0/0 = NaN（它逐篇計算，沒命中也算）。
# 空文件的行為另外在 test_bm25.py 測。


class LuceneIdfBM25(BM25Okapi):
    def _calc_idf(self, nd):
        n = self.corpus_size
        self.idf = {w: math.log((n - df + 0.5) / (df + 0.5) + 1.0) for w, df in nd.items()}


def _texts() -> list[str]:
    return [c.text for c in load_chunks(FIX / "chunks.jsonl")] + FAKE_TEXTS


QUERIES = [
    "被當一半會怎樣",
    "學生加退選",
    "學分 學分",  # 查詢詞重複，兩邊都要每次都算
    "休學可以休幾年",
    "toeic 550",
    "停修",
    "完全不相關的問題 xyz",
]

TOKENIZERS = {
    "char": char_tokenize,
    "bigram": bigram_tokenize,
    "bigram+userdict": make_tokenizer(load_lexicon(), bigram_tokenize),
}


@pytest.mark.parametrize("k1,b", [(1.5, 0.75), (1.2, 0.5), (2.0, 1.0), (0.9, 0.0)])
@pytest.mark.parametrize("tok_name", TOKENIZERS)
@pytest.mark.parametrize("query", QUERIES)
def test_scores_and_top10_match_rank_bm25(query, tok_name, k1, b):
    tokenizer = TOKENIZERS[tok_name]
    docs = [tokenizer(t) for t in _texts()]
    q = tokenizer(query)

    ref = LuceneIdfBM25(docs, k1=k1, b=b).get_scores(q)
    ours = bm25.score(build_index(docs), q, k1, b)

    for i, expected in enumerate(ref):
        assert ours.get(i, 0.0) == pytest.approx(float(expected), abs=1e-6)

    # top-10：rank_bm25 給每篇都打分（沒命中是 0），我們只回傳有命中的；同分時 doc 編號小的在前
    ref_top = sorted((i for i, s in enumerate(ref) if s > 0), key=lambda i: (-ref[i], i))[:10]
    assert [i for i, _ in bm25.top_k(ours, 10)] == ref_top


def test_idf_alignment_is_needed():
    """確認對照真的有意義：不覆寫 IDF 時，常見詞（df > N/2）的分數會不同。"""
    docs = [char_tokenize(t) for t in _texts()]
    q = char_tokenize("學生")
    raw = BM25Okapi(docs).get_scores(q)
    ours = bm25.score(build_index(docs), q)
    assert any(abs(ours.get(i, 0.0) - float(s)) > 1e-6 for i, s in enumerate(raw))
