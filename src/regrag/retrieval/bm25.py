"""② 斷詞、倒排索引、BM25 與 search()。全部自己寫，不依賴外部檢索套件。

斷詞只產生建索引／查詢用的詞，不會改動 Chunk.text。
- bigram_tokenize：正式用。漢字切成單字 + 相鄰兩字（unigram + bigram），不需詞典。
- char_tokenize：baseline，漢字只切單字。
英文字母、數字各自連續一段當一個詞（"toeic550" → toeic、550）。

BM25（Okapi）：
score(D, Q) = Σ_{q ∈ Q} idf(q) · tf(q,D)·(k1+1) / (tf(q,D) + k1·(1 − b + b·|D|/avgdl))
IDF 用 ln((N − df + 0.5)/(df + 0.5) + 1)：恆為正，不會像課本 Robertson IDF
在 df > N/2 時變負數而需要另設下限。查詢詞重複出現時每次都算。
"""

import math
import os
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from regrag.ingest import load_chunks
from regrag.schemas import Chunk, Hit, RetrievalResult

K1 = 1.5
B = 0.75
# 指定 chunks 檔的環境變數；沒設就用 load_chunks() 的預設路徑（data/chunks.jsonl）
CHUNKS_ENV = "REGRAG_CHUNKS"
# 建好的索引放哪個資料夾（scripts/build_index.py 寫入；不進 repo）
INDEX_ENV = "REGRAG_INDEX_DIR"
DEFAULT_INDEX_DIR = "index"

Tokenizer = Callable[[str], list[str]]
QueryExpander = Callable[[str], str]

# ---- 斷詞 ----

_CJK = "㐀-䶿一-鿿豈-﫿"  # 漢字（含擴充 A、相容漢字）
_RUN = re.compile(f"[a-z]+|[0-9]+|[{_CJK}]+")


def normalize(text: str) -> str:
    """NFKC（全形英數轉半形）＋ 轉小寫。只用在斷詞前，不回寫資料。"""
    return unicodedata.normalize("NFKC", text).lower()


def _runs(text: str) -> list[str]:
    """切成連續的英文、數字、漢字片段，標點與空白丟掉。"""
    return _RUN.findall(normalize(text))


def _is_cjk(run: str) -> bool:
    return not run[0].isascii()


def char_tokenize(text: str) -> list[str]:
    """baseline：漢字一字一詞。"""
    tokens: list[str] = []
    for run in _runs(text):
        tokens.extend(run if _is_cjk(run) else [run])
    return tokens


def bigram_tokenize(text: str) -> list[str]:
    """漢字片段產生所有單字與相鄰兩字；單字讓一字查詢也找得到，兩字保留詞序資訊。"""
    tokens: list[str] = []
    for run in _runs(text):
        if _is_cjk(run):
            tokens.extend(run)
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        else:
            tokens.append(run)
    return tokens


# ---- 倒排索引 ----


@dataclass(frozen=True)
class InvertedIndex:
    # term → [(doc 編號, 該詞在 doc 中出現次數)]；doc 編號 = 在 chunks 中的位置，由小到大
    postings: dict[str, list[tuple[int, int]]]
    doc_lens: list[int]  # 每個 doc 斷詞後的詞數

    @property
    def n_docs(self) -> int:
        return len(self.doc_lens)

    @property
    def avgdl(self) -> float:
        return sum(self.doc_lens) / self.n_docs if self.n_docs else 0.0


def build_index(docs_tokens: list[list[str]]) -> InvertedIndex:
    postings: dict[str, list[tuple[int, int]]] = {}
    for doc_id, tokens in enumerate(docs_tokens):
        for term, tf in Counter(tokens).items():
            postings.setdefault(term, []).append((doc_id, tf))
    return InvertedIndex(postings=postings, doc_lens=[len(t) for t in docs_tokens])


# ---- 計分 ----


def idf(n_docs: int, df: int) -> float:
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


def score(
    index: InvertedIndex, query_tokens: list[str], k1: float = K1, b: float = B
) -> dict[int, float]:
    """回傳 {doc 編號: 分數}，只含至少命中一個查詢詞的 doc。"""
    scores: dict[int, float] = {}
    for term in query_tokens:
        posting = index.postings.get(term)
        if not posting:
            continue
        w = idf(index.n_docs, len(posting))
        for doc_id, tf in posting:
            norm = k1 * (1 - b + b * index.doc_lens[doc_id] / index.avgdl)
            scores[doc_id] = scores.get(doc_id, 0.0) + w * tf * (k1 + 1) / (tf + norm)
    return scores


def top_k(scores: dict[int, float], k: int) -> list[tuple[int, float]]:
    """分數高到低取前 k 筆；同分時 doc 編號小的在前，結果可重現。"""
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]


# ---- 檢索器 ----


class BM25Retriever:
    """用同一份 chunks 建索引並帶出 Hit.chunk，保證兩者版本一致。索引只用 chunk.text。"""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        tokenizer: Tokenizer = bigram_tokenize,
        k1: float = K1,
        b: float = B,
        query_expander: QueryExpander | None = None,
        index: InvertedIndex | None = None,
    ) -> None:
        """query_expander：查詢斷詞前先改寫（例如口語詞擴展），只作用在查詢端。
        index：已用同一個 tokenizer 對同一份 chunks 建好的索引（從檔案載入時用），
        沒給就現場建。
        """
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("chunk_id 重複，索引無法對應回唯一條文")
        self.chunks = list(chunks)
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.query_expander = query_expander
        if index is None:
            index = build_index([tokenizer(c.text) for c in self.chunks])
        elif index.n_docs != len(self.chunks):
            raise ValueError("索引的文件數和 chunks 不一致")
        self.index = index

    def query_tokens(self, query: str) -> list[str]:
        if self.query_expander is not None:
            query = self.query_expander(query)
        return self.tokenizer(query)

    def rank(self, query: str, k: int = 10) -> list[Hit]:
        """只回傳至少命中一個查詢詞的條文，所以可能少於 k 筆（③ 判斷拒答會用到）。"""
        if k < 1:
            raise ValueError("k 必須 >= 1")
        scores = score(self.index, self.query_tokens(query), self.k1, self.b)
        return [
            Hit(chunk_id=self.chunks[i].chunk_id, score=s, rank=r, chunk=self.chunks[i])
            for r, (i, s) in enumerate(top_k(scores, k), start=1)
        ]

    def search(self, query: str, k: int = 10) -> RetrievalResult:
        start = time.perf_counter()
        return _result(query, self.rank(query, k), start)


def _result(query: str, hits: list[Hit], start: float) -> RetrievalResult:
    latency_ms = (time.perf_counter() - start) * 1000
    return RetrievalResult(query=query, retriever="bm25", hits=hits, latency_ms=latency_ms)


# ---- 對外 search() ----

_default: BM25Retriever | None = None


def set_retriever(retriever: BM25Retriever | None) -> None:
    """換掉 search() 用的檢索器（測試、baseline 評估用）；傳 None 代表下次重新載入。"""
    global _default
    _default = retriever


def _get_retriever() -> BM25Retriever:
    """預設檢索器：bigram ＋ 校務詞庫 ＋ 口語擴展（見 presets.py）。
    index/ 下有建好且和目前 chunks、詞庫一致的索引就直接載入，否則現場建。
    """
    global _default
    if _default is None:
        from regrag.retrieval.presets import load_or_build  # 避免循環 import

        path = os.environ.get(CHUNKS_ENV)
        chunks = load_chunks(path) if path else load_chunks()
        _default = load_or_build("bm25", chunks, os.environ.get(INDEX_ENV, DEFAULT_INDEX_DIR))
    return _default


def search(query: str, k: int = 10) -> RetrievalResult:
    """對外介面。第一次呼叫才載入 chunks、建索引，時間算進 latency_ms（見 schemas）。"""
    start = time.perf_counter()
    return _result(query, _get_retriever().rank(query, k), start)
