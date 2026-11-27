"""② 檢索。對外只提供 search。斷詞、倒排索引、BM25 必須自己寫。"""

from regrag.retrieval.bm25 import search

__all__ = ["search"]
