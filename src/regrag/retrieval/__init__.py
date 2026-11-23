"""② 檢索。對外只提供 search。斷詞、倒排索引、BM25 必須自己寫。"""

from regrag.schemas import RetrievalResult


def search(query: str, k: int = 10) -> RetrievalResult:
    raise NotImplementedError("② 尚未實作")
