"""③ 生成。對外只提供 answer。"""

import time

from regrag import retrieval
from regrag.llm import backends
from regrag.llm.backends import Backend
from regrag.llm.generate import generate
from regrag.schemas import Answer


def answer(query: str, k: int = 5, backend: Backend = "ollama") -> Answer:
    """檢索 → LLM 回答。latency_ms 從這裡開始算（合約：answer() 端到端）。

    先檢查參數、建 client 再檢索：k 不合法、缺 key、backend 打錯這類錯誤要在檢索前就丟出來。
    """
    started_at = time.perf_counter()
    if k < 1:
        raise ValueError(f"k 必須 >= 1，收到 {k}")
    client = backends.make_client(backend)
    result = retrieval.search(query, k=k)
    return generate(result, client, started_at)
