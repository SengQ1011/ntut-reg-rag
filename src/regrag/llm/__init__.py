"""③ 生成。對外只提供 answer。"""

from typing import Literal

from regrag.schemas import Answer


def answer(query: str, k: int = 5, backend: Literal["ollama", "gemini"] = "ollama") -> Answer:
    raise NotImplementedError("③ 尚未實作")
