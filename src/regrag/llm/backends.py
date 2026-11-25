"""依 backend 名稱建出 LLMClient。模型名稱從環境變數讀，沒設就用預設值。"""

import os
from typing import Literal

from regrag.llm.client import LLMClient
from regrag.llm.gemini import GeminiClient

Backend = Literal["ollama", "gemini"]

GEMINI_MODEL_ENV = "GEMINI_MODEL"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


def make_client(backend: Backend) -> LLMClient:
    if backend == "gemini":
        return GeminiClient(model=_env_or_default(GEMINI_MODEL_ENV, DEFAULT_GEMINI_MODEL))
    if backend == "ollama":
        raise NotImplementedError("Ollama 後端尚未實作，目前請用 backend='gemini'")
    raise ValueError(f"不認識的 backend：{backend!r}（可用：ollama、gemini）")


def _env_or_default(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default
