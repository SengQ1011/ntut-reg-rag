"""LLM 後端的介面。generate() 只依賴這個 Protocol，測試時換成假的 LLM。"""

from typing import Protocol


class LLMClient(Protocol):
    model: str  # 寫進 Answer.model，例如 "qwen3:8b"

    def complete(self, prompt: str) -> str:
        """送出 prompt，回傳模型輸出的原始文字。連線或逾時錯誤直接丟例外。"""
        ...
