"""把 LLM 的原始輸出解析成 LLMOutput。救不回來就丟 LLMOutputError（不可當成拒答）。"""

from pydantic import BaseModel, ConfigDict, ValidationError

_THINK_END = "</think>"


class LLMOutputError(Exception):
    """LLM 輸出不符格式。raw 保留原始輸出，方便除錯與統計。"""

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


class LLMOutput(BaseModel):
    """③ 內部用：LLM 依 prompt 規定輸出的 JSON。citations 是 prompt 裡的條文編號。"""

    model_config = ConfigDict(strict=True, frozen=True)

    answer: str
    citations: list[int]
    refused: bool


def _strip_thinking(raw: str) -> str:
    """有 </think> 就只留最後一個 </think> 之後的文字。

    不靠成對的 <think>…</think>：有些模型的對話模板把 <think> 放在 prompt 裡，
    輸出只會有思考內容加上 </think>。
    """
    _, sep, after = raw.rpartition(_THINK_END)
    return after if sep else raw


def _extract_json_object(raw: str) -> str | None:
    """去掉思考內容，取第一個 { 到最後一個 } 之間的文字（順便去掉 ``` 圍欄與前後廢話）。"""
    text = _strip_thinking(raw)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return None
    return text[start : end + 1]


def parse_llm_output(raw: str) -> LLMOutput:
    candidate = _extract_json_object(raw)
    if candidate is None:
        raise LLMOutputError("LLM 輸出裡找不到 JSON 物件", raw)
    try:
        return LLMOutput.model_validate_json(candidate)
    except ValidationError as e:
        raise LLMOutputError(f"LLM 輸出的 JSON 不符格式：{e.error_count()} 個錯誤", raw) from e
