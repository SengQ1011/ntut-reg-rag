"""Gemini 後端，實作 LLMClient。API key 從環境變數讀，不寫在程式裡。"""

import os

from google import genai
from google.genai import types

from regrag.llm.parsing import LLMOutputError

API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_TIMEOUT_S = 60.0


class GeminiClient:
    """呼叫 Gemini API。連線、逾時、額度用完等錯誤由 SDK 直接丟例外，不在這裡吞掉。

    model 由呼叫端指定（預設值在 backends.py 管），這裡不預設。
    api_key 不給就讀環境變數 GEMINI_API_KEY；給了就用它（例如網頁使用者自己輸入的 key）。
    key 只交給 SDK，不存成屬性、不寫進 log。
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        if not model.strip():
            raise ValueError("model 不可為空")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s 必須 > 0，收到 {timeout_s}")
        if api_key is not None and not api_key.strip():
            raise ValueError("api_key 不可為空字串；不給 key 請傳 None")
        self.model = model
        self._client = genai.Client(
            api_key=api_key.strip() if api_key is not None else _read_api_key(),
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )

    def complete(self, prompt: str) -> str:
        """回傳模型輸出的原始文字；要求 JSON 輸出，但清理與驗證交給 parsing。"""
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        if not response.text:
            # 例如被安全過濾擋下；屬於系統錯誤，不可當成拒答
            raise LLMOutputError("Gemini 沒有回傳文字", raw="")
        return response.text


def _read_api_key() -> str:
    """明確只讀 GEMINI_API_KEY，避免 SDK 默默改用其他環境變數。"""
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"未設定環境變數 {API_KEY_ENV}")
    return key
