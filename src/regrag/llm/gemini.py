"""Gemini 後端，實作 LLMClient。"""

from google import genai
from google.genai import types

TIMEOUT_MS = 60_000  # SDK 預設不逾時，不設的話卡住就永遠不回來


class GeminiClient:
    """呼叫 Gemini API。連線、逾時、額度用完等錯誤由 SDK 直接丟例外，不在這裡吞掉。

    api_key 不給時由 SDK 讀環境變數 GEMINI_API_KEY，沒有 key 會丟 ValueError；
    給了就用它（例如網頁使用者自己輸入的 key）。key 只交給 SDK，不存成屬性、不寫進 log。
    """

    def __init__(self, model: str, *, api_key: str | None = None) -> None:
        if api_key is not None and not api_key.strip():
            # SDK 收到空字串會改用伺服器環境變數的 key；網頁使用者留空時不可默默用掉它
            raise ValueError("api_key 不可為空字串；不給 key 請傳 None")
        self.model = model
        self._client = genai.Client(
            api_key=api_key.strip() if api_key is not None else None,
            http_options=types.HttpOptions(timeout=TIMEOUT_MS),
        )

    def complete(self, prompt: str) -> str:
        """回傳模型輸出的原始文字；要求 JSON 輸出，但清理與驗證交給 parsing。"""
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        # 空回應（例如被安全過濾擋下）交給 parsing 丟 LLMOutputError，不會被當成拒答
        return response.text or ""
