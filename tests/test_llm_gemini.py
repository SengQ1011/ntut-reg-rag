"""GeminiClient 測試。把 SDK 的 genai.Client 換成假的，不連網。"""

from types import SimpleNamespace
from typing import Any

import pytest

from regrag.llm import gemini
from regrag.llm.gemini import GeminiClient


class FakeModels:
    def __init__(self, text: str | None) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)


class FakeGenaiClient:
    """模仿 genai.Client：記下建構參數，models.generate_content 回固定文字。"""

    instances: list["FakeGenaiClient"] = []
    reply: str | None = '{"answer": "可以。", "citations": [1], "refused": false}'

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.models = FakeModels(FakeGenaiClient.reply)
        FakeGenaiClient.instances.append(self)


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> type[FakeGenaiClient]:
    FakeGenaiClient.instances = []
    FakeGenaiClient.reply = '{"answer": "可以。", "citations": [1], "refused": false}'
    monkeypatch.setattr(gemini.genai, "Client", FakeGenaiClient)
    return FakeGenaiClient


def test_model_name_kept(fake_sdk: type[FakeGenaiClient]):
    assert GeminiClient(model="fake-gemini").model == "fake-gemini"


def test_no_api_key_lets_sdk_read_env(fake_sdk: type[FakeGenaiClient]):
    GeminiClient(model="fake-gemini")
    assert fake_sdk.instances[0].kwargs["api_key"] is None


def test_explicit_api_key_passed_stripped(fake_sdk: type[FakeGenaiClient]):
    GeminiClient(model="fake-gemini", api_key=" user-key\n")
    assert fake_sdk.instances[0].kwargs["api_key"] == "user-key"


@pytest.mark.parametrize("api_key", ["", "   "])
def test_blank_explicit_api_key_raises(fake_sdk: type[FakeGenaiClient], api_key: str):
    with pytest.raises(ValueError, match="api_key"):
        GeminiClient(model="fake-gemini", api_key=api_key)


def test_timeout_is_set(fake_sdk: type[FakeGenaiClient]):
    GeminiClient(model="fake-gemini")
    assert fake_sdk.instances[0].kwargs["http_options"].timeout > 0


def test_complete_sends_prompt_and_returns_raw_text(fake_sdk: type[FakeGenaiClient]):
    fake_sdk.reply = "```json\n{}\n```"  # 原樣交回，清理是 parsing 的事
    client = GeminiClient(model="fake-gemini")

    assert client.complete("問題 prompt") == "```json\n{}\n```"
    call = fake_sdk.instances[0].models.calls[0]
    assert call["model"] == "fake-gemini"
    assert call["contents"] == "問題 prompt"


def test_complete_requests_json_output(fake_sdk: type[FakeGenaiClient]):
    GeminiClient(model="fake-gemini").complete("p")
    config = fake_sdk.instances[0].models.calls[0]["config"]
    assert config.response_mime_type == "application/json"


def test_empty_response_returns_empty_string(fake_sdk: type[FakeGenaiClient]):
    fake_sdk.reply = None  # SDK 被安全過濾擋下時 text 是 None
    assert GeminiClient(model="fake-gemini").complete("p") == ""
