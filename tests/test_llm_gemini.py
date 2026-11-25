"""GeminiClient 測試。把 SDK 的 genai.Client 換成假的，不連網。"""

from types import SimpleNamespace
from typing import Any

import pytest

from regrag.llm import gemini
from regrag.llm.gemini import API_KEY_ENV, GeminiClient
from regrag.llm.parsing import LLMOutputError


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
    monkeypatch.setenv(API_KEY_ENV, "fake-key")
    return FakeGenaiClient


def test_model_name_kept(fake_sdk: type[FakeGenaiClient]):
    assert GeminiClient(model="fake-gemini").model == "fake-gemini"


def test_api_key_read_from_env_and_passed_to_sdk(fake_sdk: type[FakeGenaiClient]):
    GeminiClient(model="fake-gemini")
    assert fake_sdk.instances[0].kwargs["api_key"] == "fake-key"


def test_timeout_passed_to_sdk_in_milliseconds(fake_sdk: type[FakeGenaiClient]):
    GeminiClient(model="fake-gemini", timeout_s=12.5)
    http_options = fake_sdk.instances[0].kwargs["http_options"]
    assert http_options.timeout == 12_500


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_api_key_raises(
    fake_sdk: type[FakeGenaiClient], monkeypatch: pytest.MonkeyPatch, value: str | None
):
    if value is None:
        monkeypatch.delenv(API_KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(API_KEY_ENV, value)
    with pytest.raises(RuntimeError, match=API_KEY_ENV):
        GeminiClient(model="fake-gemini")


def test_explicit_api_key_used_without_env(
    fake_sdk: type[FakeGenaiClient], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    GeminiClient(model="fake-gemini", api_key="user-key")
    assert fake_sdk.instances[0].kwargs["api_key"] == "user-key"


def test_explicit_api_key_overrides_env(fake_sdk: type[FakeGenaiClient]):
    GeminiClient(model="fake-gemini", api_key="user-key")
    assert fake_sdk.instances[0].kwargs["api_key"] == "user-key"


@pytest.mark.parametrize("api_key", ["", "   "])
def test_blank_explicit_api_key_raises(fake_sdk: type[FakeGenaiClient], api_key: str):
    with pytest.raises(ValueError, match="api_key"):
        GeminiClient(model="fake-gemini", api_key=api_key)


@pytest.mark.parametrize("timeout_s", [0, -1])
def test_timeout_must_be_positive(fake_sdk: type[FakeGenaiClient], timeout_s: float):
    with pytest.raises(ValueError, match="timeout_s"):
        GeminiClient(model="fake-gemini", timeout_s=timeout_s)


def test_blank_model_raises(fake_sdk: type[FakeGenaiClient]):
    with pytest.raises(ValueError, match="model"):
        GeminiClient(model="  ")


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


@pytest.mark.parametrize("text", [None, ""])
def test_empty_response_raises(fake_sdk: type[FakeGenaiClient], text: str | None):
    fake_sdk.reply = text
    with pytest.raises(LLMOutputError, match="沒有回傳文字"):
        GeminiClient(model="fake-gemini").complete("p")
