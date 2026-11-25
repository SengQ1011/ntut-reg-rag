"""make_client：依 backend 建 client，模型名稱從環境變數讀。不連網。"""

import pytest

from regrag.llm import backends
from regrag.llm.backends import DEFAULT_GEMINI_MODEL, GEMINI_MODEL_ENV, make_client


class FakeGeminiClient:
    def __init__(self, model: str) -> None:
        self.model = model


@pytest.fixture(autouse=True)
def fake_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "GeminiClient", FakeGeminiClient)


def test_gemini_model_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(GEMINI_MODEL_ENV, "fake-gemini-x")
    client = make_client("gemini")
    assert isinstance(client, FakeGeminiClient)
    assert client.model == "fake-gemini-x"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_gemini_model_defaults_when_unset(monkeypatch: pytest.MonkeyPatch, value: str | None):
    if value is None:
        monkeypatch.delenv(GEMINI_MODEL_ENV, raising=False)
    else:
        monkeypatch.setenv(GEMINI_MODEL_ENV, value)
    assert make_client("gemini").model == DEFAULT_GEMINI_MODEL


def test_gemini_model_env_is_stripped(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(GEMINI_MODEL_ENV, "  fake-gemini-x \n")
    assert make_client("gemini").model == "fake-gemini-x"


def test_ollama_not_implemented_yet():
    with pytest.raises(NotImplementedError, match="gemini"):
        make_client("ollama")


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        make_client("openai")  # type: ignore[arg-type]
