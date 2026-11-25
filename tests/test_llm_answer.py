"""answer()：search → 選 client → generate。search 與 client 都換成假的，不連網。"""

import time

import pytest

from regrag import llm
from regrag import retrieval as retrieval_module
from regrag.llm import backends
from regrag.schemas import RetrievalResult

REPLY = '{"answer": "應予輔導。", "citations": [1], "refused": false}'


class FakeLLM:
    model = "fake-llm"

    def complete(self, prompt: str) -> str:
        return REPLY


class Recorder:
    """記下 search 與 make_client 被怎麼呼叫。"""

    def __init__(self, result: RetrievalResult, search_delay_s: float = 0.0) -> None:
        self.result = result
        self.search_delay_s = search_delay_s
        self.search_calls: list[tuple[str, int]] = []
        self.backends: list[str] = []

    def search(self, query: str, k: int = 10) -> RetrievalResult:
        self.search_calls.append((query, k))
        time.sleep(self.search_delay_s)
        return self.result

    def make_client(self, backend: str) -> FakeLLM:
        self.backends.append(backend)
        return FakeLLM()


@pytest.fixture
def patch(monkeypatch: pytest.MonkeyPatch):
    def _patch(result: RetrievalResult, search_delay_s: float = 0.0) -> Recorder:
        rec = Recorder(result, search_delay_s)
        monkeypatch.setattr(retrieval_module, "search", rec.search)
        monkeypatch.setattr(backends, "make_client", rec.make_client)
        return rec

    return _patch


def test_answer_returns_contract_answer(patch, retrieval: RetrievalResult):
    patch(retrieval)
    ans = llm.answer(retrieval.query, backend="gemini")
    assert ans.refused is False
    assert ans.model == "fake-llm"
    assert ans.retrieval == retrieval


def test_answer_passes_query_k_and_backend(patch, retrieval: RetrievalResult):
    rec = patch(retrieval)
    llm.answer("成績不及格會怎樣？", k=3, backend="gemini")
    assert rec.search_calls == [("成績不及格會怎樣？", 3)]
    assert rec.backends == ["gemini"]


def test_default_backend_is_ollama(patch, retrieval: RetrievalResult):
    rec = patch(retrieval)
    llm.answer(retrieval.query)
    assert rec.backends == ["ollama"]


def test_latency_includes_search_time(patch, retrieval: RetrievalResult):
    patch(retrieval, search_delay_s=0.05)
    ans = llm.answer(retrieval.query, backend="gemini")
    assert ans.latency_ms >= 50


def test_client_error_raised_before_search(
    patch, monkeypatch: pytest.MonkeyPatch, retrieval: RetrievalResult
):
    """缺 key 等設定錯誤要在檢索前就丟出來，不浪費一次 search。"""
    rec = patch(retrieval)

    def broken_client(backend: str) -> FakeLLM:
        raise RuntimeError("未設定環境變數")

    monkeypatch.setattr(backends, "make_client", broken_client)
    with pytest.raises(RuntimeError):
        llm.answer("問題", backend="gemini")
    assert rec.search_calls == []
