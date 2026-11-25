"""generate：假檢索結果 + 假 LLM → 通過合約驗證的 Answer。不連網、不呼叫真的 LLM。"""

import logging
import time

import pytest

from regrag.llm.generate import REFUSAL_MESSAGE, generate
from regrag.llm.parsing import LLMOutputError
from regrag.schemas import Answer, RetrievalResult


class FakeLLM:
    """回傳寫死的字串，並記下收到的 prompt。"""

    model = "fake-llm"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def reply(answer: str = "應予輔導。", citations: str = "[1]", refused: str = "false") -> str:
    return f'{{"answer": "{answer}", "citations": {citations}, "refused": {refused}}}'


def test_normal_answer_passes_contract(retrieval: RetrievalResult):
    llm = FakeLLM(reply(citations="[1]"))
    ans = generate(retrieval, llm)

    Answer.model_validate(ans.model_dump())
    assert ans.query == retrieval.query
    assert ans.answer == "應予輔導。"
    assert ans.refused is False
    assert ans.model == "fake-llm"
    assert ans.retrieval == retrieval
    assert ans.latency_ms >= 0


def test_prompt_and_answer_use_retrieval_query(retrieval: RetrievalResult):
    llm = FakeLLM(reply())
    ans = generate(retrieval, llm)
    (prompt,) = llm.prompts
    assert retrieval.query in prompt
    assert ans.query == ans.retrieval.query == retrieval.query


def test_citation_fields_copied_from_hit_chunk(retrieval: RetrievalResult):
    ans = generate(retrieval, FakeLLM(reply(citations="[2]")))
    (cit,) = ans.citations
    chunk = retrieval.hits[1].chunk
    assert cit.chunk_id == chunk.chunk_id
    assert cit.doc_title == chunk.doc_title
    assert cit.article_no == chunk.article_no
    assert cit.source_url == chunk.source_url


def test_only_top_k_hits_go_into_prompt(retrieval: RetrievalResult):
    llm = FakeLLM(reply(citations="[1]"))
    generate(retrieval, llm, k=1)
    (prompt,) = llm.prompts
    assert retrieval.hits[0].chunk.text in prompt
    assert retrieval.hits[1].chunk.text not in prompt


def test_citation_beyond_k_is_invalid(retrieval: RetrievalResult):
    # k=1 時只給 LLM 看一條，回 [2] 就是不合法編號，即使檢索結果裡有第二筆
    ans = generate(retrieval, FakeLLM(reply(citations="[1, 2]")), k=1)
    assert [c.chunk_id for c in ans.citations] == [retrieval.hits[0].chunk_id]


def test_empty_hits_refuse_without_calling_llm(empty_retrieval: RetrievalResult):
    llm = FakeLLM(reply())
    ans = generate(empty_retrieval, llm)
    assert ans.refused is True
    assert ans.citations == []
    assert ans.answer == REFUSAL_MESSAGE
    assert llm.prompts == []


def test_llm_refusal_ignores_citations(retrieval: RetrievalResult):
    llm = FakeLLM(reply(answer="查無規定", citations="[1]", refused="true"))
    ans = generate(retrieval, llm)
    assert ans.refused is True
    assert ans.citations == []
    assert ans.answer == REFUSAL_MESSAGE


def test_invalid_numbers_dropped_and_logged(
    retrieval: RetrievalResult, caplog: pytest.LogCaptureFixture
):
    llm = FakeLLM(reply(citations="[0, 1, 7, -1]"))
    with caplog.at_level(logging.WARNING, logger="regrag.llm.generate"):
        ans = generate(retrieval, llm)
    assert [c.chunk_id for c in ans.citations] == [retrieval.hits[0].chunk_id]
    assert "7" in caplog.text


def test_duplicate_numbers_merged_in_order(retrieval: RetrievalResult):
    ans = generate(retrieval, FakeLLM(reply(citations="[2, 1, 2]")))
    assert [c.chunk_id for c in ans.citations] == [
        retrieval.hits[1].chunk_id,
        retrieval.hits[0].chunk_id,
    ]


@pytest.mark.parametrize("citations", ["[7, 8]", "[]"])
def test_answer_without_valid_citation_raises(retrieval: RetrievalResult, citations: str):
    llm = FakeLLM(reply(citations=citations))
    with pytest.raises(LLMOutputError):
        generate(retrieval, llm)


def test_unparseable_output_raises(retrieval: RetrievalResult):
    with pytest.raises(LLMOutputError):
        generate(retrieval, FakeLLM("依學則規定應予輔導。"))


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_positive(retrieval: RetrievalResult, k: int):
    llm = FakeLLM(reply())
    with pytest.raises(ValueError):
        generate(retrieval, llm, k=k)
    assert llm.prompts == []


def test_latency_counts_from_started_at(retrieval: RetrievalResult):
    started_at = time.perf_counter() - 0.5  # 假裝 answer() 在 0.5 秒前就開始了
    ans = generate(retrieval, FakeLLM(reply()), started_at=started_at)
    assert ans.latency_ms >= 500
