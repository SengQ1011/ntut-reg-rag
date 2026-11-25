"""build_prompt：條文依 rank 編號放進 prompt，原文一字不改。"""

from collections.abc import Callable

from regrag.llm.prompt import build_prompt
from regrag.schemas import Chunk, RetrievalResult


def test_prompt_contains_query(retrieval: RetrievalResult):
    prompt = build_prompt(retrieval.query, retrieval.hits)
    assert retrieval.query in prompt


def test_prompt_contains_each_article_verbatim(retrieval: RetrievalResult):
    prompt = build_prompt(retrieval.query, retrieval.hits)
    for hit in retrieval.hits:
        assert hit.chunk.text in prompt
        assert hit.chunk.doc_title in prompt
        assert hit.chunk.article_no in prompt


def test_prompt_numbers_follow_rank_order(retrieval: RetrievalResult):
    prompt = build_prompt(retrieval.query, retrieval.hits)
    first, second = retrieval.hits
    pos1 = prompt.index("[1]")
    pos2 = prompt.index("[2]")
    assert pos1 < prompt.index(first.chunk.text) < pos2 < prompt.index(second.chunk.text)


def test_prompt_does_not_expose_chunk_ids(retrieval: RetrievalResult):
    # LLM 只回編號，不需要看到 chunk_id；放進去反而會讓它照抄 id
    prompt = build_prompt(retrieval.query, retrieval.hits)
    for hit in retrieval.hits:
        assert hit.chunk_id not in prompt


def test_prompt_includes_applies_to_when_present(
    fake_chunks: list[Chunk],
    make_retrieval: Callable[[str, list[Chunk]], RetrievalResult],
):
    versioned = fake_chunks[0].model_copy(update={"applies_to": "113學年度起入學"})
    r = make_retrieval("問題", [versioned])
    prompt = build_prompt(r.query, r.hits)
    assert "113學年度起入學" in prompt


def test_prompt_asks_for_json_fields(retrieval: RetrievalResult):
    prompt = build_prompt(retrieval.query, retrieval.hits)
    for field in ('"answer"', '"citations"', '"refused"'):
        assert field in prompt
