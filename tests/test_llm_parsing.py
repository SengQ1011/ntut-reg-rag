"""parse_llm_output：救得回來的格式要救，救不回來就丟 LLMOutputError。"""

import pytest

from regrag.llm.parsing import LLMOutputError, parse_llm_output

GOOD = '{"answer": "應予輔導。", "citations": [1], "refused": false}'


def test_plain_json():
    out = parse_llm_output(GOOD)
    assert out.answer == "應予輔導。"
    assert out.citations == [1]
    assert out.refused is False


def test_code_fence():
    out = parse_llm_output(f"```json\n{GOOD}\n```")
    assert out.citations == [1]


def test_think_block_removed():
    raw = '<think>想一下 {"answer": "錯的"}</think>\n' + GOOD
    out = parse_llm_output(raw)
    assert out.answer == "應予輔導。"


def test_think_close_tag_only():
    # <think> 在 prompt 模板裡時，輸出只有思考內容加 </think>
    raw = '想一下 {"answer": "錯的"}\n</think>\n' + GOOD
    out = parse_llm_output(raw)
    assert out.answer == "應予輔導。"


def test_surrounding_chatter():
    out = parse_llm_output(f"好的，以下是回答：\n{GOOD}\n希望有幫助！")
    assert out.citations == [1]


@pytest.mark.parametrize(
    "raw",
    [
        "依學則規定應予輔導。",  # 完全沒有 JSON
        '{"answer": "應予輔導。", "citations": [1], "refused": false',  # 少括號
        '{"citations": [1], "refused": false}',  # 缺 answer
        '{"answer": "x", "citations": "1", "refused": false}',  # citations 不是清單
        '{"answer": "x", "citations": [true], "refused": false}',  # 編號不是整數
        '{"answer": "x", "citations": [1], "refused": "no"}',  # refused 不是布林
        '["answer", "citations"]',  # 不是物件
        "",
    ],
)
def test_unparseable_raises_with_raw(raw: str):
    with pytest.raises(LLMOutputError) as exc_info:
        parse_llm_output(raw)
    assert exc_info.value.raw == raw
