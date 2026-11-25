"""組 prompt。條文依檢索 rank 標成 [1]、[2]…，LLM 引用時只回編號。"""

from regrag.schemas import Hit

_INSTRUCTION = "你是國立臺北科技大學的校規問答助理。只能根據下面提供的條文回答問題。"

_RULES = """\
## 回答規則
1. 只根據上面的條文回答，不可使用條文以外的知識，用繁體中文回答。
2. 把回答用到的條文編號放進 citations，例如 [1, 3]；只能用上面出現過的編號。
3. 如果上面的條文都無法回答這個問題，refused 設為 true，citations 留空。
4. 只輸出一個 JSON 物件，不要輸出任何其他文字，格式如下：
{"answer": "回答內容", "citations": [1], "refused": false}"""


def _format_article(number: int, hit: Hit) -> str:
    c = hit.chunk
    header = f"[{number}] {c.doc_title} {c.article_no}"
    if c.applies_to:
        header += f"（適用：{c.applies_to}）"
    return f"{header}\n{c.text}"


def build_prompt(query: str, hits: list[Hit]) -> str:
    """條文原文照放、不改字；編號從 1 開始，順序同 hits。"""
    articles = "\n\n".join(_format_article(i, hit) for i, hit in enumerate(hits, start=1))
    return f"{_INSTRUCTION}\n\n## 條文\n{articles}\n\n## 問題\n{query}\n\n{_RULES}"
