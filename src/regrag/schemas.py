"""介面合約。四個模組之間只透過這裡的型別互傳資料。

改動這個檔案要開 PR，四位組員都 review 過才能 merge。

時間欄位的約定：
- 所有 latency_ms 都是牆鐘時間（time.perf_counter() 前後相減 × 1000），單位毫秒。
- 量測範圍以各欄位註解為準；不同欄位量的區段是巢狀的，不是相加的。
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, HttpUrl, model_validator


class _Contract(BaseModel):
    """所有合約型別的基底。JSON 多出不認識的欄位（例如拼錯字）直接報錯，不會默默丟掉。"""

    model_config = ConfigDict(extra="forbid")


class Chunk(_Contract):
    """① 產出，一條條文 = 一個 chunk。附表、附件 MVP 不收。"""

    # "{doc_id}#art{n}"。n 一律用阿拉伯數字，「之一」寫成 -1：
    # 第12條 → "ntut-xueze#art12"；第12-1條、第十二條之一 → "ntut-xueze#art12-1"
    # ④ 標 qrels 時手寫的也是這個 id，兩邊轉法必須一致。
    chunk_id: str
    doc_id: str  # 法規代碼，對應 data/manifest.csv 的 doc_id
    doc_title: str  # 法規全名，照原文。③ 組引用時顯示
    article_no: str  # 照原文寫法，例如 "第12條"、"第十二條之一"；給人看，機器比對用 chunk_id
    # 條文內容，不含「第N條」標題。不改字：只允許去頭尾空白、把 PDF 排版換行接起來；
    # 不可改字、全半形轉換、刪標點、刪項款編號。用詞不同要靠 ② 的詞庫解決，不是改資料。
    text: str
    source_url: HttpUrl  # 原文網頁或 PDF 連結。③ 組引用時給使用者點
    office: str  # 發布單位，例如 教務處／學務處／資工系。MVP 只給網頁顯示用
    amended_date: date | None  # 文件上寫的最後修正日期；沒寫就填 null（欄位不可省略）
    # 適用入學學年度，自由文字，例如 "113學年度起入學"；無分版本填 null。
    # ③ 放進 prompt 讓 LLM 判斷版本；② 不拿來篩選。
    applies_to: str | None
    # 這條切自哪一份原始快照，等於 data/manifest.csv 的 sha256。
    # 擷取時間等其他快照資訊用它去 manifest 查。
    raw_sha256: str


class Hit(_Contract):
    """② 產出的單筆檢索結果。"""

    chunk_id: str  # 等於 chunk.chunk_id；④ 輸出 TREC run 檔用
    # 越大越相關。尺度依 retriever 而定（BM25 沒有上限），不同 retriever 的分數不可互相比較。
    score: float
    rank: int  # 從 1 開始，必須等於它在 RetrievalResult.hits 中的位置 + 1
    # 命中的整條 chunk，由 ② 從建索引時用的同一份 chunks 帶出。
    # ③ 直接用它組 prompt 與引用，不要自己另外 load_chunks()，以免兩邊版本不一致。
    chunk: Chunk


class RetrievalResult(_Contract):
    """② search() 的回傳值。"""

    query: str  # 原樣回傳呼叫端給的查詢字串
    # MVP 只有 bm25。baseline、調參變體的名稱記在 ④ 的 run 檔，不在這裡。
    retriever: Literal["bm25", "dense", "hybrid"]
    hits: list[Hit]  # 依 rank 由小到大排序，最多 k 筆
    # 只量 search() 本身：從函式被呼叫到回傳，單位毫秒。
    # 若第一次呼叫時才載入索引，載入時間也算在內（所以第一筆會特別慢）。
    # 用途：Answer.latency_ms − 這個值 ≈ 生成那段的時間。
    latency_ms: float


class Citation(_Contract):
    """③ 產出的單筆引用。欄位由 ③ 從對應的 hit.chunk 複製，讓網頁不用回頭翻 hits。"""

    chunk_id: str  # 必須出現在同一個 Answer 的 retrieval.hits 裡（引用只能來自檢索結果）
    doc_title: str
    article_no: str
    source_url: HttpUrl


class Answer(_Contract):
    """③ answer() 的回傳值，也是 request log 每一行的內容。"""

    query: str
    answer: str  # 給使用者看的回答；拒答時放向使用者說明查無規定的文字
    citations: list[Citation]
    # True 只代表「校規裡查無相關規定」。LLM 逾時、輸出解析失敗等系統錯誤一律丟例外，
    # 不可標成 refused，否則拒答率會被灌水。
    # refused=True 時 citations 必須為空；refused=False 時至少要有一筆引用（下方會檢查）。
    refused: bool
    model: str  # "qwen3:8b"、"gemini-..."
    retrieval: RetrievalResult  # 這次回答用的檢索結果，原樣保留
    # 量 answer() 端到端：從函式被呼叫到回傳，單位毫秒。
    # 包含 search()、組 prompt、LLM 生成、解析輸出與引用檢查；
    # 不含 FastAPI／網頁的傳輸時間。kickoff「300 token 回答 20 秒內」的門檻就是看這個值。
    latency_ms: float

    @model_validator(mode="after")
    def _refused_matches_citations(self) -> "Answer":
        if self.refused and self.citations:
            raise ValueError("refused=True 時 citations 必須為空")
        if not self.refused and not self.citations:
            raise ValueError("refused=False 時至少要有一筆引用")
        return self


class DevQuery(_Contract):
    """④ 管理的評估題目（dev set／holdout）。出題規則見 eval/guidelines.md。"""

    qid: str  # ④ 編號，例如 "dev-001"。凍結後不可改號，qrels 與 run 檔都靠它對應
    query: str  # 出題者寫的原句，凍結後不改字
    author: str  # 出題者，雙人標註時用來找人
    source: Literal["self", "faq", "forum"]  # dev set 都是 self；faq／forum 給 holdout 用
    # 題型，對應 guidelines 的五種：直接問／口語問法／要兩條以上／版本相關／無解
    qtype: Literal["direct", "colloquial", "multi", "version", "unanswerable"]
    # false = 校規沒有規定、應拒答，qrels 裡不能有這題。必須和 qtype 一致（下方會檢查）
    answerable: bool
    note: str = ""  # 選填，例如標註時的爭議

    @model_validator(mode="after")
    def _answerable_matches_qtype(self) -> "DevQuery":
        if self.answerable == (self.qtype == "unanswerable"):
            raise ValueError("answerable=False 若且唯若 qtype='unanswerable'")
        return self
