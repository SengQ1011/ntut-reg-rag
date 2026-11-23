"""介面合約。四個模組之間只透過這裡的型別互傳資料。

改動這個檔案要開 PR，四位組員都 review 過才能 merge。
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, HttpUrl


class Chunk(BaseModel):  # ① 產出，一條文 = 一個 chunk
    chunk_id: str  # "{doc_id}#art{條號}"，例如 "ntut-xueze#art12"
    doc_id: str  # 法規代碼，對應 manifest
    doc_title: str  # 法規名稱
    article_no: str  # "第12條"、"第12-1條"
    text: str  # 條文原文，不改寫
    source_url: HttpUrl
    office: str  # 教務處／學務處／資工系
    amended_date: date | None  # 最後修正日期
    applies_to: str | None  # 適用入學學年度，例如 "113學年度起入學"
    fetched_at: datetime
    raw_sha256: str  # 原始檔案的雜湊，對應快照


class Hit(BaseModel):
    chunk_id: str
    score: float
    rank: int  # 從 1 開始


class RetrievalResult(BaseModel):  # ② 產出
    query: str
    retriever: Literal["bm25", "dense", "hybrid"]
    hits: list[Hit]
    latency_ms: float


class Citation(BaseModel):
    chunk_id: str
    doc_title: str
    article_no: str
    source_url: HttpUrl


class Answer(BaseModel):  # ③ 產出
    query: str
    answer: str
    citations: list[Citation]
    refused: bool  # True = 查無相關規定
    model: str  # "qwen3:8b"、"gemini-..."
    retrieval: RetrievalResult
    latency_ms: float


class DevQuery(BaseModel):  # ④ 管理
    qid: str  # "dev-001"
    query: str
    author: str
    source: Literal["self", "faq", "forum"]
    answerable: bool
    note: str = ""
