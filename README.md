# ntut-reg-rag

北科大校規問答 RAG（資訊檢索與應用期末專題）。目標：輸入問題，BM25 找出相關條文，本機 LLM 根據條文回答並附條號與原文連結。

## 快速開始

```bash
# 1. 安裝 uv：https://docs.astral.sh/uv/getting-started/installation/
# 2. 安裝套件（會自動裝 Python 3.11）
uv sync
# 3. 跑測試
uv run pytest
```

## 分工與模組

| 模組 | 負責 | 位置 | 對外函式 |
|---|---|---|---|
| ① 資料建置 | 鄭錦鑫 | `src/regrag/ingest/`、`data/` | `load_chunks()` |
| ② 檢索算法 | 林信頤 | `src/regrag/retrieval/` | `search()` |
| ③ LLM 與介面 | 葉凱成 | `src/regrag/llm/`、`src/regrag/api/` | `answer()` |
| ④ 評估與文檔 | 楊子力 | `src/regrag/evaluation/`、`eval/` | — |

介面合約在 `src/regrag/schemas.py`，假資料在 `tests/fixtures/`。還沒拿到別人的真資料前，先用假資料開發。

## 文件

- [第一週 Kickoff 計畫](docs/kickoff.md)：12/1 MVP 的完成條件、每日分工、風險
- [dev set 出題與標註準則](eval/guidelines.md)
- 期末報告、簡報放在組內共用的 Google Drive 資料夾

## 協作規則

- 不直接推 main，一律開 PR；每個 PR 至少一位其他組員 review，作者要能口頭解釋改了什麼。
- 改 `schemas.py` 要四人都 review。
- CI（ruff + pytest）綠燈才能 merge。
- 寫程式的 AI 請先讀 `CLAUDE.md`。

## 評估數字

（MVP 完成後填入：Recall@5、MRR、nDCG@10，並註明資料版本與 dev set 版本）
