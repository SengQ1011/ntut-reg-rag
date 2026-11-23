# ntut-reg-rag
北科大校規問答 RAG。Python 3.11，用 uv 管理套件。

## 規則
- 模組之間只透過 src/regrag/schemas.py 的型別與 load_chunks / search / answer 互叫：
  - ① `regrag.ingest.load_chunks(path) -> list[Chunk]`
  - ② `regrag.retrieval.search(query, k=10) -> RetrievalResult`
  - ③ `regrag.llm.answer(query, k=5, backend="ollama") -> Answer`
- 不要修改 schemas.py，除非任務明說要改合約。改合約要開 PR，四人都 review 過才能 merge。
- 檢索（斷詞、倒排索引、BM25）必須自己寫，不可用 rank_bm25、Pyserini、LangChain、
  LlamaIndex 替代。rank_bm25 只能出現在 tests/ 當對照。
- 不要改寫條文原文；Chunk.text 必須和原始檔一致。
- 任何新函式都要有 pytest，測試用 tests/fixtures 的假資料，不要連網、不要呼叫 LLM。
- 假資料的 doc_id 一律以 `fake-` 開頭，不要混進 data/。
- 不要讀取或依賴 eval/test（封存測試集）。
- 不要把 API key 寫進程式，一律讀環境變數（.env 已在 .gitignore）。
- 交付前執行：uv run ruff check . && uv run ruff format . && uv run pytest

## 常用指令（標「未實作」的等對應模組完成）
- 安裝：uv sync
- 測試：uv run pytest
- 重建索引（未實作）：uv run python scripts/build_index.py
- 問答（未實作）：uv run python -m regrag.ask "問題"
- 評估（未實作）：uv run python -m regrag.evaluation.run_eval --run eval/runs/xxx.trec

## 程式風格
- 型別標註、函式短小、中文註解可以。不確定的地方直接問，不要自己猜。
