# 第一週 Kickoff 計畫（11/23–12/1 MVP）

> 從 seng 的 kickoff 計畫文件搬過來，讓全組都看得到。之後以這份為準，有變動直接開 PR 改。

## 目標
12/1 的 MVP 只要求一條最細但完整可跑的流程：打開網頁、輸入問題，BM25 從教務處核心校規（10–20 份）找出條文，Qwen3 根據條文回答並附條號和原文連結；同時要有 40 題 dev set 和評估數字。向量檢索、RRF、查詢改寫都留到 12/1 之後。

做法是 walking skeleton：先定好介面（`src/regrag/schemas.py`），11/25 用 5 份法規把整條流程接通，之後四個模組再各自加厚。

## MVP 完成條件（12/1 逐項打勾）
- [ ] `docker compose up` 後瀏覽器能問答，回答附條號與原文連結
- [ ] 查不到相關規定的問題會拒答
- [ ] `data/chunks.jsonl` 涵蓋 MVP 範圍的法規，並通過 schema 驗證與抽樣核對
- [ ] dev set 40 題已標註並凍結，評估腳本能輸出 Recall@5、MRR、nDCG@10
- [ ] BM25 的結果明顯優於 baseline，數字寫在 README
- [ ] CI 綠燈：lint、測試、BM25 對照測試、dev set 回歸測試
- [ ] 不是 ③ 的組員能照 README 從零跑起來

## 每日回報與同步
- 每天 22:00 前在 LINE 群組回三行：今天完成／明天要做／卡在哪。卡住超過半天直接講。
- 每週一次 15 分鐘同步，只看：誰卡住、要砍範圍還是有人接手、有沒有要一起決定的事。

## 每日分工
節奏：11/25（三）接通 skeleton，11/26（四）凍結 dev set，11/28–29 週末緩衝，12/1（二）內部 demo 並打 tag v0.1。時數是「用 AI 寫程式」的預估。

| 日期 | 組員 | 任務 | 產出物 | 完成定義 | 預估時數 |
|---|---|---|---|---|---|
| 11/23 (一) | 全員 | 30 分鐘 kickoff：看完本計畫，確認介面合約 | — | 每人說得出自己 12/1 要交什麼 | 0.5 |
| 11/23 (一) | ③ | 開 repo：資料夾結構、uv 環境、CLAUDE.md、schemas.py、假資料、CI（ruff + pytest） | repo main 分支 | 其他三人 clone 後 uv sync && uv run pytest 通過 | 2 |
| 11/23 (一) | ① | 開始資料盤點，找校網法規彙編入口 | data/inventory.csv 草稿 | 列出 15 份以上候選法規，每份有 URL 與格式 | 1.5 |
| 11/23 (一) | ② | 讀 HW1 倒排索引程式，規劃搬進 retrieval/ | issue 拆分 | 說得出 HW1 的斷詞、索引、計分各在哪 | 1 |
| 11/23 (一) | ④ | 寫標註準則與 dev set 範本，通知每人出 10 題 | eval/guidelines.md、eval/dev_template.jsonl | 組員看完能照著出題 | 1.5 |
| 11/24 (二) | ① | 盤點完成、MVP 範圍定案；先快照並切分 5 份 | data/manifest.csv、chunks.jsonl v0 | 通過 schema 驗證；抽 5 條與原文一致 | 2 |
| 11/24 (二) | ② | Jieba + 倒排索引 + BM25 在假資料上跑通；加 rank_bm25 對照測試 | retrieval/bm25.py、tests/test_bm25.py | 對照測試通過 | 2 |
| 11/24 (二) | ③ | 硬體實測（見「硬體實測」一節）；generate() 吃假檢索結果產出 Answer | docs/hardware.md、llm/ | 達門檻，或決定改用 4B；Answer 通過 schema | 2 |
| 11/24 (二) | ①②③④ | 各出 10 題 dev 題目，不看系統、不看條文改寫 | 交給 ④ | 10 題裡含 1 題無解 | 0.5 |
| 11/24 (二) | ④ | 收題、去重、整理格式 | eval/dev.jsonl | 40 題，格式正確 | 1 |
| 11/25 (三) | ② + ③ | Skeleton 日：兩人一起接通 python -m regrag.ask "問題"，從 chunks v0 檢索後由 Qwen3 回答 | CLI | 3 題都能輸出帶條號的回答 | 各 1.5 |
| 11/25 (三) | ① | 快照並切分剩下的 MVP 法規，處理 PDF | chunks.jsonl v1 草稿 | 80% 以上的 MVP 法規入庫 | 2 |
| 11/25 (三) | ④ | 用 ranx 寫評估腳本，讀 run + qrels 輸出指標 | eval/run_eval.py | 對手造的小 run 檔，和手算結果一致 | 1.5 |
| 11/26 (四) | ① | chunks v1 完成，打 data-v0.1 tag | chunks.jsonl v1、manifest | 交給 ② 抽樣核對 | 1.5 |
| 11/26 (四) | ② | 校務詞庫（userdict）、索引建置腳本、抽樣核對 ① 的 20 條 chunk | userdict.txt、scripts/build_index.py | 一行指令重建索引；20 條核對結果記在 issue | 2 |
| 11/26 (四) | ③ | FastAPI /ask + 引用檢查（引用必須來自檢索結果）+ 暫定拒答規則 | api/ | curl /ask 回傳 Answer JSON；引用檢查有單元測試 | 2 |
| 11/26 (四) | ④ + 出題者 | qrels 雙人標註（出題者 + ④），當天凍結 dev-v1 | eval/dev.qrels | 40 題都有標註；不一致的題目有記錄 | ④ 2，其他人 0.5 |
| 11/27 (五) | ① | 收集測試集素材：官方 FAQ、Dcard／PTT 原文，交給 ④ | 問題清單（附來源連結） | 30 則以上 | 1.5 |
| 11/27 (五) | ② | 跑 dev set：BM25 vs baseline（無詞庫、單字切分） | eval/runs/*.trec + 結果表 | BM25 優於 baseline，數字寫進 README | 1.5 |
| 11/27 (五) | ③ | 簡易網頁（輸入框 + 回答 + 引用連結）+ JSONL request log | api/static/、logs/ | 瀏覽器可問答，點引用能開原文 | 1.5 |
| 11/27 (五) | ④ | CI 加入 dev set 回歸測試（只跑 BM25，不跑 LLM） | .github/workflows/ci.yml | 故意弄壞 BM25 的 PR 會亮紅燈 | 1.5 |
| 11/28–29 (六日) | 全員 | 緩衝：修 bug、補測試、互相 review，不加新功能 | — | 標 MVP 的 issue 都關閉或明確延後 | 1–2 |
| 11/30 (一) | ③ | Docker Compose（Ollama 跑在 host）+ README 啟動步驟 | docker-compose.yml、README | — | 1.5 |
| 11/30 (一) | ① | 重現測試：在自己電腦照 README 從零跑起來 | 回報問題 | 30 分鐘內跑起來 | 1 |
| 11/30 (一) | ② | 失敗題分析，作為 12/8 加向量檢索的依據 | eval/error_analysis.md | 每題失敗原因都有分類 | 1.5 |
| 11/30 (一) | ④ | 人工核對 20 題回答的引用 | eval/citation_check.md | 記錄引用正確率 | 1.5 |
| 12/1 (二) | 全員 | 內部 demo：3 題可答 + 1 題拒答 + 評估數字；打 tag v0.1 | release v0.1 | 摘要裡的完成條件全部打勾 | 1 |
| 12/1 (二) | ④ | 從測試素材挑 10–20 題做 holdout | eval/holdout.jsonl | 題目不拿來調參 | 1 |

每人合計約 13.5–14.5 小時。第一次碰的東西（Docker、Ollama、PDF 解析）實際常會花到兩倍。

## ① 資料盤點：`data/inventory.csv` 欄位

| 欄位 | 說明 | 範例 |
|---|---|---|
| doc_id | 自訂代碼，之後用在 chunk_id | ntut-xueze |
| title | 法規全名，照原文 | 國立臺北科技大學學則 |
| office | 發布單位 | 教務處 |
| url | 原始網頁或 PDF 連結 |  |
| format | html／pdf-text／pdf-scan／docx | pdf-text |
| amended_date | 最後修正日期（文件上寫的） |  |
| applies_to | 是否依入學學年度分版本 | 無／113 起 |
| est_articles | 估計條數 | 60 |
| priority | MVP／later／skip | MVP |
| notes | 表格、附件、掃描、異常格式 | 含學分表格 |

快照方式：原始檔存成 `data/raw/{doc_id}/{YYYYMMDD}.{html|pdf}`，不覆蓋舊檔；`data/manifest.csv` 記 doc_id、url、fetched_at、sha256、bytes、http_status；切分只讀 `data/raw/`，不重新連網。爬取每次請求間隔 1 秒以上。

## ③ 硬體實測門檻（11/24）
3070 Ti 筆電、Ollama `qwen3:8b`（Q4_K_M）、關 thinking mode、插電高效能，prompt 約 2,500 token。

| 項目 | 怎麼量 | 通過門檻 |
|---|---|---|
| 模型全部在 GPU | ollama ps | 顯示 100% GPU |
| VRAM 峰值 | nvidia-smi（context 4096） | 7.5 GB 以下 |
| 首字延遲 | Ollama API 的 prompt_eval_duration | 3 秒以內 |
| 生成速度 | eval_count ÷ eval_duration | 15 token/s 以上 |
| 完整回答 | 問題送出到 300 token 回答結束 | 20 秒以內 |
| 連續負載溫度 | 連續問 20 題，記錄 GPU 溫度與時脈 | 不超過 87°C，最後 5 題速度掉幅不超過 20% |
| 輸出格式 | 要求輸出帶條號的 JSON，跑 20 次 | 19 次以上可解析 |

不過門檻：先改 4B 重測，再不行就把 MVP 的生成切到 Gemini（介面不變，只換 backend）。門檻是經驗值，不是實測數字。

## 驗證方式

| 產出物 | 驗證方式 | 誰驗 | 通過標準 |
|---|---|---|---|
| chunks.jsonl | CI 用 pydantic 逐行驗證；另外人工抽 20 條，對照原始 PDF／網頁 | CI + ② | schema 全過；20 條的條號與內文全部一致 |
| 快照 | 用同一份 raw 重跑切分 | CI | 輸出的 sha256 完全相同 |
| 自寫 BM25 | 和 rank_bm25 對照：同一份斷詞結果、同一組 k1 與 b，在假資料和 dev 查詢上比較 | CI | top-10 排序一致，分數誤差在 1e-6 以內 |
| 檢索品質 | dev set 上比較 BM25 與 baseline（單字切分、不加詞庫） | ② 跑、④ 核 | BM25 的 Recall@5 與 MRR 都高於 baseline，數字記在 README |
| 檢索回歸 | 每個 PR 都跑 dev set（只跑 BM25） | CI | Recall@5 不低於上一版減 0.02 |
| 評估腳本 | 用 3 題手造的 run 和 qrels，與手算指標對照 | CI | 數字一致 |
| 引用 | 自動檢查每個引用的 chunk_id 都在檢索結果裡；另外人工核對 20 題回答與原文 | CI + ④ | 自動檢查 100%；人工核對記錄正確率（MVP 不設門檻） |
| 拒答 | dev set 的 4 題無解題 | ④ | 4 題要拒答 3 題以上，且有解題的誤拒要記錄 |
| 可重現 | 不是 ③ 的組員照 README 從零 clone 並啟動 | ① | 30 分鐘內在瀏覽器問到答案 |
| 硬體 | bench_llm.py 的輸出放進 repo | ③ | 符合上一節的門檻 |

CI 不跑 LLM（runner 沒有 GPU，輸出也不穩定）。rank_bm25 的 IDF 下限處理和課本公式不同，對照前要先對齊（實作時請看原始碼確認）。

## 最可能卡住的風險

| 風險 | 訊號（何時知道） | 備案 |
|---|---|---|
| 1. 資料比預期難抓：法規分散、PDF 是掃描檔或大量表格，「第 X 條」格式不一致 | 11/24 盤點結果；11/25 晚上入庫率低於 80% | MVP 縮到「教務處 HTML + 文字型 PDF」10 份；掃描檔一律延後；關鍵的 1–2 份手動轉文字。dev 題目只出在已入庫的範圍 |
| 2. 整合太晚、AI 寫的程式互相不相容 | 11/25 skeleton 沒跑通；PR 一直卡在衝突 | 介面合約 11/23 定案，改合約要四人同意；skeleton 沒通就先停下各自的新功能，全員修到通為止；CLAUDE.md 統一 AI 的規則；main 分支開保護，CI 綠燈才能 merge |
| 3. 本機 LLM 太慢或過熱 | 11/24 硬體實測不過門檻 | 先換 qwen3:4b；再不行就切 Gemini（同一介面，只換 backend）。demo 當天備好筆電散熱座，並把 Gemini 設成自動備援 |

時間真的不夠時，先砍「版本相關題」和「測試集素材收集」，這兩項不影響 12/1 的完成條件。
