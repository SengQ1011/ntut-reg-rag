"""② 檢索器設定與索引存取。

設定（preset）：
- bm25：正式版。bigram ＋ 校務詞庫 ＋ 口語查詢擴展。search() 預設用這個。
- baseline：kickoff 規定的對照組。漢字單字切分、不加詞庫。

索引存成 JSON（index/{preset}.json），附「指紋」：chunks 內容、詞庫內容、斷詞設定、k1、b
任何一項變了，指紋就不同，load_or_build() 會改為現場重建，不會讀到過期的索引。
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from regrag.retrieval.bm25 import (
    K1,
    B,
    BM25Retriever,
    InvertedIndex,
    QueryExpander,
    Tokenizer,
    bigram_tokenize,
    char_tokenize,
)
from regrag.retrieval.lexicon import DEFAULT_PATH, load_lexicon, make_expander, make_tokenizer
from regrag.schemas import Chunk

INDEX_FORMAT = 1


@dataclass(frozen=True)
class Preset:
    name: str
    tokenizer: Tokenizer
    query_expander: QueryExpander | None
    # 影響索引內容的設定，寫進指紋（函式本身無法比較，所以用文字描述＋詞庫原文）
    signature: str


def make_preset(name: str, userdict: str | Path = DEFAULT_PATH) -> Preset:
    if name == "baseline":
        return Preset("baseline", char_tokenize, None, "char")
    if name == "bm25":
        path = Path(userdict)
        lexicon = load_lexicon(path)
        sig = "bigram+userdict:" + hashlib.sha256(path.read_bytes()).hexdigest()
        return Preset("bm25", make_tokenizer(lexicon, bigram_tokenize), make_expander(lexicon), sig)
    raise ValueError(f"未知的 preset：{name}（可用：{', '.join(PRESET_NAMES)}）")


PRESET_NAMES = ("bm25", "baseline")


def fingerprint(preset: Preset, chunks: Sequence[Chunk], k1: float, b: float) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([INDEX_FORMAT, preset.signature, k1, b], ensure_ascii=False).encode())
    for c in chunks:
        h.update(json.dumps([c.chunk_id, c.text], ensure_ascii=False).encode())
    return h.hexdigest()


def build(name: str, chunks: Sequence[Chunk], k1: float = K1, b: float = B) -> BM25Retriever:
    p = make_preset(name)
    return BM25Retriever(chunks, p.tokenizer, k1, b, query_expander=p.query_expander)


def index_path(index_dir: str | Path, name: str) -> Path:
    return Path(index_dir) / f"{name}.json"


def save_index(
    retriever: BM25Retriever, name: str, index_dir: str | Path, k1: float = K1, b: float = B
) -> Path:
    preset = make_preset(name)
    data = {
        "format": INDEX_FORMAT,
        "preset": name,
        "fingerprint": fingerprint(preset, retriever.chunks, k1, b),
        "chunk_ids": [c.chunk_id for c in retriever.chunks],
        "doc_lens": retriever.index.doc_lens,
        "postings": retriever.index.postings,
    }
    path = index_path(index_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # 先寫暫存檔再改名，中途中斷也不會留下半個檔案
    return path


def load_index(
    name: str, chunks: Sequence[Chunk], index_dir: str | Path, k1: float = K1, b: float = B
) -> InvertedIndex | None:
    """指紋相符才回傳索引；檔案不存在、格式不對或過期都回傳 None。"""
    path = index_path(index_dir, name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("fingerprint") != fingerprint(make_preset(name), chunks, k1, b):
        return None
    postings = {t: [(d, tf) for d, tf in p] for t, p in data["postings"].items()}
    return InvertedIndex(postings=postings, doc_lens=list(data["doc_lens"]))


def load_or_build(
    name: str,
    chunks: Sequence[Chunk],
    index_dir: str | Path | None = None,
    k1: float = K1,
    b: float = B,
) -> BM25Retriever:
    p = make_preset(name)
    index = load_index(name, chunks, index_dir, k1, b) if index_dir is not None else None
    return BM25Retriever(chunks, p.tokenizer, k1, b, query_expander=p.query_expander, index=index)
