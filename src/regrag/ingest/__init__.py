"""① 資料建置。對外只提供 load_chunks。"""

import json
from pathlib import Path

from regrag.schemas import Chunk


def load_chunks(path: str | Path = "data/chunks.jsonl") -> list[Chunk]:
    """讀 chunks.jsonl，每行驗證成 Chunk；格式不符會直接丟 ValidationError。"""
    chunks: list[Chunk] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk.model_validate(json.loads(line)))
    return chunks
