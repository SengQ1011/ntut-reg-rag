"""② 校務詞庫（userdict）：自己寫的詞典比對與口語查詢擴展，不依賴 Jieba。

詞庫只影響建索引／查詢用的詞，不會改動 Chunk.text。兩種用途：

1. 校務詞（三字以上）：bigram 只能表示兩字，像「加退選」「二分之一」這種長詞會被拆散。
   斷詞時把文中出現的校務詞整個加成一個額外的詞，讓完整詞比對拿到較高分。
   兩字以下的詞 bigram 已經涵蓋，不重複加，避免同一個詞被算兩次。
2. 口語對照（只用在查詢端）：學生說「被當」，條文寫「不及格」。查詢含左邊的詞時，
   把右邊的校規用詞接在查詢後面再斷詞。條文用詞不同要靠這裡解決，不是改資料。

檔案格式（UTF-8，一行一筆，# 開頭為註解）：
    加退選
    被當 => 不及格
    當兵 => 兵役 服役
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from regrag.retrieval.bm25 import Tokenizer, bigram_tokenize, normalize

SYNONYM_SEP = "=>"
MIN_TERM_LEN = 3  # 兩字以下 bigram 已涵蓋
DEFAULT_PATH = Path(__file__).with_name("userdict.txt")


@dataclass(frozen=True)
class Lexicon:
    terms: frozenset[str] = frozenset()  # 校務詞（normalize 後，皆 >= MIN_TERM_LEN 字）
    # 口語詞 → 校規用詞（皆 normalize 後）
    synonyms: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def max_len(self) -> int:
        return max((len(t) for t in self.terms), default=0)


def parse_lexicon(lines: list[str]) -> Lexicon:
    terms: set[str] = set()
    synonyms: dict[str, tuple[str, ...]] = {}
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if SYNONYM_SEP in line:
            left, right = (s.strip() for s in line.split(SYNONYM_SEP, 1))
            targets = tuple(normalize(t) for t in right.split())
            if not left or not targets:
                raise ValueError(f"詞庫第 {lineno} 行格式錯誤：{raw!r}")
            synonyms[normalize(left)] = synonyms.get(normalize(left), ()) + targets
            # 校規用詞本身若夠長，也當校務詞
            terms.update(t for t in targets if len(t) >= MIN_TERM_LEN)
        elif len(line.split()) != 1:
            raise ValueError(f"詞庫第 {lineno} 行一行只能放一個詞：{raw!r}")
        elif len(normalize(line)) >= MIN_TERM_LEN:
            terms.add(normalize(line))
    return Lexicon(terms=frozenset(terms), synonyms=synonyms)


def load_lexicon(path: str | Path = DEFAULT_PATH) -> Lexicon:
    return parse_lexicon(Path(path).read_text(encoding="utf-8").splitlines())


def find_terms(text: str, lexicon: Lexicon) -> list[str]:
    """回傳文中出現的所有校務詞（可重疊；出現幾次回傳幾次，tf 才正確）。"""
    s = normalize(text)
    found: list[str] = []
    for i in range(len(s)):
        for n in range(MIN_TERM_LEN, min(lexicon.max_len, len(s) - i) + 1):
            if s[i : i + n] in lexicon.terms:
                found.append(s[i : i + n])
    return found


def make_tokenizer(lexicon: Lexicon, base: Tokenizer = bigram_tokenize) -> Tokenizer:
    """基本斷詞 ＋ 文中出現的校務詞。建索引和查詢必須用同一個。"""

    def tokenize(text: str) -> list[str]:
        return base(text) + find_terms(text, lexicon)

    return tokenize


def make_expander(lexicon: Lexicon) -> Callable[[str], str]:
    """查詢含口語詞時，把對應的校規用詞接在後面。只用在查詢端，不用在條文。"""

    def expand(query: str) -> str:
        q = normalize(query)
        extra = [t for word, targets in lexicon.synonyms.items() if word in q for t in targets]
        return " ".join([query, *extra]) if extra else query

    return expand
