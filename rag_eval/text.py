"""Text normalisation shared by retrieval, checks and validation."""

from __future__ import annotations

import re
import unicodedata

_PUNCT_MAP = str.maketrans(
    {
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        "“": '"', "”": '"', "„": '"', "‟": '"',
        "–": "-", "—": "-", "−": "-", " ": " ",
    }
)

_CONTRACTIONS = [
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bcannot\b"), "can not"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'m\b"), " am"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"'s\b"), ""),
]

NEGATORS = frozenset(
    {"not", "no", "never", "nobody", "nothing", "none", "neither", "nor", "without", "unable"}
)

# Modal verbs and negators are deliberately absent: "can see" and "can not see" must differ.
STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be because been before being
    below between both by did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me
    more most my myself of off on once only or other our ours ourselves out over own same she so
    some such than that the their theirs them themselves then there these they this those
    through to too under until up very was we were what when where which while who whom why with
    you your yours yourself yourselves us let get got make sure please thanks thank sorry great
    hi hello glad happy question problem okay ok yes
    """.split()
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r"[,;:]|\b(?:but|however|although)\b")
_NUM_SEP = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TOKEN = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, ASCII quotes, expanded contractions, no thousands separators."""
    text = unicodedata.normalize("NFKC", text).translate(_PUNCT_MAP).lower()
    for pattern, repl in _CONTRACTIONS:
        text = pattern.sub(repl, text)
    text = _NUM_SEP.sub("", text)
    return text


def stem(token: str) -> str:
    """Crude suffix stripping; only has to make 'reviews' meet 'review'."""
    if token.isdigit():
        return token
    for suffix in ("ies", "es", "s", "ed", "ing"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    # A trailing 'e' is dropped so 'approve', 'approved' and 'approves' meet.
    return token[:-1] if len(token) > 3 and token.endswith("e") else token


def tokens(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Normalised, stemmed tokens in order."""
    raw = _TOKEN.findall(normalize(text))
    if not keep_stopwords:
        raw = [t for t in raw if t not in STOPWORDS]
    return [stem(t) for t in raw]


def content_tokens(text: str) -> list[str]:
    """Tokens that carry meaning: no stopwords, no negators."""
    return [t for t in tokens(text) if t not in NEGATORS]


def is_negated(text: str) -> bool:
    return any(t in NEGATORS for t in _TOKEN.findall(normalize(text)))


def numbers(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(normalize(text)) if t.isdigit()}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def clauses(sentence: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(sentence) if c and c.strip()]
