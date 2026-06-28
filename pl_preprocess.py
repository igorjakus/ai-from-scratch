import string

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


def _ensure_punkt() -> None:
    for name in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{name}")
            return
        except LookupError:
            continue
    nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt", quiet=True)


def tokenize_words(text: str) -> list[str]:
    _ensure_punkt()
    text = text.lower().replace("\xa0", " ")
    out: list[str] = []
    punct_strip = string.punctuation + "„”«»…–—"
    for sent in sent_tokenize(text, language="polish"):
        for tok in word_tokenize(sent, language="polish"):
            w = tok.strip(punct_strip)
            if w and any(ch.isalnum() for ch in w):
                out.append(w)
    return out


def words_from_file(path: str, encoding: str = "utf-8") -> list[str]:
    with open(path, encoding=encoding) as f:
        return tokenize_words(f.read())
