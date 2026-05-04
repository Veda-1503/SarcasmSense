import re
import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def clean_text(text: str) -> str:
    text = text.replace("\u2014", " - ").replace("\u2013", " - ").replace("\u2026", "...")
    text = normalize_text(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\w\s.,!?'\"()\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str) -> list:
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    raw = sentence_endings.split(text)
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) > 3:
            sentences.append(s)
    if not sentences:
        sentences = [text.strip()]
    return sentences


def preprocess(text: str) -> dict:
    cleaned = clean_text(text)
    sentences = split_into_sentences(cleaned)
    return {
        "original": text,
        "cleaned": cleaned,
        "sentences": sentences,
    }