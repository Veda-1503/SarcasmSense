import spacy
from functools import lru_cache

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True
            )
            _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _count_clauses(sent) -> dict:
    verbs = [t for t in sent if t.pos_ == "VERB"]
    subordinators = [
        t for t in sent
        if t.dep_ in ("mark", "advcl", "relcl", "ccomp", "xcomp")
    ]
    coordinators = [
        t for t in sent
        if t.dep_ == "cc" and t.text.lower() in ("and", "but", "or", "nor", "yet", "so")
    ]
    return {
        "verb_count": len(verbs),
        "sub_count": len(subordinators),
        "coord_count": len(coordinators),
    }


def classify_clause(sentence: str) -> str:
    nlp = _get_nlp()
    doc = nlp(sentence)
    sents = list(doc.sents)
    if not sents:
        return "Simple"
    sent = sents[0]
    counts = _count_clauses(sent)
    v = counts["verb_count"]
    s = counts["sub_count"]
    c = counts["coord_count"]
    if v <= 1 and s == 0 and c == 0:
        return "Simple"
    elif s > 0 and c > 0:
        return "Compound-Complex"
    elif s > 0:
        return "Complex"
    elif c > 0 or v > 1:
        return "Compound"
    return "Simple"


def detect_clauses(sentences: list) -> list:
    results = []
    for s in sentences:
        clause_type = classify_clause(s)
        results.append({"sentence": s, "clause_type": clause_type})
    return results
