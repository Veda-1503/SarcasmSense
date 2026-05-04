import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import (
    SENTIMENT_MODEL_PATH, SENTIMENT_BASE_MODEL,
    SENTIMENT_LABELS, SENTIMENT_INVERT, DEVICE, MAX_SEQ_LEN
)

_SENTIMENT_MODEL = None
_SENTIMENT_TOKENIZER = None


def _load_sentiment_model():
    global _SENTIMENT_MODEL, _SENTIMENT_TOKENIZER
    if _SENTIMENT_MODEL is None:
        load_path = SENTIMENT_MODEL_PATH if os.path.isdir(SENTIMENT_MODEL_PATH) else SENTIMENT_BASE_MODEL
        _SENTIMENT_TOKENIZER = AutoTokenizer.from_pretrained(load_path)
        _SENTIMENT_MODEL = AutoModelForSequenceClassification.from_pretrained(
            load_path,
            num_labels=3,
            ignore_mismatched_sizes=True
        )
        _SENTIMENT_MODEL.to(DEVICE)
        _SENTIMENT_MODEL.eval()
    return _SENTIMENT_TOKENIZER, _SENTIMENT_MODEL


def analyze_sentiment(text: str) -> dict:
    tokenizer, model = _load_sentiment_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_SEQ_LEN
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    label_idx = int(np.argmax(probs))
    label = SENTIMENT_LABELS[label_idx]
    confidence = float(probs[label_idx])
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "scores": {SENTIMENT_LABELS[i]: round(float(probs[i]), 4) for i in range(3)},
    }


def analyze_sentiment_batch(texts: list) -> list:
    tokenizer, model = _load_sentiment_model()
    results = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_SEQ_LEN
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        for p in probs:
            label_idx = int(np.argmax(p))
            label = SENTIMENT_LABELS[label_idx]
            confidence = float(p[label_idx])

            if probs[2] > 0.4 and probs[0] > 0.3:
                label = "negative"
                
            results.append({
                "label": label,
                "confidence": round(confidence, 4),
                "scores": {SENTIMENT_LABELS[j]: round(float(p[j]), 4) for j in range(3)},
            })
    return results


def invert_sentiment(label: str) -> str:
    return SENTIMENT_INVERT.get(label, label)