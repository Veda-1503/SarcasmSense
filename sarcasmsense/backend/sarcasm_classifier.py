import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import (
    FINE_TUNED_MODELS, DEVICE, MAX_SEQ_LEN,
    SARCASM_THRESHOLD, SARCASM_LEVEL_THRESHOLDS
)

_SARCASM_CACHE = {}


def _load_model(domain: str):
    if domain in _SARCASM_CACHE:
        return _SARCASM_CACHE[domain]
    cfg = FINE_TUNED_MODELS[domain]
    model_path = cfg["path"]
    base_model = cfg["base"]
    load_path = model_path if os.path.isdir(model_path) else base_model
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        load_path,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)
    model.eval()
    _SARCASM_CACHE[domain] = (tokenizer, model)
    return tokenizer, model


def get_sarcasm_level(confidence: float) -> str:
    for level, (lo, hi) in SARCASM_LEVEL_THRESHOLDS.items():
        if lo <= confidence < hi:
            return level
    return "High"


def detect_sarcasm(text: str, domain: str) -> dict:
    if domain not in FINE_TUNED_MODELS:
        domain = "apps"
    tokenizer, model = _load_model(domain)
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
    sarcasm_prob = float(probs[1])
    not_sarcasm_prob = float(probs[0])
    is_sarcasm = sarcasm_prob >= SARCASM_THRESHOLD
    confidence = sarcasm_prob if is_sarcasm else not_sarcasm_prob
    level = get_sarcasm_level(sarcasm_prob)
    return {
        "is_sarcasm": is_sarcasm,
        "label": "Yes" if is_sarcasm else "No",
        "confidence": round(confidence, 4),
        "sarcasm_prob": round(sarcasm_prob, 4),
        "level": level,
    }


def detect_sarcasm_batch(texts: list, domain: str) -> list:
    if domain not in FINE_TUNED_MODELS:
        domain = "apps"
    tokenizer, model = _load_model(domain)
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
            sarcasm_prob = float(p[1])
            is_sarcasm = sarcasm_prob >= SARCASM_THRESHOLD
            confidence = sarcasm_prob if is_sarcasm else float(p[0])
            level = get_sarcasm_level(sarcasm_prob)
            results.append({
                "is_sarcasm": is_sarcasm,
                "label": "Yes" if is_sarcasm else "No",
                "confidence": round(confidence, 4),
                "sarcasm_prob": round(sarcasm_prob, 4),
                "level": level,
            })
    return results