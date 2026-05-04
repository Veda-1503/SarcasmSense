import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from config import DOMAINS, SARCASM_DATASET
import pandas as pd

MODEL_PATH = "models/domain_classifier.pkl"

DOMAIN_KEYWORDS = {
    "apps": [
        "app", "application", "software", "update", "install", "crash", "bug",
        "interface", "ui", "ux", "download", "feature", "version", "mobile",
        "subscription", "notification", "permission", "battery", "storage", "android", "ios"
    ],
    "movies": [
        "movie", "film", "actor", "director", "plot", "scene", "cast", "cinema",
        "watch", "screen", "script", "acting", "character", "sequel", "theater",
        "streaming", "rating", "genre", "blockbuster", "oscar"
    ],
    "hotels": [
        "hotel", "room", "staff", "checkin", "checkout", "amenity", "breakfast",
        "lobby", "housekeeping", "bed", "pillow", "shower", "pool", "spa",
        "resort", "accommodation", "receptionist", "concierge", "minibar", "stay"
    ],
    "restaurants": [
        "restaurant", "food", "meal", "waiter", "chef", "menu", "dish", "taste",
        "flavor", "service", "ambiance", "portion", "dessert", "appetizer", "dinner",
        "lunch", "breakfast", "reservation", "delivery", "takeout", "cuisine"
    ],
    "ecommerce": [
        "product", "shipping", "delivery", "order", "package", "seller", "price",
        "refund", "return", "quality", "item", "purchase", "buy", "store",
        "marketplace", "checkout", "cart", "discount", "coupon", "review", "amazon"
    ],
    "healthcare": [
        "doctor", "hospital", "patient", "medicine", "treatment", "diagnosis",
        "symptom", "prescription", "clinic", "nurse", "health", "medical",
        "surgery", "pharmacy", "appointment", "therapy", "insurance", "side effect",
        "dosage", "recovery"
    ],
}


def _build_training_data():
    rows = []
    for domain, kws in DOMAIN_KEYWORDS.items():
        for kw in kws:
            rows.append((kw, domain))
            rows.append((f"I tried the {kw} and it was okay", domain))
            rows.append((f"The {kw} here is really great", domain))
            rows.append((f"Terrible {kw} experience overall", domain))
    try:
        df = pd.read_csv(SARCASM_DATASET)
        if "Comment" in df.columns and "Domain" in df.columns:
            for _, row in df.iterrows():
                d = str(row["Domain"]).strip().lower()
                if d in DOMAINS:
                    rows.append((str(row["Comment"]), d))
    except Exception:
        pass
    texts, labels = zip(*rows)
    return list(texts), list(labels)


def train_domain_classifier():
    texts, labels = _build_training_data()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced")),
    ])
    pipeline.fit(texts, labels)
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    return pipeline


def load_domain_classifier():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return train_domain_classifier()


_CLASSIFIER = None


def classify_domain(text: str) -> str:
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = load_domain_classifier()
    pred = _CLASSIFIER.predict([text])[0]
    return pred.lower()