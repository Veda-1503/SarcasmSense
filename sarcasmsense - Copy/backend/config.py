import os
import torch

BASE_MODELS = {
    "movies": "distilroberta-base",
    "healthcare": "roberta-base",
    "ecommerce": "LiYuan/amazon-review-sentiment-analysis",
    "restaurants": "distilbert-base-uncased",
    "hotels": "roberta-base",
    "apps": "distilbert-base-uncased"
}

FINE_TUNED_MODELS = {
    "apps":        {"base": "distilbert-base-uncased",                    "path": "models/apps_sarcasm"},
    "movies":      {"base": "distilroberta-base",                         "path": "models/movies_sarcasm"},
    "hotels":      {"base": "roberta-base",                               "path": "models/hotels_sarcasm"},
    "restaurants": {"base": "distilbert-base-uncased",                    "path": "models/restaurants_sarcasm"},
    "ecommerce":   {"base": "LiYuan/amazon-review-sentiment-analysis",    "path": "models/ecommerce_sarcasm"},
    "healthcare":  {"base": "roberta-base",                               "path": "models/healthcare_sarcasm"},
}

SENTIMENT_MODEL_PATH = "models/sentiment_model"
SENTIMENT_BASE_MODEL = "distilbert-base-uncased"

SARCASM_DATASET   = "domain_data/sarcasm_dataset.csv"
SENTIMENT_DATASET = "domain_data/sentiment_dataset.csv"
TEST_DATASET      = "domain_data/test_dataset.csv"

SARCASM_THRESHOLD = 0.65

SARCASM_LEVEL_THRESHOLDS = {
    "Low":    (0.0, 0.4),
    "Medium": (0.4, 0.7),
    "High":   (0.7, 1.0),
}

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_INVERT = {"positive": "negative", "negative": "positive", "neutral": "neutral"}

DOMAINS = list(BASE_MODELS.keys())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_EPOCHS        = 5
TRAIN_BATCH_SIZE    = 16
EVAL_BATCH_SIZE     = 32
LEARNING_RATE       = 2e-5
MAX_SEQ_LEN         = 128
WEIGHT_DECAY        = 0.01
WARMUP_RATIO        = 0.1

MODEL_DISPLAY_NAMES = {
    "apps":        "DistilBERT (Fine-Tuned)",
    "movies":      "DistilRoBERTa (Fine-Tuned)",
    "hotels":      "RoBERTa (Fine-Tuned)",
    "restaurants": "DistilBERT (Fine-Tuned)",
    "ecommerce":   "Amazon Review BERT (Fine-Tuned)",
    "healthcare":  "RoBERTa (Fine-Tuned)",
}