# SarcasmSense

> Multi-stage sarcasm-aware sentiment analysis pipeline using clause-level decomposition and domain-specific transformer models.

---

## Overview

SarcasmSense is a minor project in NLP that tackles a fundamental flaw in standard sentiment analysis: sarcasm is treated as noise rather than addressed upfront. This pipeline detects sarcasm first, decomposes text into clauses, routes to a domain-specific model, and only then produces a sentiment label.

The system achieves **94.46% accuracy** on the sarcasm detection stage (two domains) and **82.3% overall accuracy** across six domains, with a **+5.7% gain** over baselines that skip sarcasm handling.

**Supported domains:** Healthcare, E-commerce, Hotels, Restaurants, Movies, Apps

---

## Pipeline

```
Raw Text Input
      |
      v
  Preprocessing
  (clean, tokenize, sentence-split via spaCy)
      |
      v
  Sarcasm Detection
  (RoBERTa + Gradient Boosting ensemble → Psarc ∈ [0,1])
      |
      v
  Clause-Level Decomposition
  (dependency parsing → Simple / Compound / Complex / Compound-Complex)
      |
      v
  Domain Classification
  (TF-IDF + Logistic Regression → one of 6 domains)
      |
      v
  Domain-Specific Sentiment Model
  (fine-tuned BERT / RoBERTa / DistilBERT per domain)
      |
      v
  Context Fusion + Sarcasm Correction
  (polarity inversion if Psarc > threshold)
      |
      v
  Final Sentiment Label + Confidence
  (positive / negative / neutral)
```

### Sarcasm Correction Formula

```
S_adjusted = S_score * (1 - Psarc) + α * Psarc
```

Where `α` is the sarcasm correction factor and `Psarc` is the soft sarcasm probability carried forward from Stage 2.

---

## Model Architecture

| Domain | Base Model | Type |
|---|---|---|
| Movies | distilroberta-base | DistilRoBERTa (Fine-Tuned) |
| Healthcare | roberta-base | RoBERTa (Fine-Tuned) |
| E-commerce | LiYuan/amazon-review-sentiment-analysis | Amazon Review BERT (Fine-Tuned) |
| Restaurants | distilbert-base-uncased | DistilBERT (Fine-Tuned) |
| Hotels | roberta-base | RoBERTa (Fine-Tuned) |
| Apps | distilbert-base-uncased | DistilBERT (Fine-Tuned) |

**Sarcasm detector:** RoBERTa + Gradient Boosting ensemble (trained on Reddit SARC and news headlines)

**Global sentiment model:** DistilBERT fine-tuned on combined multi-domain dataset

---

## Results

### Sarcasm Detection (Model 1 — two domains)

| Metric | Score |
|---|---|
| Accuracy | 94.46% |

### Multi-Domain Sentiment Pipeline (Model 2 — six domains)

| Metric | Without Sarcasm Handling | With Sarcasm Handling |
|---|---|---|
| Accuracy | 77.0% | 82.7% |
| Precision | 0.794 | 0.824 |
| Recall | 0.816 | 0.859 |
| F1 Score | 0.784 | 0.833 |
| Weighted F1 | 0.765 | 0.826 |

**Performance gain from sarcasm handling: +5.7%**

### Per-Domain Results

| Domain | Model | Samples | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| Movies | DistilRoBERTa | 506,000 | 84.85% | 0.844 | 0.839 |
| Healthcare | RoBERTa | 581,000 | 88.25% | 0.821 | 0.798 |
| E-commerce | Amazon BERT | 558,000 | 89.15% | 0.884 | 0.802 |
| Restaurants | DistilBERT | 510,000 | 90.85% | 0.984 | 0.899 |
| Hotels | RoBERTa | 510,000 | 85.35% | 0.856 | 0.853 |
| Apps | DistilBERT | 579,000 | 77.55% | 0.774 | 0.777 |

---

## Project Structure

```
sarcasmsense/
├── backend/
│   ├── api/
│   │   └── app.py                  # FastAPI application
│   ├── models/                     # Fine-tuned model checkpoints
│   │   ├── apps_sarcasm/
│   │   ├── ecommerce_sarcasm/
│   │   ├── healthcare_sarcasm/
│   │   ├── hotels_sarcasm/
│   │   ├── movies_sarcasm/
│   │   ├── restaurants_sarcasm/
│   │   ├── sentiment_model/
│   │   └── domain_classifier.pkl
│   ├── domain_data/
│   │   ├── sarcasm_dataset.csv
│   │   ├── sentiment_dataset.csv
│   │   └── test_dataset.csv
│   ├── config.py                   # Hyperparameters and model paths
│   ├── preprocessing.py            # Text cleaning and sentence splitting
│   ├── clause_detection.py         # spaCy-based clause classification
│   ├── domain_classifier.py        # TF-IDF + LogReg domain router
│   ├── main_pipeline.py            # End-to-end pipeline orchestration
│   ├── finetune.py                 # Fine-tuning scripts for all models
│   └── evaluation.py              # Evaluation with/without sarcasm metrics
└── Model1/
    ├── Model.ipynb                 # Sarcasm detection model (two-domain)
    └── Testing_Saved_Model.ipynb  # Evaluation notebook
```

---

## API Reference

The backend runs as a FastAPI application.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/analyze` | Analyze a single text input |
| `POST` | `/evaluate` | Batch evaluate a CSV file |

### `/analyze` Request Body

```json
{
  "text": "Oh great, another Monday morning.",
  "domain": "movies"
}
```

### `/analyze` Response

```json
{
  "text": "Oh great, another Monday morning.",
  "domain": "movies",
  "final_sentiment": "negative",
  "final_confidence": 0.914,
  "overall_sarcasm": true,
  "overall_sarcasm_prob": 0.956,
  "model_used": "distilroberta-base",
  "sentence_results": [...]
}
```

### `/evaluate` CSV Format

Required columns: `review`, `domain`, `final_sentiment`

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/Veda-1503/SarcasmSense.git
cd sarcasmsense/backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Fine-tune models (or use pre-trained checkpoints if available)
python finetune.py all

# 5. Start the API server
uvicorn api.app:app --reload --port 8000
```

API will be available at **http://localhost:8000**

---

## Configuration

All settings are in `backend/config.py`:

| Variable | Default | Description |
|---|---|---|
| `SARCASM_THRESHOLD` | 0.65 | Minimum Psarc to trigger polarity inversion |
| `TRAIN_EPOCHS` | 5 | Fine-tuning epochs |
| `TRAIN_BATCH_SIZE` | 16 | Training batch size |
| `LEARNING_RATE` | 2e-5 | AdamW learning rate |
| `MAX_SEQ_LEN` | 128 | Maximum token sequence length |
| `WEIGHT_DECAY` | 0.01 | Optimizer weight decay |

---

## Tech Stack

- **Backend:** Python, FastAPI
- **NLP:** HuggingFace Transformers, spaCy, NLTK
- **Models:** BERT, RoBERTa, DistilBERT, DistilRoBERTa
- **ML:** scikit-learn, PyTorch
- **Sarcasm Ensemble:** RoBERTa + Gradient Boosting

---


---

## License

This project is licensed under the [MIT License](LICENSE).
