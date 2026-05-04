# SarcasmSense — Sarcasm-Aware Sentiment Analysis

A full-stack system for domain-specific sarcasm detection and sentiment analysis using fine-tuned transformer models.

## Architecture

```
project/
├── backend/               # FastAPI backend
│   ├── api/app.py         # REST API endpoints
│   ├── main_pipeline.py   # Full analysis pipeline
│   ├── clause_detection.py
│   ├── domain_classifier.py
│   ├── sarcasm_classifier.py
│   ├── sentiment_classifier.py
│   ├── preprocessing.py
│   ├── finetune.py
│   ├── evaluation.py
│   ├── config.py
│   └── domain_data/       # CSV datasets
│       ├── sarcasm_dataset.csv
│       ├── sentiment_dataset.csv
│       └── test_dataset.csv
│
├── frontend/              # React + Vite + TailwindCSS
│   └── src/
│       ├── App.jsx
│       ├── pages/
│       │   ├── RealTimePage.jsx
│       │   └── EvaluationPage.jsx
│       └── components/
│           ├── SentimentChip.jsx
│           ├── SarcasmBadge.jsx
│           └── ConfidenceBar.jsx
│
├── start_backend.sh
└── start_frontend.sh
```

## Setup & Running

### 1. Fine-tune models (required before first use)

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Fine-tune all models
python finetune.py all
```

### 2. Start the backend

```bash
bash start_backend.sh
# Runs on http://localhost:8000
# API docs: http://localhost:8000/docs
```

### 3. Start the frontend

```bash
bash start_frontend.sh
# Runs on http://localhost:5173
```

## API Endpoints

### `POST /analyze`
```json
{ "text": "Great, the app crashed again as expected." }
```
Returns full pipeline result with sentence-level sarcasm and sentiment analysis.

### `POST /evaluate`
Upload a CSV file with columns: `review`, `domain`, `final_sentiment`  
Returns domain-wise metrics, overall metrics, sarcasm comparison, and confusion matrix.

## Domains Supported
- **Apps** — DistilBERT (Fine-Tuned)
- **Movies** — DistilRoBERTa (Fine-Tuned)
- **Hotels** — RoBERTa (Fine-Tuned)
- **Restaurants** — DistilBERT (Fine-Tuned)
- **Ecommerce** — Amazon Review BERT (Fine-Tuned)
- **Healthcare** — RoBERTa (Fine-Tuned)

## Pipeline Flow
```
Input Review
  → Preprocessing (clean + sentence split)
  → Domain Classification (auto-detect)
  → Per Sentence:
      → Clause Detection (Simple/Compound/Complex/Compound-Complex)
      → Sarcasm Detection (domain-specific model)
      → Sentiment Analysis
      → Sarcasm Adjustment (High→invert, Medium→adjust)
  → Weighted Aggregation
  → Final Sentiment Output
```
