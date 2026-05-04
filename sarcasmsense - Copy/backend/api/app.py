import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import tempfile
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from main_pipeline import run_pipeline
from evaluation import run_evaluation, _eval_without_sarcasm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from config import DOMAINS, MODEL_DISPLAY_NAMES

app = FastAPI(title="Sarcasm-Aware Sentiment Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str
    domain: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        result = run_pipeline(req.text.strip(), domain=req.domain)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        required_cols = {"review", "domain", "final_sentiment"}
        if not required_cols.issubset(set(df.columns)):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain columns: {required_cols}. Found: {list(df.columns)}"
            )

        df["domain"] = df["domain"].str.lower().str.strip()
        df["final_sentiment"] = df["final_sentiment"].str.lower().str.strip()

        labels = ["positive", "neutral", "negative"]
        domain_results = []
        all_true_with = []
        all_pred_with = []
        all_true_without = []
        all_pred_without = []

        from main_pipeline import run_pipeline as _rp
        from sentiment_classifier import analyze_sentiment as _sa

        for domain in DOMAINS:
            df_d = df[df["domain"] == domain]
            if len(df_d) == 0:
                continue

            y_true_w = []
            y_pred_w = []
            y_true_wo = []
            y_pred_wo = []

            for _, row in df_d.iterrows():
                review = str(row["review"])
                true_label = str(row["final_sentiment"]).strip().lower()

                # With sarcasm
                result_w = _rp(review, domain=domain)
                pred_w = result_w["final_sentiment"].strip().lower()

                # Without sarcasm
                result_wo = _sa(review)
                pred_wo = result_wo["label"].strip().lower()

                y_true_w.append(true_label)
                y_pred_w.append(pred_w)
                y_true_wo.append(true_label)
                y_pred_wo.append(pred_wo)

            all_true_with.extend(y_true_w)
            all_pred_with.extend(y_pred_w)
            all_true_without.extend(y_true_wo)
            all_pred_without.extend(y_pred_wo)

            domain_results.append({
                "domain": domain,
                "model": MODEL_DISPLAY_NAMES.get(domain, "DistilBERT (Fine-Tuned)"),
                "sample_count": len(df_d),
                "accuracy": round(accuracy_score(y_true_w, y_pred_w), 4),
                "macro_precision": round(precision_score(y_true_w, y_pred_w, labels=labels, average="macro", zero_division=0), 4),
                "macro_recall": round(recall_score(y_true_w, y_pred_w, labels=labels, average="macro", zero_division=0), 4),
                "macro_f1": round(f1_score(y_true_w, y_pred_w, labels=labels, average="macro", zero_division=0), 4),
                "weighted_f1": round(f1_score(y_true_w, y_pred_w, labels=labels, average="weighted", zero_division=0), 4),
                "weighted_precision": round(precision_score(y_true_w, y_pred_w, labels=labels, average="weighted", zero_division=0), 4),
                "weighted_recall": round(recall_score(y_true_w, y_pred_w, labels=labels, average="weighted", zero_division=0), 4),
            })

        # Overall with sarcasm
        overall_acc_w = round(accuracy_score(all_true_with, all_pred_with), 4)
        overall_prec_w = round(precision_score(all_true_with, all_pred_with, labels=labels, average="macro", zero_division=0), 4)
        overall_rec_w = round(recall_score(all_true_with, all_pred_with, labels=labels, average="macro", zero_division=0), 4)
        overall_mf1_w = round(f1_score(all_true_with, all_pred_with, labels=labels, average="macro", zero_division=0), 4)
        overall_wf1_w = round(f1_score(all_true_with, all_pred_with, labels=labels, average="weighted", zero_division=0), 4)

        # Overall without sarcasm
        overall_acc_wo = round(accuracy_score(all_true_without, all_pred_without), 4)
        overall_prec_wo = round(precision_score(all_true_without, all_pred_without, labels=labels, average="macro", zero_division=0), 4)
        overall_rec_wo = round(recall_score(all_true_without, all_pred_without, labels=labels, average="macro", zero_division=0), 4)
        overall_mf1_wo = round(f1_score(all_true_without, all_pred_without, labels=labels, average="macro", zero_division=0), 4)
        overall_wf1_wo = round(f1_score(all_true_without, all_pred_without, labels=labels, average="weighted", zero_division=0), 4)

        # Confusion matrix
        cm = confusion_matrix(all_true_with, all_pred_with, labels=labels)
        cm_list = cm.tolist()

        gain = round(overall_acc_w - overall_acc_wo, 4)

        return {
            "domain_results": domain_results,
            "overall_with_sarcasm": {
                "accuracy": overall_acc_w,
                "macro_precision": overall_prec_w,
                "macro_recall": overall_rec_w,
                "macro_f1": overall_mf1_w,
                "weighted_f1": overall_wf1_w,
            },
            "overall_without_sarcasm": {
                "accuracy": overall_acc_wo,
                "macro_precision": overall_prec_wo,
                "macro_recall": overall_rec_wo,
                "macro_f1": overall_mf1_wo,
                "weighted_f1": overall_wf1_wo,
            },
            "performance_gain": gain,
            "confusion_matrix": {
                "labels": labels,
                "matrix": cm_list,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
