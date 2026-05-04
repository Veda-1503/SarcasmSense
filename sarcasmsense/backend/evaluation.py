import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from config import DOMAINS, MODEL_DISPLAY_NAMES, TEST_DATASET
from main_pipeline import run_pipeline


def _eval_domain(df_domain: pd.DataFrame, domain: str) -> dict:
    y_true = []
    y_pred = []
    for _, row in df_domain.iterrows():
        review = str(row["review"])
        true_label = str(row["final_sentiment"]).strip().lower()
        result = run_pipeline(review, domain=domain)
        pred_label = result["final_sentiment"].strip().lower()
        y_true.append(true_label)
        y_pred.append(pred_label)
    labels = ["positive", "neutral", "negative"]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    mf1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    wf1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    wp = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    wr = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        "domain": domain,
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": mf1,
        "weighted_f1": wf1,
        "weighted_precision": wp,
        "weighted_recall": wr,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _eval_without_sarcasm(df: pd.DataFrame) -> float:
    from sentiment_classifier import analyze_sentiment
    y_true = []
    y_pred = []
    for _, row in df.iterrows():
        review = str(row["review"])
        true_label = str(row["final_sentiment"]).strip().lower()
        result = analyze_sentiment(review)
        y_true.append(true_label)
        y_pred.append(result["label"])
    return accuracy_score(y_true, y_pred)


def run_evaluation(test_path: str = TEST_DATASET):
    if not os.path.exists(test_path):
        print(f"[ERROR] Test dataset not found: {test_path}")
        sys.exit(1)

    df = pd.read_csv(test_path)
    df["domain"] = df["domain"].str.lower().str.strip()
    df["final_sentiment"] = df["final_sentiment"].str.lower().str.strip()

    print("\n" + "=" * 120)
    print("📊 DOMAIN-WISE PERFORMANCE")
    print("=" * 120)
    header = f"{'Domain':<15} {'Model Used':<35} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'Macro F1':>10} {'Weighted F1':>12}"
    print(header)
    print("-" * 120)

    all_true = []
    all_pred = []
    domain_results = {}

    for domain in DOMAINS:
        df_d = df[df["domain"] == domain]
        if len(df_d) == 0:
            continue
        res = _eval_domain(df_d, domain)
        domain_results[domain] = res
        all_true.extend(res["y_true"])
        all_pred.extend(res["y_pred"])
        model_name = MODEL_DISPLAY_NAMES.get(domain, "DistilBERT (Fine-Tuned)")
        domain_cap = domain.capitalize()
        print(
            f"{domain_cap:<15} {model_name:<35} "
            f"{res['accuracy']*100:>6.1f}% "
            f"{res['macro_precision']:>7.2f} "
            f"{res['macro_recall']:>7.2f} "
            f"{res['macro_f1']:>10.2f} "
            f"{res['weighted_f1']:>12.2f}"
        )

    labels = ["positive", "neutral", "negative"]
    overall_acc = accuracy_score(all_true, all_pred)
    overall_mp = precision_score(all_true, all_pred, labels=labels, average="macro", zero_division=0)
    overall_mr = recall_score(all_true, all_pred, labels=labels, average="macro", zero_division=0)
    overall_mf1 = f1_score(all_true, all_pred, labels=labels, average="macro", zero_division=0)
    overall_wp = precision_score(all_true, all_pred, labels=labels, average="weighted", zero_division=0)
    overall_wr = recall_score(all_true, all_pred, labels=labels, average="weighted", zero_division=0)
    overall_wf1 = f1_score(all_true, all_pred, labels=labels, average="weighted", zero_division=0)

    print("\n" + "=" * 120)
    print("📈 OVERALL PERFORMANCE")
    print("=" * 120)
    print(f"{'Accuracy':<30}: {overall_acc*100:.1f}%")
    print(f"{'Macro Precision':<30}: {overall_mp:.2f}")
    print(f"{'Macro Recall':<30}: {overall_mr:.2f}")
    print(f"{'Macro F1-score':<30}: {overall_mf1:.2f}")
    print(f"{'Weighted Precision':<30}: {overall_wp:.2f}")
    print(f"{'Weighted Recall':<30}: {overall_wr:.2f}")
    print(f"{'Weighted F1-score':<30}: {overall_wf1:.2f}")

    acc_no_sarcasm = _eval_without_sarcasm(df)
    gain = overall_acc - acc_no_sarcasm

    print("\n" + "=" * 120)
    print("SARCASM IMPACT ANALYSIS")
    print("=" * 120)
    print(f"{'Without Sarcasm Handling':<30}: {acc_no_sarcasm*100:.1f}%")
    print(f"{'With Sarcasm Handling':<30}: {overall_acc*100:.1f}%")
    sign = "+" if gain >= 0 else ""
    print(f"{'Performance Gain':<30}: {sign}{gain*100:.1f}%")

    cm = confusion_matrix(all_true, all_pred, labels=labels)
    print("\n" + "=" * 120)
    print("📉 CONFUSION MATRIX")
    print("=" * 120)
    print(f"\n           Predicted")
    print(f"         {'Pos':>5} {'Neu':>5} {'Neg':>5}\n")
    row_labels = ["Actual Pos", "Actual Neu", "Actual Neg"]
    for i, row_label in enumerate(row_labels):
        row = cm[i] if i < len(cm) else [0, 0, 0]
        if len(row) < 3:
            row = list(row) + [0] * (3 - len(row))
        print(f"{row_label}  {row[0]:>5} {row[1]:>5} {row[2]:>5}")

    print("\n" + "=" * 120)
    return {
        "overall_accuracy": overall_acc,
        "macro_f1": overall_mf1,
        "weighted_f1": overall_wf1,
        "domain_results": domain_results,
    }