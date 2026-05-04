import os
import sys
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from config import (
    FINE_TUNED_MODELS, SENTIMENT_BASE_MODEL, SENTIMENT_MODEL_PATH,
    SARCASM_DATASET, SENTIMENT_DATASET, SENTIMENT_LABELS,
    DEVICE, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    LEARNING_RATE, MAX_SEQ_LEN, WEIGHT_DECAY, WARMUP_RATIO, DOMAINS
)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=MAX_SEQ_LEN
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=MAX_SEQ_LEN
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def _eval_epoch(model, loader, num_labels):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            labels = batch["labels"].numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


def finetune_sarcasm_model(domain: str):
    if not os.path.exists(SARCASM_DATASET):
        print(f"[ERROR] Sarcasm dataset not found: {SARCASM_DATASET}")
        return

    df = pd.read_csv(SARCASM_DATASET)
    df["Domain"] = df["Domain"].str.lower().str.strip()
    df_domain = df[df["Domain"] == domain].copy()

    if len(df_domain) < 10:
        print(f"[WARN] Insufficient data for domain '{domain}', using full dataset")
        df_domain = df.copy()

    texts = df_domain["Comment"].astype(str).tolist()

    # ✅ FIXED LINE (ONLY CHANGE)
    labels = (
        df_domain["Sarcasm"]
        .astype(str)
        .str.strip()
        .str.capitalize()
        .map({"Yes": 1, "No": 0})
        .astype(int)
        .tolist()
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, stratify=labels, random_state=42
    )

    cfg = FINE_TUNED_MODELS[domain]
    base_model = cfg["base"]
    save_path = cfg["path"]

    print(f"\n[FINETUNE] Training sarcasm model for domain: {domain}")
    print(f"  Base model: {base_model}")
    print(f"  Train samples: {len(train_texts)} | Val samples: {len(val_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)

    train_ds = SarcasmDataset(train_texts, train_labels, tokenizer)
    val_ds = SarcasmDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * TRAIN_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = 0.0
    for epoch in range(1, TRAIN_EPOCHS + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, scheduler)
        acc, f1 = _eval_epoch(model, val_loader, 2)
        print(f"  Epoch {epoch}/{TRAIN_EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    print(f"  [DONE] Best Val F1: {best_f1:.4f} | Saved to: {save_path}\n")


def finetune_sentiment_model():
    if not os.path.exists(SENTIMENT_DATASET):
        print(f"[ERROR] Sentiment dataset not found: {SENTIMENT_DATASET}")
        return

    df = pd.read_csv(SENTIMENT_DATASET)
    df["Sentiment"] = df["Sentiment"].str.lower().str.strip()
    df = df[df["Sentiment"].isin(SENTIMENT_LABELS)].copy()

    label_map = {lbl: i for i, lbl in enumerate(SENTIMENT_LABELS)}
    texts = df["Comment"].astype(str).tolist()
    labels = [label_map[s] for s in df["Sentiment"]]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, stratify=labels, random_state=42
    )

    print(f"\n[FINETUNE] Training global sentiment model")
    print(f"  Base model: {SENTIMENT_BASE_MODEL}")
    print(f"  Train samples: {len(train_texts)} | Val samples: {len(val_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_BASE_MODEL,
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)

    train_ds = SentimentDataset(train_texts, train_labels, tokenizer)
    val_ds = SentimentDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * TRAIN_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = 0.0
    for epoch in range(1, TRAIN_EPOCHS + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, scheduler)
        acc, f1 = _eval_epoch(model, val_loader, 3)
        print(f"  Epoch {epoch}/{TRAIN_EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(SENTIMENT_MODEL_PATH, exist_ok=True)
            model.save_pretrained(SENTIMENT_MODEL_PATH)
            tokenizer.save_pretrained(SENTIMENT_MODEL_PATH)

    print(f"  [DONE] Best Val F1: {best_f1:.4f} | Saved to: {SENTIMENT_MODEL_PATH}\n")


def finetune_all():
    finetune_sentiment_model()
    for domain in DOMAINS:
        finetune_sarcasm_model(domain)


def finetune_missing():
    finetune_sentiment_model()
    for domain in DOMAINS:
        path = FINE_TUNED_MODELS[domain]["path"]
        if not os.path.isdir(path):
            print(f"[AUTO] Model missing for domain '{domain}'. Fine-tuning now...")
            finetune_sarcasm_model(domain)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        domain_arg = sys.argv[1].lower()
        if domain_arg == "all":
            finetune_all()
        elif domain_arg == "sentiment":
            finetune_sentiment_model()
        elif domain_arg in DOMAINS:
            finetune_sarcasm_model(domain_arg)
        else:
            print(f"[ERROR] Unknown domain: {domain_arg}")
            print(f"  Valid options: all | sentiment | {' | '.join(DOMAINS)}")
    else:
        finetune_all()