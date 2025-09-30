import os, json, random, time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
)
#config
@dataclass
class CFG:
    # Training corpus (Mody et al.)
    DATA_PATH: str = "/kaggle/input/hate-speech-detection-curated-dataset/HateSpeechDatasetBalanced.csv"
    TEXT_COL: str = "Content"
    LABEL_COL: str = "Label"
    REDDIT_JSONL_PATH: str = "/kaggle/input/annotated-data-20-08/all_anotdata_combined.jsonl" 

    OUTPUT_DIR: str = "./bert_hs_model"
    MODEL_NAME: str = "bert-base-uncased"
    MAX_LEN: int = 256
    BATCH_SIZE: int = 32
    EPOCHS: int = 3
    LR: float = 2e-5
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.06
    SEED: int = 41
    VAL_FRAC: float = 0.1
    USE_EARLY_STOP: bool = True
    EARLY_STOP_PATIENCE: int = 2
    NUM_WORKERS: int = 2
    MAX_SAMPLES: int = 50_000  # cap for training corpus

cfg = CFG()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# Repro
random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.SEED)

print("Loading Mody et al. training corpus …")
t0 = time.time()

# Load & prep Mody et al. (train/val)

df = pd.read_csv(cfg.DATA_PATH, dtype={cfg.TEXT_COL: str})
print(f"Original dataset size: {len(df):,}")
df.columns = [c.strip() for c in df.columns]
df[cfg.TEXT_COL] = df[cfg.TEXT_COL].astype(str).str.strip()
df[cfg.LABEL_COL] = (
    df[cfg.LABEL_COL].astype(str).str.strip().str.lower()
      .map({"0":0, "1":1, "false":0, "true":1, "non-hateful":0, "hateful":1})
)
df = df.dropna(subset=[cfg.TEXT_COL, cfg.LABEL_COL])
df[cfg.LABEL_COL] = df[cfg.LABEL_COL].astype(int)
df = df[df[cfg.TEXT_COL].str.len() > 0].reset_index(drop=True)

if len(df) > cfg.MAX_SAMPLES:
    print(f"Limiting dataset to {cfg.MAX_SAMPLES:,} samples")
    df = df.sample(n=cfg.MAX_SAMPLES, random_state=cfg.SEED).reset_index(drop=True)

print(f"Final dataset size: {len(df):,}")
print("Label distribution:\n", df[cfg.LABEL_COL].value_counts(normalize=True))
print(f"Mody data loaded in {time.time()-t0:.1f}s.")

y = df[cfg.LABEL_COL].values
can_stratify = len(np.unique(y)) > 1
train_df, val_df = train_test_split(
    df, test_size=cfg.VAL_FRAC, random_state=cfg.SEED,
    stratify=y if can_stratify else None
)
print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Stratified: {can_stratify}")

# Tokenizer / HF datasets

tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, use_fast=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

from datasets import Dataset as HFDataset

def to_hf(ds: pd.DataFrame) -> HFDataset:
    return HFDataset.from_pandas(
        ds[[cfg.TEXT_COL, cfg.LABEL_COL]].rename(columns={cfg.TEXT_COL: "text", cfg.LABEL_COL: "labels"})
    )

def tok_fn(batch):
    return tokenizer(
        batch["text"], truncation=True, max_length=cfg.MAX_LEN,
        padding=False, return_attention_mask=True
    )

print("Tokenizing Mody train/val …")
train_hf = to_hf(train_df).map(tok_fn, batched=True, remove_columns=["text"], num_proc=4)
val_hf   = to_hf(val_df).map(tok_fn,   batched=True, remove_columns=["text"], num_proc=4)
print("Tokenization done.")

# Model

model = AutoModelForSequenceClassification.from_pretrained(cfg.MODEL_NAME, num_labels=2).to(device)

def class_weights_from_series(series) -> Optional[torch.Tensor]:
    n0 = int((series==0).sum()); n1 = int((series==1).sum())
    if n0 == 0 or n1 == 0: return None
    N = n0 + n1; w0 = N/(2*n0); w1 = N/(2*n1); s = (w0+w1)/2
    return torch.tensor([w0/s, w1/s], dtype=torch.float)

class_weights = class_weights_from_series(train_df[cfg.LABEL_COL])
print("Class weights:", None if class_weights is None else class_weights.tolist())

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        if labels is not None and labels.dtype != torch.long:
            inputs["labels"] = labels.to(torch.long)
            labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    pm, rm, f1m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "f1_macro": f1m}

args = TrainingArguments(
    output_dir=cfg.OUTPUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=cfg.BATCH_SIZE,
    per_device_eval_batch_size=cfg.BATCH_SIZE * 2,
    learning_rate=cfg.LR,
    weight_decay=cfg.WEIGHT_DECAY,
    num_train_epochs=cfg.EPOCHS,
    warmup_ratio=cfg.WARMUP_RATIO,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
    dataloader_num_workers=cfg.NUM_WORKERS,
    fp16=torch.cuda.is_available(),
    seed=cfg.SEED,
    data_seed=cfg.SEED,
    dataloader_drop_last=False,
    remove_unused_columns=True,
    save_safetensors=True,
)

callbacks = []
if cfg.USE_EARLY_STOP:
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.EARLY_STOP_PATIENCE))

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_hf,
    eval_dataset=val_hf,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
    class_weights=class_weights
)

print("Starting training on Mody's corpus")
trainer.train()
print("Best checkpoint:", trainer.state.best_model_checkpoint)

# Save best
trainer.save_model(cfg.OUTPUT_DIR)
tokenizer.save_pretrained(cfg.OUTPUT_DIR)
with open(os.path.join(cfg.OUTPUT_DIR, "label_mapping.json"), "w") as f:
    json.dump({"0":"non-hateful","1":"hateful"}, f, indent=2)

# In-domain validation report (Mody)
metrics_val = trainer.evaluate(eval_dataset=val_hf)
print("Validation metrics (Mody):", metrics_val)
print(f"Model saved to: {cfg.OUTPUT_DIR}")

# Evaluation on Reddit manual subset

def load_reddit_jsonl_for_hs(path: str) -> pd.DataFrame:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # text
            text = obj.get("reply") or obj.get("reply_body") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            # label
            y = obj.get("HS_label")
            if y is None:
                continue
            try:
                y = int(y)
            except Exception:
                # allow stringy labels "0"/"1"/"true"/"false"
                s = str(y).strip().lower()
                if s in {"1","true","hateful"}:
                    y = 1
                elif s in {"0","false","non-hateful"}:
                    y = 0
                else:
                    continue
            rows.append({"text": text.strip(), "labels": y, "rid": obj.get("reply_id") or obj.get("id") or f"r{i}"})
    return pd.DataFrame(rows)

print("\nLoading Reddit subset for evaluation …")
reddit_df = load_reddit_jsonl_for_hs(cfg.REDDIT_JSONL_PATH)
print(f"Reddit labeled rows: {len(reddit_df):,}")
if len(reddit_df) == 0:
    raise RuntimeError("No usable rows found in Reddit JSONL. Check path/fields: reply/HS_label.")

from datasets import Dataset as HFDataset
reddit_hf = HFDataset.from_pandas(reddit_df[["text","labels","rid"]])
reddit_hf = reddit_hf.map(
    lambda b: tokenizer(b["text"], truncation=True, max_length=cfg.MAX_LEN, padding=False, return_attention_mask=True),
    batched=True, remove_columns=["text"], num_proc=4
)

# Evaluate on Reddit set
print("Evaluating on Reddit labeled subset")
reddit_logits = trainer.predict(reddit_hf).predictions
reddit_labels = np.array(reddit_df["labels"].tolist())
reddit_preds = reddit_logits.argmax(axis=-1)

acc = accuracy_score(reddit_labels, reddit_preds)
p, r, f1, _ = precision_recall_fscore_support(reddit_labels, reddit_preds, average="binary", zero_division=0)
print({
    "reddit_accuracy": acc,
    "reddit_precision": p,
    "reddit_recall": r,
    "reddit_f1": f1
})
print("\nReddit classification report (per-class):\n",
      classification_report(reddit_labels, reddit_preds, digits=4, zero_division=0))

# Save Reddit predictions for error analysis
probs = torch.softmax(torch.tensor(reddit_logits), dim=-1).numpy()[:,1]
out = reddit_df.copy()
out["pred"] = reddit_preds
out["prob_HS1"] = probs
pred_path = os.path.join(cfg.OUTPUT_DIR, "reddit_manual_eval_predictions.csv")
out.to_csv(pred_path, index=False)
print("Saved Reddit predictions to:", pred_path)
