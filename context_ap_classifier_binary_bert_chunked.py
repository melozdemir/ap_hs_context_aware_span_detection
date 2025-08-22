
import os, json, time, random, math
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, precision_recall_curve
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
try:
    from transformers.optimization import get_linear_schedule_with_warmup
except Exception:
    get_linear_schedule_with_warmup = None

#configurations

@dataclass
class CFG:
    data_path: str = "/kaggle/input/annotated-data-20-08/all_anotdata_combined.jsonl"
    output_dir: str = "/kaggle/working/ap_ctx_pytorch_plus"
    model_name: str = "bert-base-uncased"
    seed: int = 42

    # lengths & batching
    max_len: int = 384
    reply_min_tokens: int = 128       
    batch_size: int = 8
    grad_accum_steps: int = 1

    # training
    epochs: int = 6
    lr: float = 2e-5
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    num_workers: int = 2
    weight_classes: bool = True

    # scheduler
    use_scheduler: bool = True
    warmup_ratio: float = 0.06

    # losses
    use_focal: bool = True
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0

    # context aware + chunking
    context_mode: str = "none"         # "pair" to use context; "none" reply-only
    context_dropout: float = 0.15
    chunking: bool = True              # cover full context via chunks
    ctx_stride: int = 64             
    agg: str = "max"                

    # calibration + thresholding
    do_temp_scaling: bool = True
    threshold_mode: str = "max_f1"
    target_precision: float = 0.60

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(CFG.seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

#data input/output
def read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "reply" not in obj or "AP_label" not in obj:
                continue
            meta = obj.get("meta") or {}
            items.append({
                "reply": obj["reply"],
                "context": obj.get("context") or "",
                "label": int(obj["AP_label"]),
                "id": str(obj.get("id", i)),  # ensure consistent type
                "post_id": str(meta.get("post_id") or f"noid_{i}"),
            })
    return items

def grouped_split(items: List[Dict], seed=42, train=0.7, val=0.15):
    groups = [it["post_id"] for it in items]
    idx = np.arange(len(items))
    gss = GroupShuffleSplit(n_splits=1, train_size=train, random_state=seed)
    tr, tmp = next(gss.split(idx, groups=groups))
    remain = 1.0 - train
    val_frac = val / remain
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_frac, random_state=seed)
    dv_rel, te_rel = next(gss2.split(tmp, groups=[groups[i] for i in tmp]))
    dv = np.array(tmp[dv_rel]); te = np.array(tmp[te_rel])
    return np.array(tr), dv, te

def subset(items, idxs): return [items[i] for i in idxs]

def show_stats(name, rows):
    c = Counter(r["label"] for r in rows)
    pos = c.get(1,0); n = max(1,len(rows))
    print(f"{name}: n={len(rows)} | AP=1 {pos} ({pos/n:.1%}) | groups={len(set(r['post_id'] for r in rows))}")

#tokenizer

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)

def _token_ids(text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]

def _make_ctx_chunks(ids: List[int], win: int, stride: int) -> List[List[int]]:
    if win <= 0 or len(ids) <= win:
        return [ids[:win]]
    step = max(1, win - stride)
    chunks = [ids[i:i+win] for i in range(0, len(ids), step)]
    if len(chunks[-1]) < win and len(ids) > win:
        chunks[-1] = ids[-win:]  # ensure tail coverage
    return chunks

def expand_items_with_chunks(items: List[Dict]) -> List[Dict]:
    out = []
    for it in items:
        reply = it["reply"]; ctx = (it.get("context") or "").strip()
        r_ids = _token_ids(reply)

        if CFG.context_mode != "pair" or not ctx:
            # reply-only
            budget = CFG.max_len - 2  # [CLS],[SEP]
            r_keep = min(len(r_ids), budget)
            enc = tokenizer.prepare_for_model(ids=r_ids[:r_keep], add_special_tokens=True)
            out.append({"enc": enc, "label": it["label"], "orig_id": it["id"], "post_id": it["post_id"]})
            continue

        # with context
        c_ids = _token_ids(ctx)
        budget = CFG.max_len - 3  # [CLS],[SEP],[SEP]
        r_keep = min(len(r_ids), max(CFG.reply_min_tokens, min(len(r_ids), budget)))
        r_keep = min(r_keep, budget)
        ctx_budget = max(0, budget - r_keep)

        if not CFG.chunking or len(c_ids) <= ctx_budget:
            enc = tokenizer.prepare_for_model(ids=r_ids[:r_keep], pair_ids=c_ids[:ctx_budget], add_special_tokens=True)
            out.append({"enc": enc, "label": it["label"], "orig_id": it["id"], "post_id": it["post_id"]})
            continue

        # chunk the context
        for ch in _make_ctx_chunks(c_ids, ctx_budget, CFG.ctx_stride):
            enc = tokenizer.prepare_for_model(ids=r_ids[:r_keep], pair_ids=ch, add_special_tokens=True)
            out.append({"enc": enc, "label": it["label"], "orig_id": it["id"], "post_id": it["post_id"]})
    return out

#dataset
class ChunkedDataset(Dataset):
    def __init__(self, samples: List[Dict], training: bool = False):
        self.samples = samples
        self.training = training

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        item = {k: s["enc"][k] for k in s["enc"] if k in ("input_ids","token_type_ids","attention_mask")}
        item["labels"] = int(s["label"])
        item["orig_id"] = s["orig_id"]
        item["post_id"] = s["post_id"]
        # (Optional) context dropout for robustness could be added here if needed.
        return item

def collate_fn_chunked(features):
    labels = torch.tensor([f.pop("labels") for f in features], dtype=torch.long)
    orig_ids = [f.pop("orig_id") for f in features]
    post_ids = [f.pop("post_id") for f in features]
    batch = tokenizer.pad(features, padding=True, return_tensors="pt")
    batch["labels"] = labels
    batch["orig_id"] = orig_ids
    batch["post_id"] = post_ids
    return batch

#loss functions
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha: Optional[torch.Tensor]=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        p_t = torch.softmax(logits, dim=-1)[torch.arange(len(targets), device=logits.device), targets]
        loss = ((1 - p_t) ** self.gamma) * ce
        return loss.mean()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    def forward(self, logits, targets):
        n_class = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (n_class - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        if self.weight is not None:
            w = self.weight.to(logits.device).unsqueeze(0)
            log_probs = log_probs * w
        return -(true_dist * log_probs).sum(dim=1).mean()

def class_weights_from(rows: List[Dict]) -> Optional[torch.Tensor]:
    y = np.array([x["label"] for x in rows])
    n0, n1 = (y==0).sum(), (y==1).sum()
    if n0 == 0 or n1 == 0: return None
    N = len(y); w0 = N/(2*n0); w1 = N/(2*n1)
    s = (w0 + w1) / 2.0
    return torch.tensor([w0/s, w1/s], dtype=torch.float)

#training/ev. helpers
def forward_step(model, batch, criterion, temperature: float = 1.0):
    inputs = {"input_ids": batch["input_ids"].to(DEVICE),
              "attention_mask": batch["attention_mask"].to(DEVICE)}
    if "token_type_ids" in batch:
        inputs["token_type_ids"] = batch["token_type_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    outputs = model(**inputs)
    logits = outputs.logits / temperature
    loss = criterion(logits, labels)
    return loss, logits

@torch.no_grad()
def infer_loader(model, loader, criterion, temperature: float = 1.0):
    model.eval()
    by_id_logits = defaultdict(list)
    by_id_label = {}
    for batch in loader:
        _, logits = forward_step(model, batch, criterion, temperature=temperature)
        logits = logits.detach().cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        ids = batch["orig_id"]  # list[str]
        for lgt, y, oid in zip(logits, labels, ids):
            by_id_logits[oid].append(lgt)
            by_id_label[oid] = int(y)

    uniq_ids = list(by_id_label.keys())  # keep insertion order; avoid str/int sort issues
    agg_logits, labels = [], []
    for oid in uniq_ids:
        L = np.stack(by_id_logits[oid], axis=0)  # [num_chunks, 2]
        if CFG.agg == "max":
            pos = L[:, 1].max()
            neg = L[:, 0].mean()
            l = np.array([neg, pos], dtype=np.float32)
        else:
            l = L.mean(axis=0)
        agg_logits.append(l)
        labels.append(by_id_label[oid])

    logits = np.stack(agg_logits, axis=0) if agg_logits else np.zeros((0,2), dtype=np.float32)
    labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:,1] if logits.size else np.array([])
    return logits, probs, labels, uniq_ids

def evaluate_argmax(logits, labels, probs):
    if len(labels) == 0:
        return {"acc":float("nan"), "prec":float("nan"), "rec":float("nan"),
                "f1":float("nan"), "auc":float("nan"), "auprc":float("nan")}
    argmax_preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, argmax_preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, argmax_preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs) if len(set(labels))>1 else float("nan")
        auprc = average_precision_score(labels, probs) if len(set(labels))>1 else float("nan")
    except Exception:
        auc, auprc = float("nan"), float("nan")
    return {"acc":acc, "prec":p, "rec":r, "f1":f1, "auc":auc, "auprc":auprc}

def tune_threshold_from_probs(labels, probs, mode="max_f1", target_precision=0.6):
    if len(labels) == 0:
        return 0.5, float("nan")
    prec, rec, thr = precision_recall_curve(labels, probs)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    if mode == "precision_at":
        idx = np.where(prec[:-1] >= target_precision)[0]
        if len(idx): 
            return float(thr[idx[0]]), float(f1[idx[0]])
    if len(thr)==0: 
        return 0.5, float(f1[:-1].max()) if len(f1)>1 else 0.5
    best = int(np.nanargmax(f1[:-1]))
    return float(thr[best]), float(f1[best])

def temperature_scale(model, loader, criterion):

    model.eval()
    logits, _, labels, _ = infer_loader(model, loader, criterion, temperature=1.0)
    if logits.size == 0:
        return 1.0
    logits_t = torch.tensor(logits, dtype=torch.float, device=DEVICE)
    y = torch.tensor(labels, dtype=torch.long, device=DEVICE)
    T = torch.nn.Parameter(torch.ones([] , device=DEVICE))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    def _closure():
        opt.zero_grad(set_to_none=True)
        scaled = logits_t / T.clamp_min(1e-6)
        loss = nn.functional.cross_entropy(scaled, y)
        loss.backward()
        return loss
    opt.step(_closure)
    T_star = float(T.detach().clamp_min(1e-6).cpu())
    return T_star

def pack_preds(items_by_id: Dict[str, Dict], probs, preds, ids):
    rows = []
    for p, yhat, oid in zip(probs, preds, ids):
        it = items_by_id.get(oid)
        if it is None:
            continue
        rows.append({
            "id": it.get("id"),
            "post_id": it.get("post_id"),
            "AP_label": it["label"],
            "prob_AP1": float(p),
            "pred_AP1": int(yhat),
            "reply": it["reply"][:200],
            "context": (it.get("context") or "")[:200],
        })
    return rows

def main():
    # data
    data = read_jsonl(CFG.data_path)
    print(f"Loaded {len(data)} items")
    pos = sum(x["label"]==1 for x in data)
    print(f"AP=1: {pos} ({pos/len(data):.1%})")

    tr_idx, dv_idx, te_idx = grouped_split(data, seed=CFG.seed)
    train_items = subset(data, tr_idx)
    val_items   = subset(data, dv_idx)
    test_items  = subset(data, te_idx)
    show_stats("Train", train_items)
    show_stats("val",   val_items)
    show_stats("Test",  test_items)

    # expand into chunked samples
    train_samples = expand_items_with_chunks(train_items)
    val_samples   = expand_items_with_chunks(val_items)
    test_samples  = expand_items_with_chunks(test_items)
    print(f"Train samples (chunks): {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    # datasets/loaders
    train_ds = ChunkedDataset(train_samples, training=True)
    val_ds   = ChunkedDataset(val_samples,   training=False)
    test_ds  = ChunkedDataset(test_samples,  training=False)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                              collate_fn=collate_fn_chunked, num_workers=CFG.num_workers,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=CFG.batch_size, shuffle=False,
                              collate_fn=collate_fn_chunked, num_workers=CFG.num_workers,
                              pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=CFG.batch_size, shuffle=False,
                              collate_fn=collate_fn_chunked, num_workers=CFG.num_workers,
                              pin_memory=torch.cuda.is_available())

    config = AutoConfig.from_pretrained(CFG.model_name, num_labels=2)
    model  = AutoModelForSequenceClassification.from_pretrained(CFG.model_name, config=config).to(DEVICE)

    # class weights from original TRAIN items (no leakage)
    cw = class_weights_from(train_items) if CFG.weight_classes else None
    if cw is not None:
        print("Class weights (normalized):", cw.tolist())

    # criterion
    if CFG.use_focal:
        criterion = FocalLoss(gamma=CFG.focal_gamma, alpha=cw.to(DEVICE) if cw is not None else None)
    elif CFG.label_smoothing > 0:
        criterion = LabelSmoothingLoss(smoothing=CFG.label_smoothing, weight=cw)
    else:
        criterion = nn.CrossEntropyLoss(weight=cw.to(DEVICE) if cw is not None else None)

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    total_steps = math.ceil(len(train_loader) / max(1, CFG.grad_accum_steps)) * CFG.epochs
    if CFG.use_scheduler and get_linear_schedule_with_warmup is not None:
        warmup_steps = int(total_steps * CFG.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    else:
        scheduler = None

    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    # training loop
    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    for epoch in range(1, CFG.epochs+1):
        model.train()
        t0 = time.time()
        running_loss, nsteps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                loss, _ = forward_step(model, batch, criterion, temperature=1.0)
            scaler.scale(loss).backward()
            if step % CFG.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                if CFG.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
            running_loss += loss.item()
            nsteps += 1

        # validation
        val_logits, val_probs, val_labels, _ = infer_loader(model, val_loader, criterion, temperature=1.0)
        val_arg = evaluate_argmax(val_logits, val_labels, val_probs)

        # temperature scaling 
        T_star = 1.0
        if CFG.do_temp_scaling:
            T_star = temperature_scale(model, val_loader, criterion)

        val_logits_T, val_probs_T, val_labels_T, _ = infer_loader(model, val_loader, criterion, temperature=T_star)
        best_t, tuned_f1 = tune_threshold_from_probs(
            val_labels_T, val_probs_T,
            mode=("precision_at" if CFG.threshold_mode=="precision_at" else "max_f1"),
            target_precision=CFG.target_precision
        )

        print(f"Epoch {epoch:02d} | loss={running_loss/max(1,nsteps):.4f} | "
              f"val_f1(argmax)={val_arg['f1']:.4f} | val_auc={val_arg['auc']:.4f} | "
              f"T={T_star:.3f} | tuned_f1(val)={tuned_f1:.4f} | t*={best_t:.3f} | "
              f"time={(time.time()-t0):.1f}s")

        # early stopping on tuned val F1
        if tuned_f1 > best_val_f1 + 1e-6:
            best_val_f1 = tuned_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_T = T_star
            best_t_final = best_t
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 2:
                print("Early stopping.")
                break

    # restore best
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        T_star = best_T
        best_t = best_t_final
    else:
        T_star = temperature_scale(model, val_loader, criterion) if CFG.do_temp_scaling else 1.0
        _, val_probs_T, val_labels_T, _ = infer_loader(model, val_loader, criterion, temperature=T_star)
        best_t, _ = tune_threshold_from_probs(val_labels_T, val_probs_T)

    # final val report
    val_logits_T, val_probs_T, val_labels_T, val_ids = infer_loader(model, val_loader, criterion, temperature=T_star)
    val_preds_T = (val_probs_T >= best_t).astype(int)
    val_acc = accuracy_score(val_labels_T, val_preds_T)
    val_p, val_r, val_f1, _ = precision_recall_fscore_support(val_labels_T, val_preds_T, average="binary", zero_division=0)
    print(f"[val] T={T_star:.3f} t*={best_t:.3f} | acc={val_acc:.3f} P={val_p:.3f} R={val_r:.3f} F1={val_f1:.3f}")

    # test
    test_logits_T, test_probs_T, test_labels_T, test_ids = infer_loader(model, test_loader, criterion, temperature=T_star)
    test_preds_T = (test_probs_T >= best_t).astype(int)
    acc = accuracy_score(test_labels_T, test_preds_T)
    p, r, f1, _ = precision_recall_fscore_support(test_labels_T, test_preds_T, average="binary", zero_division=0)
    auc  = roc_auc_score(test_labels_T, test_probs_T) if len(set(test_labels_T))>1 else float("nan")
    auprc = average_precision_score(test_labels_T, test_probs_T) if len(set(test_labels_T))>1 else float("nan")
    print({"test_accuracy":acc, "test_precision":p, "test_recall":r, "test_f1":f1, "test_auc":auc, "test_auprc":auprc})

    # save predictions
    items_by_id_val = {it["id"]: it for it in val_items}
    items_by_id_test = {it["id"]: it for it in test_items}
    val_rows  = pack_preds(items_by_id_val,  val_probs_T,  val_preds_T,  val_ids)
    test_rows = pack_preds(items_by_id_test, test_probs_T, test_preds_T, test_ids)
    pred_path = os.path.join(CFG.output_dir, "predictions_val_test.csv")
    os.makedirs(CFG.output_dir, exist_ok=True)
    pd.DataFrame(val_rows + test_rows).to_csv(pred_path, index=False)
    print("Saved predictions to", pred_path)

    # save model + tokenizer + calibration
    model.save_pretrained(CFG.output_dir)
    tokenizer.save_pretrained(CFG.output_dir)
    with open(os.path.join(CFG.output_dir, "calibration.json"), "w") as f:
        json.dump({"temperature": T_star, "threshold": best_t, "agg": CFG.agg}, f)

if __name__ == "__main__":
    main()
