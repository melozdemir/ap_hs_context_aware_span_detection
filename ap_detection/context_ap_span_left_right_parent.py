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
    # paths / model
    data_path: str = "/kaggle/input/annotated-data-20-08/all_anotdata_combined.jsonl"
    output_dir: str = "/kaggle/working/ap_ctx_span"
    model_name: str = "bert-base-uncased"
    seed: int = 42

    # lengths & batching
    max_len: int = 384
    span_min_tokens: int = 32         
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
    use_focal: bool = False        #test if it improves performance  
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0

    # context / chunking
    use_context: bool = True          
    include_parent: bool = True        # include parent in context 
    context_dropout: float = 0.15      
    chunking: bool = True             
    ctx_stride: int = 64              
    agg: str = "max"                   # "mean" or "max" aggregation over chunks

    # calibration + thresholding
    do_temp_scaling: bool = True
    threshold_mode: str = "max_f1"   
    target_precision: float = 0.60

cfg = CFG()


#utils and set up 
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

def ensure_dir(p): os.makedirs(p, exist_ok=True)
ensure_dir(cfg.output_dir)


#span aware data prep.
def _pick_best_span(spans: list) -> Optional[str]:
    if not spans:
        return None
    spans = [s for s in spans if isinstance(s, str) and s.strip()]
    return max(spans, key=len) if spans else None

def _split_by_span(reply: str, span: Optional[str]) -> tuple[str, str, str]:
    """
    Return (left, span_text, right). If span cannot be matched, fall back to whole reply as span.
    """
    if not span:
        return "", reply, ""
    rl, sl = reply.lower(), span.lower()
    i = rl.find(sl)
    if i == -1:
        sl2 = sl.strip(' "\'`')
        i = rl.find(sl2) if sl2 else -1
        if i == -1:
            return "", reply, ""
        span = reply[i:i+len(sl2)]
    left  = reply[:i]
    right = reply[i+len(span):]
    return left, span, right

def read_jsonl_as_span_rows(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            if "reply" not in obj or "AP_label" not in obj:  # skip malformed
                continue
            reply = obj["reply"]
            parent = obj.get("context") or ""
            best_span = _pick_best_span(obj.get("AP_span", []))
            left, span, right = _split_by_span(reply, best_span)
            meta = obj.get("meta") or {}
            rows.append({
                "span": span,
                "left": left,
                "right": right,
                "parent": parent,
                "label": int(obj["AP_label"]),
                "id": str(obj.get("id", i)),
                "post_id": str(meta.get("post_id") or f"noid_{i}"),
            })
    return rows

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

# Tokenizer + special tokens + chunking helpers

SPECIAL = ["[SPAN]", "[LCTX]", "[CTX_SEP]", "[RCTX]", "[PCTX]"]
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL})

def _ids(text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]

def _make_chunks(ids: List[int], win: int, stride: int) -> List[List[int]]:
    if win <= 0 or len(ids) <= win:
        return [ids[:win]]
    step = max(1, win - stride)
    chunks = [ids[i:i+win] for i in range(0, len(ids), step)]
    if len(chunks[-1]) < win and len(ids) > win:
        chunks[-1] = ids[-win:]
    return chunks

def _build_ctx_ids(left: str, right: str, parent: str, include_parent: bool) -> List[int]:
   
    ctx_text = f"{SPECIAL[1]} {left} {SPECIAL[2]} {SPECIAL[3]} {right}"
    if include_parent and parent:
        ctx_text += f" {SPECIAL[4]} {parent}"
    return _ids(ctx_text)

def encode_span_left_right(span: str, left: str, right: str, parent: str,
                           use_context: bool, include_parent: bool,
                           chunking: bool, max_len: int,
                           span_min_tokens: int, ctx_stride: int):
    span_ids = _ids(f"{SPECIAL[0]} {span}")
    if not use_context:
        budget = max_len - 2  # [CLS],[SEP]
        s_keep = min(len(span_ids), budget)
        return [tokenizer.prepare_for_model(ids=span_ids[:s_keep], add_special_tokens=True)]

    ctx_ids  = _build_ctx_ids(left or "", right or "", parent or "", include_parent)
    budget = max_len - 3  # [CLS],[SEP],[SEP]
    s_keep = min(len(span_ids), max(span_min_tokens, min(len(span_ids), budget)))
    s_keep = min(s_keep, budget)
    ctx_budget = max(0, budget - s_keep)

    if (not chunking) or len(ctx_ids) <= ctx_budget:
        return [tokenizer.prepare_for_model(ids=span_ids[:s_keep], pair_ids=ctx_ids[:ctx_budget], add_special_tokens=True)]

    encs = []
    for ch in _make_chunks(ctx_ids, ctx_budget, ctx_stride):
        encs.append(tokenizer.prepare_for_model(ids=span_ids[:s_keep], pair_ids=ch, add_special_tokens=True))
    return encs

#dataset
class SpanChunkedDataset(Dataset):

    def __init__(self, rows: List[Dict], training: bool = False):
        self.samples = []
        for r in rows:
            left, right, parent = r.get("left",""), r.get("right",""), r.get("parent","")
            if training and cfg.use_context and random.random() < cfg.context_dropout:
                left, right, parent = "", "", ""
            encs = encode_span_left_right(
                r["span"], left, right, parent,
                use_context=cfg.use_context,
                include_parent=cfg.include_parent,
                chunking=cfg.chunking,
                max_len=cfg.max_len,
                span_min_tokens=cfg.span_min_tokens,
                ctx_stride=cfg.ctx_stride
            )
            for enc in encs:
                self.samples.append({
                    "enc": {k: enc[k] for k in enc if k in ("input_ids","token_type_ids","attention_mask")},
                    "label": int(r["label"]),
                    "orig_id": r["id"],
                    "post_id": r["post_id"],
                })

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        item = dict(s["enc"])
        item["labels"] = s["label"]
        item["orig_id"] = s["orig_id"]
        item["post_id"] = s["post_id"]
        return item

def collate_fn(features):
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


#training helpers
def build_model():
    config = AutoConfig.from_pretrained(cfg.model_name, num_labels=2)
    model  = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=config)
   
    model.resize_token_embeddings(len(tokenizer))
    return model

def forward_step(model, batch, criterion, temperature: float = 1.0):
    inputs = {
        "input_ids": batch["input_ids"].to(DEVICE),
        "attention_mask": batch["attention_mask"].to(DEVICE)
    }
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
        ids = batch["orig_id"]
        for lgt, y, oid in zip(logits, labels, ids):
            by_id_logits[oid].append(lgt)
            by_id_label[oid] = int(y)

    uniq_ids = list(by_id_label.keys()) 
    agg_logits, labels = [], []
    for oid in uniq_ids:
        L = np.stack(by_id_logits[oid], axis=0) 
        if cfg.agg == "max":
            pos = L[:, 1].max()
            neg = L[:, 0].mean()
            l = np.array([neg, pos], dtype=np.float32)
        else:  # mean
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
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
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
    T = torch.nn.Parameter(torch.ones([], device=DEVICE))
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

#training/ev.
def train(model, train_loader, val_loader, class_weights=None):
    model.to(DEVICE)
    if class_weights is not None:
        class_weights = class_weights.to(DEVICE)

    # criterion
    if cfg.use_focal:
        criterion = FocalLoss(gamma=cfg.focal_gamma, alpha=class_weights)
    elif cfg.label_smoothing > 0:
        criterion = LabelSmoothingLoss(smoothing=cfg.label_smoothing, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = math.ceil(len(train_loader) / max(1, cfg.grad_accum_steps)) * cfg.epochs
    if cfg.use_scheduler and get_linear_schedule_with_warmup is not None:
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    else:
        scheduler = None

    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    best_val_f1, best_state = -1.0, None
    epochs_no_improve = 0
    history = {"epoch": [], "train_loss": [], "val_auc": [], "val_f1": []}

    for epoch in range(1, cfg.epochs+1):
        model.train()
        t0 = time.time()
        running_loss, nsteps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                loss, _ = forward_step(model, batch, criterion, temperature=1.0)
            scaler.scale(loss).backward()
            if step % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                if cfg.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
            running_loss += loss.item()
            nsteps += 1

        # Validation
        val_logits, val_probs, val_labels, _ = infer_loader(model, val_loader, criterion, temperature=1.0)
        val_stats = evaluate_argmax(val_logits, val_labels, val_probs)

        # Temperature scaling + threshold tuning
        T_star = temperature_scale(model, val_loader, criterion) if cfg.do_temp_scaling else 1.0
        val_logits_T, val_probs_T, val_labels_T, _ = infer_loader(model, val_loader, criterion, temperature=T_star)
        best_t, tuned_f1 = tune_threshold_from_probs(
            val_labels_T, val_probs_T,
            mode=("precision_at" if cfg.threshold_mode=="precision_at" else "max_f1"),
            target_precision=cfg.target_precision
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(running_loss/max(1,nsteps))
        history["val_auc"].append(val_stats["auc"])
        history["val_f1"].append(tuned_f1)

        print(f"Epoch {epoch:02d} | loss={running_loss/max(1,nsteps):.4f} | "
              f"val_auc={val_stats['auc']:.4f} | val_f1(argmax)={val_stats['f1']:.4f} | "
              f"T={T_star:.3f} | tuned_f1(val)={tuned_f1:.4f} | t*={best_t:.3f} | "
              f"time={(time.time()-t0):.1f}s")

        # early stopping with f1 score
        if tuned_f1 > best_val_f1 + 1e-6:
            best_val_f1 = tuned_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            global _BEST_T, _BEST_TH
            _BEST_T, _BEST_TH = T_star, best_t
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 2:
                print("Early stopping.")
                break

    # get the best model
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return history

@torch.no_grad()
def evaluate_final(model, loader, T: float, thresh: float, criterion):
    logits, probs, labels, ids = infer_loader(model, loader, criterion, temperature=T)
    preds = (probs >= thresh).astype(int)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    auc  = roc_auc_score(labels, probs) if len(set(labels))>1 else float("nan")
    auprc = average_precision_score(labels, probs) if len(set(labels))>1 else float("nan")
    return {"acc": acc, "prec": p, "rec": r, "f1": f1, "auc": auc, "auprc": auprc,
            "labels": labels, "preds": preds, "probs": probs, "ids": ids}


def main():
    print("Loading data (span-aware)")
    data = read_jsonl_as_span_rows(cfg.data_path)
    print(f"Loaded {len(data)} items")
    pos = sum(x["label"]==1 for x in data)
    print(f"AP=1: {pos} ({pos/len(data):.1%})")

    tr_idx, dv_idx, te_idx = grouped_split(data, seed=cfg.seed)
    train_items = subset(data, tr_idx)
    val_items   = subset(data, dv_idx)
    test_items  = subset(data, te_idx)
    show_stats("Train", train_items)
    show_stats("Val",   val_items)
    show_stats("Test",  test_items)

    # Build datasets
    train_ds = SpanChunkedDataset(train_items, training=True)
    val_ds   = SpanChunkedDataset(val_items,   training=False)
    test_ds  = SpanChunkedDataset(test_items,  training=False)

    print(f"Samples — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=cfg.num_workers,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=cfg.num_workers,
                              pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=cfg.num_workers,
                              pin_memory=torch.cuda.is_available())

    model = build_model().to(DEVICE)

    def class_weights_from_rows(rows: List[Dict]):
        y = np.array([x["label"] for x in rows])
        n0, n1 = (y==0).sum(), (y==1).sum()
        if n0 == 0 or n1 == 0: return None
        N = len(y); w0 = N/(2*n0); w1 = N/(2*n1)
        s = (w0 + w1) / 2.0
        return torch.tensor([w0/s, w1/s], dtype=torch.float)

    cw = class_weights_from_rows(train_items) if cfg.weight_classes else None
    if cw is not None:
        print("Class weights (normalized):", cw.tolist())

    history = train(model, train_loader, val_loader, class_weights=cw)

    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(cfg.output_dir, "training_metrics.csv"), index=False)

    # Final calibration & threshold 
    criterion = nn.CrossEntropyLoss(weight=cw.to(DEVICE) if cw is not None else None)
    T_star = _BEST_T if cfg.do_temp_scaling else 1.0
    th_star = _BEST_TH if cfg.do_temp_scaling else 0.5

    # Evaluate (val & test)
    print("\nEvaluating on validation …")
    val_res = evaluate_final(model, val_loader, T_star, th_star, criterion)
    print({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in val_res.items() if k in ["acc","prec","rec","f1","auc","auprc"]})

    print("\nEvaluating on test …")
    test_res = evaluate_final(model, test_loader, T_star, th_star, criterion)
    print({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in test_res.items() if k in ["acc","prec","rec","f1","auc","auprc"]})

    # Save predictions
    items_by_id_val  = {it["id"]: it for it in val_items}
    items_by_id_test = {it["id"]: it for it in test_items}

    def pack_preds(items_by_id: Dict[str, Dict], probs, preds, ids):
        rows = []
        for p, yhat, oid in zip(probs, preds, ids):
            it = items_by_id.get(oid)
            if it is None:
                continue
            reply_recon = (it.get("left","") + it.get("span","") + it.get("right","")).strip()
            rows.append({
                "id": it.get("id"),
                "post_id": it.get("post_id"),
                "AP_label": it["label"],
                "prob_AP1": float(p),
                "pred_AP1": int(yhat),
                "span": it.get("span","")[:200],
                "left": it.get("left","")[:200],
                "right": it.get("right","")[:200],
                "parent": (it.get("parent") or "")[:200],
                "reply_reconstructed": reply_recon[:250]
            })
        return rows

    val_rows  = pack_preds(items_by_id_val,  val_res["probs"],  val_res["preds"],  val_res["ids"])
    test_rows = pack_preds(items_by_id_test, test_res["probs"], test_res["preds"], test_res["ids"])
    pred_path = os.path.join(cfg.output_dir, "predictions_val_test.csv")
    pd.DataFrame(val_rows + test_rows).to_csv(pred_path, index=False)
    print("Saved predictions to", pred_path)

    # Save model + tokenizer + calibration
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    with open(os.path.join(cfg.output_dir, "calibration.json"), "w") as f:
        json.dump({"temperature": T_star, "threshold": th_star, "agg": cfg.agg}, f)
    print("Files saved to:", cfg.output_dir)


if __name__ == "__main__":
    _BEST_T, _BEST_TH = (1.0, 0.5)
    main()
