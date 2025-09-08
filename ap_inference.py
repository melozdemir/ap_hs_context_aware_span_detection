import os, json, re, html, unicodedata, time, glob
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@dataclass
class CFG:
  
    INPUT_PATH: str = "/kaggle/input/2012-2016-elections-polito/politosphere_parent_reply_2012_2016_windows_election.jsonl"
    MODEL_DIR: str = "/kaggle/input/span_left_right_reddit780comments/pytorch/default/1/ap_ctx_span"
    OUT_DIR: str = "/kaggle/working/polito_ap_out_us_2012_16"

    # Inference settings 
    max_len: int = 384
    chunking: bool = True          
    stride: int = 64            
    batch_size: int = 64
    agg: str = "max"             
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()
os.makedirs(cfg.OUT_DIR, exist_ok=True)

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
USER_RE = re.compile(r"(^|\s)u/[A-Za-z0-9_-]+")
SUB_RE  = re.compile(r"(^|\s)r/[A-Za-z0-9_]+")
ZW_RE   = re.compile(r"[\u200B-\u200F\u202A-\u202E]")
WS_RE   = re.compile(r"\s+")

def preprocess_text_for_cls(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    s = html.unescape(s)
    s = ZW_RE.sub("", s)
    s = URL_RE.sub(" ", s)
    s = USER_RE.sub(" ", s)
    s = SUB_RE.sub(" ", s)
    s = s.replace("\r", "\n")
    s = WS_RE.sub(" ", s).strip()
    return s
  
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_DIR, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(cfg.MODEL_DIR).to(cfg.device)
model.eval()

# calibration inspired by Pastell et. al. 
temperature, threshold = 1.0, 0.5
calib_path = os.path.join(cfg.MODEL_DIR, "calibration.json")
if os.path.exists(calib_path):
    try:
        c = json.load(open(calib_path, "r"))
        temperature = float(c.get("temperature", 1.0))
        threshold   = float(c.get("threshold", 0.5))
        agg_from_ckpt = c.get("agg")
        if agg_from_ckpt in ("max", "mean"):
            cfg.agg = agg_from_ckpt
        print(f"Loaded calibration: T={temperature:.3f}, threshold={threshold:.3f}, agg={cfg.agg}")
    except Exception:
        print("failed to read")

SPAN_TOK = None
try:
    special_list = getattr(tokenizer, "additional_special_tokens", []) or []
    if "[SPAN]" in special_list:
        SPAN_TOK = "[SPAN]"
except Exception:
    pass

softmax = torch.nn.Softmax(dim=-1)

def encode_reply_chunks(text: str) -> List[Dict[str, List[int]]]:
    if SPAN_TOK:
        text = f"{SPAN_TOK} {text}".strip()

    enc = tokenizer(
        text,
        truncation=True,
        max_length=cfg.max_len,
        return_overflowing_tokens=cfg.chunking,
        stride=cfg.stride if cfg.chunking else 0,
        return_tensors=None
    )

    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    tti = enc.get("token_type_ids")

    encs = []
    if input_ids and isinstance(input_ids[0], int):
        d = {"input_ids": input_ids, "attention_mask": attn}
        if tti is not None:
            d["token_type_ids"] = tti
        encs.append(d)
        return encs

    for i in range(len(input_ids)):
        d = {"input_ids": input_ids[i], "attention_mask": attn[i]}
        if tti is not None:
            d["token_type_ids"] = tti[i]
        encs.append(d)
    return encs

def pad_batch(features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    return tokenizer.pad(features, padding=True, return_tensors="pt")

def discover_inputs(pattern_or_path: str) -> List[str]:
    if os.path.isfile(pattern_or_path):
        return [pattern_or_path]
    files = sorted(glob.glob(pattern_or_path))
    if not files:
        raise FileNotFoundError(f"No JSONL files found")
    print(f"Found {len(files)} input file(s).")
    for p in files[:10]:
        print("  ", os.path.basename(p))
    return files

def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)

#run inference
def infer_jsonl(input_path: str):
    total_rows = count_lines(input_path)
    t0 = time.time()
    base = os.path.basename(input_path)
    out_path = os.path.join(cfg.OUT_DIR, base.replace(".jsonl", "__ap.jsonl"))
    n_in, n_windows, n_wrote = 0, 0, 0
    last_report_t, last_report_n = t0, 0

    with open(input_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        batch_encs, meta_buf = [], []
        by_id_meta: Dict[str, Dict] = {}

        def flush_batch():
            nonlocal n_wrote
            if not batch_encs: return
            batch = pad_batch(batch_encs)
            for k in ("input_ids","attention_mask","token_type_ids"):
                if k in batch: batch[k] = batch[k].to(cfg.device)
         -
            assert batch["input_ids"].ndim == 2, f"Bad batch shape: {batch['input_ids'].shape}"
            with torch.no_grad():
                logits = model(**{k: batch[k] for k in ("input_ids","attention_mask","token_type_ids") if k in batch}).logits
                logits = logits / temperature
                probs = softmax(logits)[:,1].detach().cpu().numpy()

            # aggregate per original row id
            by_id_probs = defaultdict(list)
            for (oid, _meta), p in zip(meta_buf, probs):
                by_id_probs[oid].append(float(p))

            for oid, plist in by_id_probs.items():
                p = max(plist) if cfg.agg == "max" else float(np.mean(plist))
                yhat = int(p >= threshold)
                meta = by_id_meta[oid]

                out_row = dict(meta)
                out_row["prob_AP1"] = p
                out_row["pred_AP1"] = yhat
                f_out.write(json.dumps(out_row) + "\n")
                n_wrote += 1

            batch_encs.clear()
            meta_buf.clear()

        for line in f_in:
            n_in += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # Required-> reply text
            reply = obj.get("reply") or obj.get("reply_body") or ""
            if not reply:
                continue

            text = preprocess_text_for_cls(reply)
            if not text:
                continue

            oid = obj.get("reply_id") or obj.get("id") or f"row_{n_in}"

            if oid not in by_id_meta:
                by_id_meta[oid] = obj.copy()

            encs = encode_reply_chunks(text)
            for enc in encs:
                batch_encs.append(enc)
                meta_buf.append((oid, by_id_meta[oid]))
                n_windows += 1

            if n_in % 10_000 == 0:
                now = time.time()
                dt = max(1e-6, now - last_report_t)
                dr = n_in - last_report_n
                rps = dr / dt
                pct = (n_in / max(1, total_rows)) * 100.0
                print(f"[{base}] {n_in:,}/{total_rows:,} ({pct:.1f}%) — ~{rps:.1f} rows/s")
                last_report_t, last_report_n = now, n_in

            if len(batch_encs) >= cfg.batch_size:
                flush_batch()
        flush_batch()

    dt = time.time() - t0
    rps = n_in / max(1e-6, dt)
    print(f"[{base}] rows={n_in:,} windows={n_windows:,} wrote={n_wrote:,} -> {out_path}  ({dt:.1f}s, {rps:.1f} rows/s)")

if __name__ == "__main__":
    files = discover_inputs(cfg.INPUT_PATH)
    for p in files:
        infer_jsonl(p)
    print("Done.")
