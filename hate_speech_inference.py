import os, json, time, glob
from dataclasses import dataclass
from typing import List

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@dataclass
class CFG:
    INPUT_PATH: str = "/kaggle/input/2012-2016-elections-polito/politosphere_parent_reply_2012_2016_windows_election.jsonl"
    MODEL_DIR: str = "/kaggle/input/hs_trainedbert/pytorch/default/1/bert_hs_model/checkpoint-400"
    OUT_DIR: str = "/kaggle/working/hs_infer_out"

    # Inference settings
    MAX_LEN: int = 256
    BATCH_SIZE: int = 256         
    THRESHOLD: float = 0.5       
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()
os.makedirs(cfg.OUT_DIR, exist_ok=True)
#helpers
def discover_inputs(pattern_or_path: str) -> List[str]:
    if os.path.isfile(pattern_or_path):
        return [pattern_or_path]
    files = sorted(glob.glob(pattern_or_path))
    if not files:
        raise FileNotFoundError(f"No JSONL files found for: {pattern_or_path}")
    print(f"Found {len(files)} input file(s).")
    for p in files[:10]:
        print("  ", os.path.basename(p))
    return files

def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)

# load model
print("Loading tokenizer and model")
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_DIR, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(cfg.MODEL_DIR).to(cfg.DEVICE)
model.eval()
print(f"Device: {cfg.DEVICE}")

@torch.inference_mode()
def infer_jsonl(input_path: str):
    total_rows = count_lines(input_path)
    base = os.path.basename(input_path)
    out_path = os.path.join(cfg.OUT_DIR, base.replace(".jsonl", "__hs.jsonl"))
    t0 = time.time()

    texts, metas = [], []
    n_in = n_scored = 0
    last_report_t, last_report_n = t0, 0

    with open(input_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        def flush_batch():
            nonlocal n_scored
            if not texts:
                return
            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=cfg.MAX_LEN,
                return_tensors="pt"
            ).to(cfg.DEVICE)

            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

            for meta, p in zip(metas, probs):
                out = dict(meta)
                out["prob_HS1"] = float(p)
                out["pred_HS1"] = int(p >= cfg.THRESHOLD)
                f_out.write(json.dumps(out) + "\n")
                n_scored += 1

            texts.clear()
            metas.clear()

        for line in f_in:
            n_in += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            reply = obj.get("reply") or obj.get("reply_body") or ""
            if not reply:
                continue
            metas.append(obj)
            texts.append(reply)
            if len(texts) >= cfg.BATCH_SIZE:
                flush_batch()

            # progress print
            if n_in % 50_000 == 0:
                now = time.time()
                dt = max(1e-6, now - last_report_t)
                dr = n_in - last_report_n
                rps = dr / dt
                pct = (n_in / max(1, total_rows)) * 100.0
                print(f"[{base}] {n_in:,}/{total_rows:,} ({pct:.1f}%) — ~{rps:.1f} rows/s")
                last_report_t, last_report_n = now, n_in

        flush_batch()

    dt = time.time() - t0
    rps = n_in / max(dt, 1e-6)
    print(f"finished. wrote {n_scored:,} rows -> {out_path}  ({dt:.1f}s, {rps:.1f} rows/s)")
if __name__ == "__main__":
    files = discover_inputs(cfg.INPUT_PATH)
    for p in files:
        infer_jsonl(p)
    print("All files finished.")
