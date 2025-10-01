# HS_15day_Election_Analysis.py
# Build 15-day time series (−14…0 relative to election day) of Hate Speech (HS) for 2012 & 2016

import os
import glob
import json
from datetime import datetime, date, timedelta, timezone
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_GLOB = "/kaggle/input/hs-infer-out-last/hs_infer_out/politosphere_parent_reply_2012_2016_windows_election__hs.jsonl"

OUT_DIR = "/kaggle/working"
OUT_TABLE = os.path.join(OUT_DIR, "hs_daily_metrics_15d.csv")
FIG_RATES = os.path.join(OUT_DIR, "hs_rates_15d.png")
FIG_COUNTS = os.path.join(OUT_DIR, "hs_counts_15d.png")

# Window relative to election day
REL_START, REL_END = -14, 0

# Election days
ELECTION_DAY = {
    2012: date(2012, 11, 6),
    2016: date(2016, 11, 8),
}


def discover_files(pattern_or_path: str):
    """Return [path] if a file, else glob all matches; error if none."""
    if os.path.isfile(pattern_or_path):
        return [pattern_or_path]
    files = sorted(glob.glob(pattern_or_path))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern_or_path}")
    print(f"Found {len(files)} file(s).")
    return files


def to_utc_date(ts_like) -> Optional[date]:
    """Epoch seconds -> UTC date; return None on failure."""
    try:
        t = int(ts_like)
        return datetime.fromtimestamp(t, tz=timezone.utc).date()
    except Exception:
        return None


def parse_abs_date(obj: dict) -> Optional[date]:
    """Prefer epoch fields; fall back to %Y-%m-%d[ %H:%M:%S] strings."""
    # epoch first
    for k in ("reply_created_utc", "created_utc", "parent_created_utc"):
        d = to_utc_date(obj.get(k))
        if d is not None:
            return d
    # then strings
    for k in ("reply_created_date", "created_date", "parent_created_date"):
        s = obj.get(k)
        if not s:
            continue
        s = str(s)
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s[:len(fmt)], fmt).date()
            except Exception:
                pass
    return None


def classify_year(obj: dict, d: Optional[date]) -> Optional[int]:
    """Use explicit election_year if present; else infer by which election window the date falls in."""
    y = obj.get("election_year")
    if y in (2012, 2016):
        return int(y)
    if d is None:
        return None
    for yy, ed in ELECTION_DAY.items():
        if (ed + timedelta(days=REL_START)) <= d <= (ed + timedelta(days=REL_END)):
            return yy
    return None


def build_daily_metrics(files) -> pd.DataFrame:
    """
    counters[(year, abs_date)] -> {'n': total comments, 'k': HS=1 count, 'sum_p': sum of prob_HS1}
    """
    counters = defaultdict(lambda: {"n": 0, "k": 0, "sum_p": 0.0})

    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                d = parse_abs_date(obj)
                y = classify_year(obj, d)
                if y not in (2012, 2016) or d is None:
                    continue

                rel_day = (d - ELECTION_DAY[y]).days
                if rel_day < REL_START or rel_day > REL_END:
                    continue

                key = (y, d)
                counters[key]["n"] += 1

                try:
                    pred = int(obj.get("pred_HS1", 0))
                except Exception:
                    pred = 0
                if pred == 1:
                    counters[key]["k"] += 1

                p = obj.get("prob_HS1")
                if p is not None:
                    try:
                        counters[key]["sum_p"] += float(p)
                    except Exception:
                        pass
    rows = []
    for y, ed in ELECTION_DAY.items():
        for rd in range(REL_START, REL_END + 1):
            abs_d = ed + timedelta(days=rd)
            n = counters[(y, abs_d)]["n"] if (y, abs_d) in counters else 0
            k = counters[(y, abs_d)]["k"] if (y, abs_d) in counters else 0
            sum_p = counters[(y, abs_d)]["sum_p"] if (
                y, abs_d) in counters else 0.0
            rate = (k / n) if n > 0 else np.nan
            mean_prob = (sum_p / n) if n > 0 else np.nan

            rows.append({
                "year": y,
                "rel_day": rd,                  # −14..0
                "abs_date": abs_d.isoformat(),
                "count_total": n,
                "count_HS1": k,
                "hs_rate": rate,
                "mean_prob": mean_prob,
            })

    df = pd.DataFrame(rows).sort_values(
        ["year", "rel_day"]).reset_index(drop=True)
    return df

# Plottings


def plot_rates(df: pd.DataFrame, out_png: str):
    plt.figure(figsize=(8.2, 4.6), dpi=DPI)
    for y, sub in df.groupby("year"):
        sub = sub.sort_values("rel_day")
        plt.plot(sub["rel_day"], sub["hs_rate"], marker="o", label=f"{y}")
    plt.axvline(0, linestyle="--", linewidth=1, color="k", alpha=0.6)
    plt.title("Hate Speech (HS) rate — 15-day window ending on Election Day")
    plt.xlabel("Days relative to election day (0 = Election Day)")
    plt.ylabel("HS rate (share of pred_HS1 = 1)")
    plt.xticks([-14, -10, -7, -5, -3, -1, 0])
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Election Year")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print(f"Saved rates plot → {out_png}")


def plot_counts(df: pd.DataFrame, out_png: str):
    plt.figure(figsize=(8.2, 4.6), dpi=DPI)
    for y, sub in df.groupby("year"):
        sub = sub.sort_values("rel_day")
        plt.plot(sub["rel_day"], sub["count_HS1"], marker="o", label=f"{y}")
    plt.axvline(0, linestyle="--", linewidth=1, color="k", alpha=0.6)
    plt.title("Hate Speech (HS) counts — 15-day window ending on Election Day")
    plt.xlabel("Days relative to election day (0 = Election Day)")
    plt.ylabel("HS count (pred_HS1 = 1)")
    plt.xticks(range(REL_START, REL_END + 1, 2))
    plt.grid(True, alpha=0.3)
    plt.legend(title="Election Year")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print(f"Saved counts plot → {out_png}")


# Main

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = discover_files(INPUT_GLOB)
    df = build_daily_metrics(files)

    df.to_csv(OUT_TABLE, index=False)
    print(f"Saved daily metrics table → {OUT_TABLE}")

    summ = df.groupby("year")[["count_HS1", "count_total"]].sum()
    summ["HS_share"] = summ["count_HS1"] / \
        summ["count_total"].replace(0, np.nan)
    print("\n15-day window totals:")
    print(summ)

    plot_rates(df, FIG_RATES)
    plot_counts(df, FIG_COUNTS)


if __name__ == "__main__":
    main()
