# Co-occurrence of Affective Polarization (AP) and Hate Speech (HS)

import os
import json
import glob
from datetime import datetime, date, timedelta, timezone
from collections import defaultdict
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


AP_PATH = "/kaggle/input/inference-span-right-left-reddit/polito_ap_out_us_2012_16/politosphere_parent_reply_2012_2016_windows_election__ap.jsonl"
HS_PATH = "/kaggle/input/hs-infer-out-last/hs_infer_out/politosphere_parent_reply_2012_2016_windows_election__hs.jsonl"

OUT_DIR = "/kaggle/working"
OUT_SUMMARY = os.path.join(OUT_DIR, "ap_hs_cooccurrence_summary.csv")
OUT_DAILY = os.path.join(OUT_DIR, "ap_hs_cooccurrence_daily_15d.csv")
FIG_BOTH = os.path.join(OUT_DIR, "ap_hs_both_counts_15d.png")
FIG_COND = os.path.join(OUT_DIR, "conditional_rates_15d.png")

# 15-day windows relative to election days
REL_START, REL_END = -14, 0

# Election days
ELECTION_DAY = {
    2012: date(2012, 11, 6),
    2016: date(2016, 11, 8),
}


# helpers
def discover_one(path_or_glob: str) -> str:

    if os.path.isfile(path_or_glob):
        return path_or_glob
    files = sorted(glob.glob(path_or_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {path_or_glob}")
    if len(files) > 1:

        for p in files[:3]:
            print("  ", os.path.basename(p))
    return files[0]


def to_utc_date(ts_like) -> Optional[date]:

    try:
        t = int(ts_like)
        return datetime.fromtimestamp(t, tz=timezone.utc).date()
    except Exception:
        return None


def parse_abs_date(obj: Dict[str, Any]) -> Optional[date]:

    for k in ("reply_created_utc", "created_utc", "parent_created_utc"):
        d = to_utc_date(obj.get(k))
        if d is not None:
            return d
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


def key_id(obj: Dict[str, Any]) -> Optional[str]:
    return obj.get("reply_id") or obj.get("id")


def within_15d_window(d: date, y: int) -> bool:
    ed = ELECTION_DAY[y]
    return (ed + timedelta(days=REL_START)) <= d <= (ed + timedelta(days=REL_END))


def rel_day(d: date, y: int) -> int:
    return (d - ELECTION_DAY[y]).days


def load_pred_map(path: str, label_key: str) -> Dict[str, Dict[str, Any]]:

    out = {}
    n, kept = 0, 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            n += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rid = key_id(obj)
            if not rid:
                continue
            if label_key not in obj:
                continue

            d = parse_abs_date(obj)
            y = obj.get("election_year")
            if y not in (2012, 2016):

                y = None

            pred = int(obj[label_key])
            prob = obj.get(label_key.replace("pred", "prob"))
            try:
                prob = float(prob)
            except Exception:
                prob = None

            out[rid] = {"pred": pred, "prob": prob,
                        "date": d, "year": y, "raw": obj}
            kept += 1
    print(f"Loaded {kept:,}/{n:,} usable rows from {os.path.basename(path)}")
    return out


def join_on_reply_id(ap_map: Dict[str, Any], hs_map: Dict[str, Any]) -> pd.DataFrame:
    """Inner join on reply_id/id; keep rows where both AP and HS exist and are within the 15-day windows."""
    rows = []
    keys = ap_map.keys() & hs_map.keys()
    for rid in keys:
        a, h = ap_map[rid], hs_map[rid]
        d = a["date"] or h["date"]
        y = a["year"] or h["year"]
        if d is None or y not in (2012, 2016):
            continue
        if not within_15d_window(d, y):
            continue

        rows.append({
            "reply_id": rid,
            "year": y,
            "abs_date": d,
            "rel_day": rel_day(d, y),
            "AP_pred": a["pred"],
            "HS_pred": h["pred"],
            "AP_prob": a["prob"],
            "HS_prob": h["prob"],
        })
    df = pd.DataFrame(rows).sort_values(
        ["year", "rel_day"]).reset_index(drop=True)
    print(f"Matched {len(df):,} replies across AP & HS within 15-day windows.")
    return df

# metrics


def overall_and_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a tidy summary with:
      - n_matched
      - AP_prevalence = P(AP=1)
      - HS_prevalence = P(HS=1)
      - joint_prev   = P(AP=1 & HS=1)
      - HS_given_AP  = P(HS=1 | AP=1)
      - AP_given_HS  = P(AP=1 | HS=1)
      - risk_diffs   = HS_given_AP - HS_prevalence, AP_given_HS - AP_prevalence
    """
    def summarize(sub):
        n = len(sub)
        ap1 = int((sub["AP_pred"] == 1).sum())
        hs1 = int((sub["HS_pred"] == 1).sum())
        both = int(((sub["AP_pred"] == 1) & (sub["HS_pred"] == 1)).sum())
        ap_prev = ap1 / n if n else np.nan
        hs_prev = hs1 / n if n else np.nan
        joint_prev = both / n if n else np.nan
        hs_given_ap = both / ap1 if ap1 else np.nan
        ap_given_hs = both / hs1 if hs1 else np.nan
        return pd.Series({
            "n_matched": n,
            "AP_prevalence": ap_prev,
            "HS_prevalence": hs_prev,
            "joint_prevalence": joint_prev,
            "HS_given_AP": hs_given_ap,
            "AP_given_HS": ap_given_hs,
            "risk_diff_HS_given_AP": hs_given_ap - hs_prev if not np.isnan(hs_given_ap) else np.nan,
            "risk_diff_AP_given_HS": ap_given_hs - ap_prev if not np.isnan(ap_given_hs) else np.nan,
        })

    parts = []
    parts.append(pd.concat({"overall": summarize(df)},
                 axis=1).T.reset_index(drop=True))
    for y in (2012, 2016):
        sub = df[df["year"] == y]
        s = summarize(sub)
        s["year"] = y
        parts.append(pd.DataFrame([s]))
    out = pd.concat(parts, ignore_index=True)
    cols = ["year", "n_matched", "AP_prevalence", "HS_prevalence",
            "joint_prevalence", "HS_given_AP", "AP_given_HS",
            "risk_diff_HS_given_AP", "risk_diff_AP_given_HS"]

    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


def daily_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per day & year:
      counts: n, ap1, hs1, both, ap_only, hs_only, neither
      rates:  P(AP=1), P(HS=1), P(both), P(HS|AP), P(AP|HS)
    """
    rows = []
    for y in (2012, 2016):
        ed = ELECTION_DAY[y]
        for rd in range(REL_START, REL_END + 1):
            sub = df[(df["year"] == y) & (df["rel_day"] == rd)]
            n = len(sub)
            ap1 = int((sub["AP_pred"] == 1).sum())
            hs1 = int((sub["HS_pred"] == 1).sum())
            both = int(((sub["AP_pred"] == 1) & (sub["HS_pred"] == 1)).sum())
            ap_only = ap1 - both
            hs_only = hs1 - both
            neither = n - (ap_only + hs_only + both)
            rows.append({
                "year": y,
                "rel_day": rd,
                "abs_date": (ed + timedelta(days=rd)).isoformat(),
                "n": n,
                "ap1": ap1,
                "hs1": hs1,
                "both": both,
                "ap_only": ap_only,
                "hs_only": hs_only,
                "neither": neither,
                "P_AP": ap1 / n if n else np.nan,
                "P_HS": hs1 / n if n else np.nan,
                "P_both": both / n if n else np.nan,
                "P_HS_given_AP": (both / ap1) if ap1 else np.nan,
                "P_AP_given_HS": (both / hs1) if hs1 else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["year", "rel_day"]).reset_index(drop=True)


# plottings
def plot_both_counts(daily_df: pd.DataFrame, out_png: str):
    plt.figure(figsize=(8.4, 4.6), dpi=DPI)
    for y, sub in daily_df.groupby("year"):
        sub = sub.sort_values("rel_day")
        plt.plot(sub["rel_day"], sub["both"], marker="o", label=f"{y}")
    plt.axvline(0, ls="--", lw=1, color="k", alpha=0.6)
    plt.title("Co-occurrence counts (AP=1 & HS=1) — 15-day window")
    plt.xlabel("Days relative to election day (0 = Election Day)")
    plt.ylabel("Count of comments with AP=1 and HS=1")
    plt.xticks(range(REL_START, REL_END + 1, 2))
    plt.grid(True, alpha=0.3)
    plt.legend(title="Election Year")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print(f"Saved: {out_png}")


def plot_conditional_rates(daily_df: pd.DataFrame, out_png: str):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), dpi=DPI)

    for idx, (metric, title) in enumerate([
        ("P_HS_given_AP", "P(HS=1 | AP=1)"),
        ("P_AP_given_HS", "P(AP=1 | HS=1)")
    ]):
        for y, sub in daily_df.groupby("year"):
            sub = sub.sort_values("rel_day")
            ax[idx].plot(sub["rel_day"], sub[metric], marker="o", label=f"{y}")
        ax[idx].axvline(0, ls="--", lw=1, color="k", alpha=0.6)
        ax[idx].set_title(f"{title} — 15-day window")
        ax[idx].set_xlabel("Days relative to election day")
        ax[idx].set_ylabel("Conditional rate")
        ax[idx].set_xticks(range(REL_START, REL_END + 1, 2))
        ax[idx].set_ylim(0, 1)
        ax[idx].grid(True, alpha=0.3)
        ax[idx].legend(title="Election Year")

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print(f"Saved: {out_png}")

# main


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ap_file = discover_one(AP_PATH)
    hs_file = discover_one(HS_PATH)

    ap_map = load_pred_map(ap_file, label_key="pred_AP1")
    hs_map = load_pred_map(hs_file, label_key="pred_HS1")

    df = join_on_reply_id(ap_map, hs_map)
    if df.empty:
        raise RuntimeError("No matched rows found")

    # Summaries
    summary = overall_and_by_year(df)
    summary.to_csv(OUT_SUMMARY, index=False)
    print("\n=== Overall & by-year summary ===")
    print(summary)
    print(f"\nSaved summary  {OUT_SUMMARY}")

    daily = daily_table(df)
    daily.to_csv(OUT_DAILY, index=False)
    print(f"Saved table {OUT_DAILY}")

    plot_both_counts(daily, FIG_BOTH)
    plot_conditional_rates(daily, FIG_COND)


if __name__ == "__main__":
    main()
