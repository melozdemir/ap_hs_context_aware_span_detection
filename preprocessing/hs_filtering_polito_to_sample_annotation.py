import zipfile
import bz2
import io
import json
import random
import re
import csv
import requests
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import math
import unicodedata

zip_path = "comments/5851729.zip"
output_file = "reddit_hatebase_filtered_2017_2019_reply_HS_filtered_09_08.jsonl"
target_sample_size = 2000
max_lines_per_file = 150000
max_collect = 80000
scan_files_cap = 600

# Load Davidson Hatebase keyword list
hatebase_url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/lexicons/refined_ngram_dict.csv"
print("Downloading Hatebase lexicon from Davidson et al.")
r = requests.get(hatebase_url)
hs_terms = []
if r.status_code == 200:
    reader = csv.reader(r.text.splitlines())
    next(reader, None)
    for row in reader:
        term = row[0].strip().lower()
        if term:
            hs_terms.append(term)
else:
    raise RuntimeError("Failed to fetch Hatebase lexicon from GitHub")

# text normalization
LEET_MAP = str.maketrans({"@": "a", "$": "s", "0": "o",
                         "1": "i", "!": "i", "3": "e", "4": "a", "5": "s", "7": "t"})


def normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "").lower()
    t = t.translate(LEET_MAP)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_patterns(terms):
    pats = []
    for term in terms:
        nt = normalize_text(term)
        if not nt:
            continue
        toks = nt.split()
        if not toks:
            continue
        if len(toks) == 1:
            patt = rf"\b{re.escape(toks[0])}\b"
        else:
            patt = r"\b" + r"\s+".join(re.escape(tok) for tok in toks) + r"\b"
        try:
            pats.append(re.compile(patt))
        except re.error:
            pass
    return pats


hs_patterns = build_patterns(hs_terms)


def contains_hate_keyword(text: str) -> bool:
    z = normalize_text(text)
    return any(p.search(z) for p in hs_patterns)

# helpers for bot dropping


def is_likely_bot(author, body):
    if not author:
        return True
    return (
        "bot" in author.lower()
        or author.lower() in ["automoderator", "[deleted]", "[removed]"]
        or ("i am a bot" in (body or "").lower())
    )


def format_date(utc):
    try:
        return datetime.utcfromtimestamp(int(utc)).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return ""


pairs_by_year = defaultdict(list)
seen_ids = set()

with zipfile.ZipFile(zip_path, 'r') as archive:
    bz2_files = [f for f in archive.namelist() if f.endswith(".bz2")]
    random.shuffle(bz2_files)
    bz2_files = bz2_files[:scan_files_cap]

    for bz2_name in tqdm(bz2_files, desc="Scanning .bz2 files"):
        comment_dict = {}
        children = defaultdict(list)

        with archive.open(bz2_name) as f:
            with bz2.BZ2File(f) as decompressed:
                for i, line in enumerate(io.TextIOWrapper(decompressed, encoding='utf-8', errors='ignore')):
                    if i >= max_lines_per_file:
                        break
                    try:
                        c = json.loads(line)
                        body = c.get("body") or ""
                        if body in ["[deleted]", "[removed]"]:
                            continue
                        if is_likely_bot(c.get("author"), body):
                            continue
                        cid = c["id"]
                        comment_dict[cid] = {
                            "id": cid,
                            "body": body,
                            "parent_id": c.get("parent_id", ""),
                            "link_id": c.get("link_id", ""),
                            "subreddit": c.get("subreddit", ""),
                            "created_utc": c.get("created_utc", ""),
                            "author": c.get("author", "")
                        }
                        children[c.get("parent_id", "")].append(cid)
                    except Exception:
                        continue
        candidates = [c for c in comment_dict.values()
                      if f"t1_{c['id']}" in children]
        random.shuffle(candidates)

        for parent in candidates:
            if sum(len(v) for v in pairs_by_year.values()) >= max_collect:
                break

            reply_ids = children.get(f"t1_{parent['id']}", [])
            if not reply_ids:
                continue
            reply = comment_dict.get(random.choice(reply_ids))
            if not reply:
                continue

            uid = parent['id'] + "_" + reply['id']
            if uid in seen_ids:
                continue
            if not contains_hate_keyword(reply["body"]):
                continue

            try:
                year = datetime.utcfromtimestamp(
                    int(reply["created_utc"])).year
            except Exception:
                continue
            if year not in [2017, 2018, 2019]:
                continue

            seen_ids.add(uid)
            pairs_by_year[year].append({
                "post_id": parent["link_id"],
                "subreddit": parent["subreddit"],
                "parent_id": parent["id"],
                "parent_body": parent["body"],
                "parent_created_utc": parent["created_utc"],
                "parent_created_date": format_date(parent["created_utc"]),
                "reply_id": reply["id"],
                "reply_body": reply["body"],
                "reply_created_utc": reply["created_utc"],
                "reply_created_date": format_date(reply["created_utc"])
            })

final_samples = []
years = [2017, 2018, 2019]
samples_per_year = math.ceil(target_sample_size / len(years))

for year in years:
    year_samples = pairs_by_year[year][:samples_per_year]
    final_samples.extend(year_samples)
    if len(final_samples) >= target_sample_size:
        break

final_samples = final_samples[:target_sample_size]
random.shuffle(final_samples)

with open(output_file, "w", encoding="utf-8") as out_f:
    for p in final_samples:
        out_f.write(json.dumps(p) + "\n")

print(f"Saved {len(final_samples)} reply-HS-filtered comment pairs {output_file}")
