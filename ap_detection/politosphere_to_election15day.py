import os
import zipfile
import bz2
import io
import json
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timezone

zip_path = "comments/5851729.zip"
output_file = "politosphere_parent_reply_2012_2016_windows_election.jsonl"

def _ts(y, m, d, H=0, M=0, S=0):
    return int(datetime(y, m, d, H, M, S, tzinfo=timezone.utc).timestamp())

# 2012: 15 days before Nov 6 (inclusive) through Nov 6 (inclusive): 2012-10-23 .. 2012-11-06
WIN_2012_START = _ts(2012, 10, 23, 0, 0, 0)
WIN_2012_END = _ts(2012, 11, 6, 23, 59, 59)

# 2016: 15 days before Nov 8 (inclusive) through Nov 8 (inclusive): 2016-10-25 .. 2016-11-08
WIN_2016_START = _ts(2016, 10, 25, 0, 0, 0)
WIN_2016_END = _ts(2016, 11, 8, 23, 59, 59)

def in_window(ts):
    try:
        t = int(ts)
    except Exception:
        return None
    if WIN_2012_START <= t <= WIN_2012_END:
        return 2012
    if WIN_2016_START <= t <= WIN_2016_END:
        return 2016
    return None

def is_likely_bot(author, body):
    a = (author or "").lower()
    b = (body or "").lower()
    if not author:
        return True
    if a in ("automoderator", "[deleted]", "[removed]"):
        return True
    if "bot" in a:
        return True
    if "i am a bot" in b:
        return True
    return False

def format_date(utc):
    try:
        return datetime.utcfromtimestamp(int(utc)).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return ""

ALLOWED_TAGS = {"2012-10", "2012-11", "2016-10", "2016-11"}
seen_pairs = set()
n_written = 0

with open(output_file, "w", encoding="utf-8") as out_f:
    with zipfile.ZipFile(zip_path, 'r') as archive:
        bz2_files = [
            f for f in archive.namelist()
            if f.endswith(".bz2")
            and any(tag in os.path.basename(f) for tag in ALLOWED_TAGS)
        ]
        for bz2_name in tqdm(bz2_files, desc="Scanning election-month .bz2 files"):
            comment_dict = {}
            children = defaultdict(list)

            # 1) Read all comments from this bz2 file - bots dropped
            with archive.open(bz2_name) as f:
                with bz2.BZ2File(f) as decompressed:
                    for line in io.TextIOWrapper(decompressed, encoding='utf-8', errors='ignore'):
                        try:
                            c = json.loads(line)
                        except Exception:
                            continue
                        body = c.get("body") or ""
                        if body in ("[deleted]", "[removed]"):
                            continue
                        author = c.get("author")
                        if is_likely_bot(author, body):
                            continue
                        cid = c.get("id")
                        if not cid:
                            continue
                        pid = c.get("parent_id", "")
                        comment_dict[cid] = {
                            "id": cid,
                            "body": body,
                            "parent_id": pid,
                            "link_id": c.get("link_id", ""),
                            "subreddit": c.get("subreddit", ""),
                            "created_utc": c.get("created_utc", ""),
                            "author": author or ""
                        }
                        children[pid].append(cid)

            parents = [c for c in comment_dict.values(
            ) if f"t1_{c['id']}" in children]
            for parent in parents:
                for rid in children.get(f"t1_{parent['id']}", []):
                    reply = comment_dict.get(rid)
                    if not reply:
                        continue
                    year = in_window(reply.get("created_utc"))
                    if year is None:
                        continue

                    uid = parent['id'] + "_" + reply['id']
                    if uid in seen_pairs:
                        continue
                    seen_pairs.add(uid)

                    row = {
                
                        "reply": reply["body"],
                        "context": parent["body"],

                        "post_id": parent["link_id"],
                        "subreddit": parent["subreddit"],

                        "parent_id": parent["id"],
                        "parent_body": parent["body"],
                        "parent_created_utc": parent["created_utc"],
                        "parent_created_date": format_date(parent["created_utc"]),

                        "reply_id": reply["id"],
                        "reply_body": reply["body"],
                        "reply_created_utc": reply["created_utc"],
                        "reply_created_date": format_date(reply["created_utc"]),
                        "reply_author": reply["author"],

                        "election_year": year,
                        "window": "2012-10-23..2012-11-06" if year == 2012 else "2016-10-25..2016-11-08"
                    }
                    out_f.write(json.dumps(row) + "\n")
                    n_written += 1

print(f"Wrote to {output_file}")
