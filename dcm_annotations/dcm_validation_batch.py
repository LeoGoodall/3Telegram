#!/usr/bin/env python3
"""
DCM Validation Batch: submit 300 random posts (100 per category) x 4 features = 1,200 tasks.
Includes retrieve functionality (call with --retrieve).
"""

import os, re, json, random, csv
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = "/Volumes/One Touch/terrorgram"
JSON_DIR = os.path.join(BASE_DIR, "json_output")
CATEGORIES_CSV = os.path.join(BASE_DIR, "channel_categories.csv")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
OUTPUT_DIR = os.path.join(BASE_DIR, "dcm_validation")
MODEL = "gpt-4o-mini"
SAMPLE_PER_CATEGORY = 100
RANDOM_SEED = 42

INCLUDED_CATEGORIES = {"764", "terrorgram", "terrorgram-affiliated", "white-supremacist"}
CATEGORY_MAP = {"terrorgram-affiliated": "terrorgram"}

FEATURES = {
    "identityfusion": "identityfusion.txt",
    "violentlang": "violentlanguage.txt",
    "threat": "threat.txt",
    "outgroup_othering": "othering.txt",
}


def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'<[^>]+/?>', ' ', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#[x]?[a-fA-F0-9]+;', ' ', text)
    text = re.sub(r'\bhref\s*=\s*["\'][^"\']*["\']', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    text = re.sub(r't\.me/\S+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def channel_to_filename(name):
    parts = []
    for ch in str(name).strip():
        if ch.isspace() or ch in {"-", "\u2013"}:
            parts.append("_")
        elif ch.isalnum() or ch == "_":
            parts.append(ch)
    slug = "".join(parts)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def iter_messages(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for msg in data.get("messages", []):
        if not isinstance(msg, dict):
            continue
        mid = msg.get("message_id")
        if not mid:
            continue
        content = msg.get("content", {})
        text = content.get("text") if isinstance(content, dict) else None
        if isinstance(text, list):
            text = " ".join(
                p if isinstance(p, str) else p.get("text", "") for p in text
            )
        if text and isinstance(text, str) and text.strip():
            yield str(mid), text


def load_prompts():
    prompts = {}
    for feat, fname in FEATURES.items():
        with open(os.path.join(PROMPTS_DIR, fname), "r") as f:
            prompts[feat] = f.read().strip()
    return prompts


def sample_messages():
    """Return {category: [(channel_stem, msg_id, raw_text), ...]} with SAMPLE_PER_CATEGORY per cat."""
    cats = {}
    with open(CATEGORIES_CSV, "r") as f:
        for row in csv.DictReader(f):
            ch = row["channel"].strip()
            cat = row["category"].strip()
            if cat in INCLUDED_CATEGORIES:
                cats.setdefault(CATEGORY_MAP.get(cat, cat), []).append(ch)

    pool = {}  # {category: [(stem, mid, text), ...]}
    for cat, channels in cats.items():
        pool[cat] = []
        for ch in channels:
            stem = channel_to_filename(ch)
            jp = os.path.join(JSON_DIR, f"{stem}.json")
            if not os.path.exists(jp):
                continue
            for mid, text in iter_messages(jp):
                cleaned = clean_text(text)
                if len(cleaned.split()) >= 3:
                    pool[cat].append((stem, mid, text))

    random.seed(RANDOM_SEED)
    sampled = {}
    for cat, msgs in pool.items():
        n = min(SAMPLE_PER_CATEGORY, len(msgs))
        sampled[cat] = random.sample(msgs, n)
        print(f"  {cat}: sampled {n} from {len(msgs):,}")
    return sampled


def create_task(custom_id, text, prompt):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": clean_text(text)},
            ],
        },
    }


def submit():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prompts = load_prompts()

    # Read from existing validation_sample.csv
    sample_csv = os.path.join(OUTPUT_DIR, "validation_sample.csv")
    rows = []
    with open(sample_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    print(f"Loaded {len(rows)} messages from {sample_csv}")

    # Write batch JSONL
    batch_path = os.path.join(OUTPUT_DIR, "validation_batch.jsonl")
    task_count = 0
    with open(batch_path, "w", encoding="utf-8") as f:
        for row in rows:
            cat, ch, mid, text = row["category"], row["channel"], row["message_id"], row["text"]
            for feat, prompt in prompts.items():
                cid = f"{cat}_{ch}_{mid}_{feat}"
                line = json.dumps(create_task(cid, text, prompt), ensure_ascii=False)
                f.write(line + "\n")
                task_count += 1
    print(f"Batch file: {batch_path} ({task_count} tasks)")

    # Upload and submit
    with open(batch_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    tracking = {
        "batch_job_id": batch_job.id,
        "file_id": batch_file.id,
        "task_count": task_count,
        "submitted_at": datetime.now().isoformat(),
    }
    tracking_path = os.path.join(OUTPUT_DIR, "validation_tracking.json")
    with open(tracking_path, "w") as f:
        json.dump(tracking, f, indent=2)
    print(f"Submitted: {batch_job.id}")


def retrieve():
    tracking_path = os.path.join(OUTPUT_DIR, "validation_tracking.json")
    with open(tracking_path, "r") as f:
        tracking = json.load(f)

    batch = client.batches.retrieve(tracking["batch_job_id"])
    print(f"Status: {batch.status}")
    if batch.request_counts:
        print(f"  Completed: {batch.request_counts.completed}/{batch.request_counts.total}")

    if batch.status != "completed":
        return

    # Download results
    content = client.files.content(batch.output_file_id)
    results_path = os.path.join(OUTPUT_DIR, "validation_results.jsonl")
    with open(results_path, "wb") as f:
        f.write(content.read())
    print(f"Downloaded: {results_path}")

    # Map internal feature keys to output column names
    FEATURE_COLUMNS = {
        "identityfusion": "identity_fusion",
        "violentlang": "violence_condoning",
        "threat": "existential_threat",
        "outgroup_othering": "outgroup_othering",
    }

    # Parse results into wide format (one row per message)
    # Collect: (category, channel, msg_id) -> {feature_col: response}
    known_features = list(FEATURES.keys())  # includes "outgroup_othering"
    annotations = {}
    with open(results_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj["custom_id"]

            # Match known feature suffix (handles underscores in feature names)
            feature = None
            prefix = cid
            for feat in sorted(known_features, key=len, reverse=True):
                if cid.endswith("_" + feat):
                    feature = feat
                    prefix = cid[: -(len(feat) + 1)]
                    break
            if feature is None:
                continue

            # prefix is now "{category}_{channel}_{msg_id}"
            parts = prefix.rsplit("_", 1)
            msg_id = parts[-1]
            cat_channel = parts[0]
            cat_parts = cat_channel.split("_", 1)
            category = cat_parts[0] if cat_parts[0] in {"764", "terrorgram"} else "white-supremacist"
            channel = cat_channel[len(category) + 1:]

            resp = obj.get("response", {}).get("body", {})
            choices = resp.get("choices", [])
            answer = choices[0]["message"]["content"].strip() if choices else ""

            key = (category, channel, msg_id)
            if key not in annotations:
                annotations[key] = {}
            col = FEATURE_COLUMNS.get(feature, feature)
            annotations[key][col] = answer

    # Load original sample to get text
    sample_csv = os.path.join(OUTPUT_DIR, "validation_sample.csv")
    text_lookup = {}
    if os.path.exists(sample_csv):
        with open(sample_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["category"], row["channel"], row["message_id"])
                text_lookup[key] = row["text"]

    feature_cols = list(FEATURE_COLUMNS.values())
    fieldnames = ["category", "channel", "message_id", "text"] + feature_cols

    out_csv = os.path.join(OUTPUT_DIR, "dcm_llm_annotations.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (cat, ch, mid), feats in sorted(annotations.items()):
            row = {"category": cat, "channel": ch, "message_id": mid, "text": text_lookup.get((cat, ch, mid), "")}
            for col in feature_cols:
                row[col] = feats.get(col, "")
            w.writerow(row)

    print(f"Parsed: {out_csv} ({len(annotations)} messages, wide format)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--retrieve":
        retrieve()
    else:
        submit()
