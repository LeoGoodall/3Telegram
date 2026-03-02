"""
Topic Modelling using BERTopic with fixed number of topics.

Uses BERTopic's nr_topics parameter to ensure consistent topic counts
across categories regardless of corpus size.
"""

import json
import os
import re
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# Configuration
RANDOM_STATE = 42
TEXT_COL = "text"
CATEGORY_COL = "category"
EMBED_MODEL = "all-MiniLM-L6-v2"
NR_TOPICS = 20  # Fixed number of topics per category

INCLUDED_CATEGORIES = {"764", "terrorgram", "terrorgram-affiliated", "white-supremacist"}
CATEGORY_MAP = {"terrorgram-affiliated": "terrorgram"}  # merge affiliated into terrorgram

# Custom stopwords: high-frequency informal terms with no discriminatory power across topics.
# These extend sklearn's standard English stopwords and only affect c-TF-IDF topic
# representation, NOT the embedding-based clustering.
CUSTOM_STOPWORDS = [
    "like", "just", "don", "dont", "doesn", "doesnt",
    "got", "good", "think", "know", "ur", "im",
    "lol", "yeah", "ve", "want", "use", "need", "people",
    "matter", "matters", "depends", "does", "yes", "no", "really", "oh", "gonna",
    "did", "make", "made", "time", "times"
]

def clean_html(text: str) -> str:
    """
    Remove HTML tags, entities, and orphaned attributes from text.
    Only removes actual HTML syntax, not words that might coincidentally match tag names.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags (including self-closing like <br/>, <br />, and tags with attributes)
    text = re.sub(r'<[^>]+/?>', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove HTML entities (named and numeric)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # &nbsp; &amp; &lt; &gt; &quot;
    text = re.sub(r'&#\d+;', ' ', text)  # &#123; (decimal)
    text = re.sub(r'&#x[a-fA-F0-9]+;', ' ', text)  # &#xAB; (hex)

    # Remove orphaned HTML attributes (attr="value" patterns)
    text = re.sub(r'\bhref\s*=\s*["\'][^"\']*["\']', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bclass\s*=\s*["\'][^"\']*["\']', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bstyle\s*=\s*["\'][^"\']*["\']', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsrc\s*=\s*["\'][^"\']*["\']', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\balt\s*=\s*["\'][^"\']*["\']', ' ', text, flags=re.IGNORECASE)

    return text


def preprocess(text: str, min_words: int = 3) -> str:
    """
    Clean text for embedding-based topic modeling.
    - Remove HTML tags and entities
    - Lowercase
    - Remove URLs (http, www, t.me links)
    - Remove @mentions
    - Skip messages shorter than min_words
    """
    # Clean HTML first
    text = clean_html(str(text))
    text = text.lower()

    # Remove URLs and mentions
    tokens = []
    for t in text.split():
        if t.startswith("http") or t.startswith("www.") or "t.me/" in t:
            continue
        if t.startswith("@"):
            continue
        tokens.append(t)

    if len(tokens) < min_words:
        return ""

    return " ".join(tokens)


def channel_to_filename(name: str) -> str:
    """
    Convert a channel display name to the corresponding JSON filename stem.
    """
    name = str(name).strip()
    parts = []
    for ch in name:
        if ch.isspace() or ch in {"-", "–"}:
            parts.append("_")
        elif ch.isalnum() or ch == "_":
            parts.append(ch)
        else:
            continue

    slug = "".join(parts)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _iter_message_texts(obj):
    """
    Extract message text using exactly the expected fields:
    - top level: "messages"
    - per message: "content.text" or "text"
    """
    if not isinstance(obj, dict):
        return

    messages = obj.get("messages", [])
    if not isinstance(messages, list):
        return

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        content = msg.get("content")
        if isinstance(content, dict):
            text = content.get("text")
        else:
            text = msg.get("text")

        if isinstance(text, str) and text.strip():
            yield text
        elif isinstance(text, list):
            parts = []
            for part in text:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    t = part.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            if parts:
                yield " ".join(parts)


def load_messages(categories_csv_path: str, json_dir: str) -> pd.DataFrame:
    """
    Build a message-level DataFrame with columns:
    - channel
    - category
    - text
    """
    categories = pd.read_csv(categories_csv_path)

    rows = []
    seen_per_category = {}

    for _, row in categories.iterrows():
        channel_name = str(row.get("channel", "")).strip()
        if not channel_name or channel_name.startswith("._"):
            continue

        category = row.get(CATEGORY_COL)
        if category not in INCLUDED_CATEGORIES:
            continue

        category = CATEGORY_MAP.get(category, category)

        if category not in seen_per_category:
            seen_per_category[category] = set()

        filename_stem = channel_to_filename(channel_name)
        if not filename_stem:
            continue

        json_path = os.path.join(json_dir, f"{filename_stem}.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for msg_text in _iter_message_texts(data):
            cleaned = preprocess(msg_text)
            if cleaned and cleaned not in seen_per_category[category]:
                seen_per_category[category].add(cleaned)
                rows.append({
                    "channel": channel_name,
                    CATEGORY_COL: category,
                    TEXT_COL: cleaned,
                })

    return pd.DataFrame(rows)


def fit_bertopic(texts: list, nr_topics: int = NR_TOPICS) -> tuple:
    """
    Fit BERTopic model with a fixed number of topics.

    Returns:
        topics: list of topic assignments per document
        topic_model: fitted BERTopic model
    """
    # Configure sub-models for BERTopic
    embedding_model = SentenceTransformer(EMBED_MODEL)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    )

    # HDBSCAN with reasonable defaults - BERTopic will handle topic reduction
    hdbscan_model = HDBSCAN(
        min_cluster_size=50,
        min_samples=10,
        metric="euclidean",
        prediction_data=True,
    )

    # Custom vectorizer: extend standard English stopwords with informal terms.
    # Only affects c-TF-IDF topic representation, not embedding-based clustering.
    vectorizer_model = CountVectorizer(
        stop_words=list(CountVectorizer(stop_words="english").get_stop_words()) + CUSTOM_STOPWORDS,
        ngram_range=(1, 2),
        min_df=2,
    )

    # Create BERTopic model with nr_topics to force exact topic count
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=nr_topics,
        verbose=True,
    )

    # Fit the model
    topics, probs = topic_model.fit_transform(texts)

    return topics, topic_model


def run_topic_model(categories_csv_path: str, json_dir: str, output_prefix: str):
    """
    Run BERTopic on each category separately, producing exactly NR_TOPICS per category.
    """
    print(f"Loading messages...")
    df = load_messages(categories_csv_path, json_dir)
    if df.empty:
        raise RuntimeError("No messages were loaded. Please check your inputs.")

    print(f"Loaded {len(df):,} messages across {df[CATEGORY_COL].nunique()} categories")

    all_topic_info = []

    for category in sorted(df[CATEGORY_COL].unique()):
        print(f"\n{'='*60}")
        print(f"Processing category: {category}")
        print(f"{'='*60}")

        mask = df[CATEGORY_COL] == category
        texts = df.loc[mask, TEXT_COL].tolist()

        if len(texts) < 100:
            print(f"  Skipping {category}: too few messages ({len(texts)})")
            continue

        print(f"  Messages: {len(texts):,}")

        # Fit BERTopic
        topics, topic_model = fit_bertopic(texts, nr_topics=NR_TOPICS)

        # Assign topics back to dataframe
        df.loc[mask, "topic"] = topics

        # Get topic info
        topic_info = topic_model.get_topic_info()
        n_topics = len(topic_info[topic_info["Topic"] != -1])
        n_noise = len([t for t in topics if t == -1])

        print(f"  Topics found: {n_topics}")
        print(f"  Noise messages: {n_noise:,} ({100*n_noise/len(texts):.1f}%)")

        # Store topic centroids for cross-category analysis
        # Get document embeddings and compute centroids
        embeddings = topic_model._extract_embeddings(texts, method="document")

        for topic_id in topic_info[topic_info["Topic"] != -1]["Topic"]:
            topic_mask = np.array(topics) == topic_id
            if topic_mask.sum() > 0:
                centroid = embeddings[topic_mask].mean(axis=0)
                all_topic_info.append({
                    "category": category,
                    "topic": int(topic_id),
                    "centroid": centroid,
                })

    # Cross-category topic alignment using centroids
    print(f"\n{'='*60}")
    print("Computing cross-category topic alignment...")
    print(f"{'='*60}")

    if all_topic_info:
        topics_df = pd.DataFrame([
            {"category": t["category"], "topic": t["topic"]}
            for t in all_topic_info
        ])

        centroid_matrix = np.vstack([t["centroid"] for t in all_topic_info])

        # Cluster the topic centroids to find cross-category themes
        meta_umap = UMAP(
            n_neighbors=min(10, len(centroid_matrix) - 1),
            n_components=2,
            min_dist=0.1,
            metric="cosine",
            random_state=RANDOM_STATE,
        )
        meta_embeddings = meta_umap.fit_transform(centroid_matrix)

        meta_clusterer = HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric="euclidean",
        )
        meta_labels = meta_clusterer.fit_predict(meta_embeddings)

        topics_df["meta_topic"] = meta_labels

        print(f"  Found {len(set(meta_labels) - {-1})} cross-category meta-topics")
    else:
        topics_df = pd.DataFrame(columns=["category", "topic", "meta_topic"])

    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    df.to_csv(f"{output_prefix}_message_topics.csv", index=False)
    print(f"  Saved: {output_prefix}_message_topics.csv")

    topics_df.to_csv(f"{output_prefix}_topic_alignment.csv", index=False)
    print(f"  Saved: {output_prefix}_topic_alignment.csv")

    print("\nDone!")


if __name__ == "__main__":
    run_topic_model(
        categories_csv_path="/Volumes/One Touch/terrorgram/channel_categories.csv",
        json_dir="/Volumes/One Touch/terrorgram/json_output",
        output_prefix="/Volumes/One Touch/terrorgram/topic_modelling/step1_labels/telegram_topics",
    )
