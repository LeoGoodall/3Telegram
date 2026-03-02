#!/usr/bin/env python3
"""
Topic Analysis Script
- Count topics per category
- Extract top terms per topic using c-TF-IDF
- Assign labels to topics
- Visualize topic distributions
- Cross-category overlap analysis
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import matplotlib.pyplot as plt

# Custom stopwords: match step1_topic_modelling.py
CUSTOM_STOPWORDS = [
    "like", "just", "don", "dont", "doesn", "doesnt",
    "got", "good", "think", "know", "ur", "im",
    "lol", "yeah", "ve", "want", "use", "need", "people",
    "matter", "matters", "depends", "does", "yes", "no", "really", "oh", "gonna",
    "did", "make", "made", "time", "times"
]

# File paths
MESSAGE_TOPICS_PATH = "/Volumes/One Touch/terrorgram/topic_modelling/step1_labels/telegram_topics_message_topics.csv"
TOPIC_ALIGNMENT_PATH = "/Volumes/One Touch/terrorgram/topic_modelling/step1_labels/telegram_topics_topic_alignment.csv"
OUTPUT_DIR = "/Volumes/One Touch/terrorgram/topic_modelling/step2_topics"


def load_data():
    """Load the topic modeling results."""
    messages_df = pd.read_csv(MESSAGE_TOPICS_PATH, low_memory=False)
    alignment_df = pd.read_csv(TOPIC_ALIGNMENT_PATH)

    # Ensure category is string type (fixes mixed int/str issue)
    messages_df["category"] = messages_df["category"].astype(str)
    alignment_df["category"] = alignment_df["category"].astype(str)

    return messages_df, alignment_df


def count_topics_per_category(df):
    """Count number of topics per category (excluding noise cluster -1)."""
    print("=" * 60)
    print("TOPICS PER CATEGORY")
    print("=" * 60)

    summary = []
    for category in sorted(df["category"].unique()):
        cat_df = df[df["category"] == category]
        topics = cat_df["topic"].unique()
        valid_topics = [t for t in topics if t != -1]
        noise_count = len(cat_df[cat_df["topic"] == -1])
        total_messages = len(cat_df)

        print(f"\n{category}:")
        print(f"  Topics: {len(valid_topics)}")
        print(f"  Messages in topics: {total_messages - noise_count:,}")
        print(f"  Noise (unclustered): {noise_count:,} ({100*noise_count/total_messages:.1f}%)")

        summary.append({
            "category": category,
            "num_topics": len(valid_topics),
            "messages_in_topics": total_messages - noise_count,
            "noise_messages": noise_count,
            "noise_pct": 100 * noise_count / total_messages
        })

    return pd.DataFrame(summary)


def extract_topic_terms(df, n_terms=10):
    """
    Extract top terms per topic using c-TF-IDF.
    c-TF-IDF: treats all documents in a topic as one document,
    then computes TF-IDF across topics.
    """
    print("\n" + "=" * 60)
    print("TOP TERMS PER TOPIC (c-TF-IDF)")
    print("=" * 60)

    topic_terms = []

    for category in sorted(df["category"].unique()):
        cat_df = df[df["category"] == category]
        topics = sorted([t for t in cat_df["topic"].unique() if t != -1])

        if not topics:
            continue

        # Concatenate all texts per topic
        topic_docs = []
        topic_ids = []
        for topic_id in topics:
            texts = cat_df[cat_df["topic"] == topic_id]["text"].tolist()
            topic_docs.append(" ".join(texts))
            topic_ids.append(topic_id)

        # Compute TF-IDF across topic-documents
        # Extend standard English stopwords with custom informal terms
        from sklearn.feature_extraction.text import CountVectorizer
        base_stops = list(CountVectorizer(stop_words="english").get_stop_words())
        all_stops = base_stops + CUSTOM_STOPWORDS

        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=all_stops,
            ngram_range=(1, 2),
            min_df=2
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(topic_docs)
        except ValueError:
            continue

        feature_names = vectorizer.get_feature_names_out()

        print(f"\n{category.upper()}")
        print("-" * 40)

        for idx, topic_id in enumerate(topic_ids):
            scores = tfidf_matrix[idx].toarray().flatten()
            top_indices = scores.argsort()[-n_terms:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            top_scores = [scores[i] for i in top_indices]

            # Show top 10 terms as the label
            label = ", ".join(top_terms[:10])

            msg_count = len(cat_df[cat_df["topic"] == topic_id])

            print(f"\n  Topic {topic_id} ({msg_count:,} messages)")
            print(f"  Label: {label}")
            print(f"  Terms: {', '.join(top_terms)}")

            topic_terms.append({
                "category": category,
                "topic": topic_id,
                "message_count": msg_count,
                "label": label,
                "top_terms": top_terms,
                "top_scores": top_scores
            })

    return pd.DataFrame(topic_terms)


def visualize_topic_distributions(df, topic_terms_df):
    """Create visualizations of topic distributions."""
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    categories = sorted(df["category"].unique())
    n_cats = len(categories)

    # Figure 1: Topic size distribution per category
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, category in enumerate(categories):
        ax = axes[idx]
        cat_df = df[df["category"] == category]

        topic_counts = cat_df[cat_df["topic"] != -1].groupby("topic").size().sort_values(ascending=True)

        if len(topic_counts) == 0:
            ax.set_title(f"{category}\n(No topics found)")
            continue

        # Get labels for topics
        cat_terms = topic_terms_df[topic_terms_df["category"] == category]
        labels = []
        for topic_id in topic_counts.index:
            term_row = cat_terms[cat_terms["topic"] == topic_id]
            if not term_row.empty:
                label = term_row.iloc[0]["label"][:30] + "..." if len(term_row.iloc[0]["label"]) > 30 else term_row.iloc[0]["label"]
                labels.append(f"T{topic_id}: {label}")
            else:
                labels.append(f"Topic {topic_id}")

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(topic_counts)))
        bars = ax.barh(range(len(topic_counts)), topic_counts.values, color=colors)
        ax.set_yticks(range(len(topic_counts)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Number of Messages")
        ax.set_title(f"{category}\n({len(topic_counts)} topics)")

        # Add count labels
        for bar, count in zip(bars, topic_counts.values):
            ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                   f"{count:,}", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/topic_distributions.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR}/topic_distributions.png")
    plt.close()

    # Figure 2: Category comparison - proportion of messages per topic
    fig, ax = plt.subplots(figsize=(12, 6))

    category_topic_counts = []
    for category in categories:
        cat_df = df[df["category"] == category]
        total = len(cat_df)
        in_topics = len(cat_df[cat_df["topic"] != -1])
        noise = len(cat_df[cat_df["topic"] == -1])
        n_topics = len([t for t in cat_df["topic"].unique() if t != -1])
        category_topic_counts.append({
            "category": category,
            "in_topics": in_topics,
            "noise": noise,
            "n_topics": n_topics
        })

    comp_df = pd.DataFrame(category_topic_counts)
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, comp_df["in_topics"], width, label="In Topics", color="steelblue")
    bars2 = ax.bar(x + width/2, comp_df["noise"], width, label="Noise/Unclustered", color="lightcoral")

    ax.set_ylabel("Number of Messages")
    ax.set_xlabel("Category")
    ax.set_title("Messages in Topics vs Noise by Category")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15)
    ax.legend()

    # Add topic count annotation
    for i, (bar, n) in enumerate(zip(bars1, comp_df["n_topics"])):
        ax.annotate(f"{n} topics", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/category_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR}/category_comparison.png")
    plt.close()


def cross_category_overlap_analysis(messages_df, alignment_df, topic_terms_df):
    """Analyze which topics from different categories cluster together."""
    print("\n" + "=" * 60)
    print("CROSS-CATEGORY TOPIC OVERLAP ANALYSIS")
    print("=" * 60)

    if alignment_df.empty or "meta_topic" not in alignment_df.columns:
        print("  No meta-topic alignment data available.")
        return None

    # Group by meta_topic to find overlapping topics
    meta_groups = alignment_df.groupby("meta_topic")

    overlap_results = []

    print("\nMeta-topics that span multiple categories:")
    print("-" * 50)

    for meta_topic, group in meta_groups:
        if meta_topic == -1:
            continue  # Skip noise

        categories_in_meta = group["category"].unique()

        if len(categories_in_meta) > 1:
            print(f"\n  Meta-topic {meta_topic}:")

            for _, row in group.iterrows():
                cat = row["category"]
                topic = row["topic"]

                # Get label for this topic
                term_row = topic_terms_df[
                    (topic_terms_df["category"] == cat) &
                    (topic_terms_df["topic"] == topic)
                ]

                if not term_row.empty:
                    label = term_row.iloc[0]["label"]
                    msg_count = term_row.iloc[0]["message_count"]
                    print(f"    {cat} Topic {topic} ({msg_count:,} msgs): {label}")
                else:
                    print(f"    {cat} Topic {topic}")

                overlap_results.append({
                    "meta_topic": meta_topic,
                    "category": cat,
                    "topic": topic,
                    "label": term_row.iloc[0]["label"] if not term_row.empty else ""
                })

    # Summary statistics
    print("\n" + "-" * 50)
    print("\nOverlap Summary:")

    multi_cat_metas = [mt for mt, g in meta_groups if len(g["category"].unique()) > 1 and mt != -1]
    single_cat_metas = [mt for mt, g in meta_groups if len(g["category"].unique()) == 1 and mt != -1]

    print(f"  Meta-topics spanning multiple categories: {len(multi_cat_metas)}")
    print(f"  Meta-topics within single category: {len(single_cat_metas)}")

    # Which categories share the most topics?
    if overlap_results:
        overlap_df = pd.DataFrame(overlap_results)

        print("\n  Category pairs with shared meta-topics:")
        for meta_topic in multi_cat_metas:
            cats = overlap_df[overlap_df["meta_topic"] == meta_topic]["category"].tolist()
            if len(cats) >= 2:
                for i, c1 in enumerate(cats):
                    for c2 in cats[i+1:]:
                        print(f"    {c1} <-> {c2} (meta-topic {meta_topic})")

    return pd.DataFrame(overlap_results) if overlap_results else None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    messages_df, alignment_df = load_data()

    print(f"Loaded {len(messages_df):,} messages")
    print(f"Categories: {messages_df['category'].unique().tolist()}")

    # 1. Count topics per category
    summary_df = count_topics_per_category(messages_df)

    # 2. Extract top terms per topic
    topic_terms_df = extract_topic_terms(messages_df)

    # 3. Visualize
    visualize_topic_distributions(messages_df, topic_terms_df)

    # 4. Cross-category overlap
    overlap_df = cross_category_overlap_analysis(messages_df, alignment_df, topic_terms_df)

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    summary_df.to_csv(f"{OUTPUT_DIR}/topic_summary.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/topic_summary.csv")

    topic_terms_df.to_csv(f"{OUTPUT_DIR}/topic_terms.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/topic_terms.csv")

    if overlap_df is not None and not overlap_df.empty:
        overlap_df.to_csv(f"{OUTPUT_DIR}/topic_overlap.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR}/topic_overlap.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
