"""
Topic Validation Script
- Step 2: Calculate topic coherence scores (NPMI-based)
- Step 3: Extract representative documents per topic
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import re
import math

# File paths
MESSAGE_TOPICS_PATH = "/Volumes/One Touch/terrorgram/topic_modelling/step1_labels/telegram_topics_message_topics.csv"
TOPIC_LABELS_PATH = "/Volumes/One Touch/terrorgram/topic_modelling/step1_labels/topic_labels.csv"
OUTPUT_DIR = "/Volumes/One Touch/terrorgram/topic_modelling/step3_validation"

# Categories to analyze (excluding proud-boys due to insufficient data)
INCLUDED_CATEGORIES = {"764", "terrorgram", "white-supremacist"}


def load_data():
    """Load message topics and labels."""
    print("Loading data...")
    messages_df = pd.read_csv(MESSAGE_TOPICS_PATH, low_memory=False)
    messages_df["category"] = messages_df["category"].astype(str)

    labels_df = pd.read_csv(TOPIC_LABELS_PATH)
    labels_df["category"] = labels_df["category"].astype(str)
    labels_df["topic"] = labels_df["topic"].astype(int)

    # Filter to included categories
    messages_df = messages_df[messages_df["category"].isin(INCLUDED_CATEGORIES)]
    labels_df = labels_df[labels_df["category"].isin(INCLUDED_CATEGORIES)]

    print(f"  Loaded {len(messages_df):,} messages across {messages_df['category'].nunique()} categories")
    return messages_df, labels_df


def tokenize(text):
    """Simple tokenization for coherence calculation."""
    if not isinstance(text, str):
        return set()
    # Lowercase and extract words
    tokens = set(re.findall(r'\b[a-z]{2,}\b', text.lower()))
    return tokens


def calculate_npmi_coherence(topic_words, texts_tokenized, word_doc_freq, n_docs):
    """
    Calculate Normalized Pointwise Mutual Information (NPMI) coherence.

    NPMI ranges from -1 to 1, where:
    - 1 = perfect co-occurrence
    - 0 = independent
    - -1 = never co-occur

    We calculate pairwise NPMI for top words and average.
    """
    if len(topic_words) < 2:
        return 0.0

    epsilon = 1e-12
    npmi_scores = []

    for i in range(len(topic_words)):
        for j in range(i + 1, len(topic_words)):
            w1, w2 = topic_words[i], topic_words[j]

            # P(w1) and P(w2)
            p_w1 = word_doc_freq.get(w1, 0) / n_docs
            p_w2 = word_doc_freq.get(w2, 0) / n_docs

            if p_w1 < epsilon or p_w2 < epsilon:
                continue

            # P(w1, w2) - co-occurrence probability
            co_occur = sum(1 for doc in texts_tokenized if w1 in doc and w2 in doc)
            p_w1_w2 = co_occur / n_docs

            if p_w1_w2 < epsilon:
                npmi = -1
            else:
                # PMI = log(P(w1,w2) / (P(w1) * P(w2)))
                pmi = math.log(p_w1_w2 / (p_w1 * p_w2))
                # Normalize by -log(P(w1,w2))
                npmi = pmi / (-math.log(p_w1_w2))

            npmi_scores.append(npmi)

    if not npmi_scores:
        return 0.0

    return np.mean(npmi_scores)


def calculate_topic_diversity(labels_df):
    """
    Calculate topic diversity - proportion of unique words across all topics.
    Higher diversity = topics are more distinct from each other.
    """
    diversity_results = []

    for category in sorted(labels_df["category"].unique()):
        cat_labels = labels_df[labels_df["category"] == category]

        all_words = []
        for _, row in cat_labels.iterrows():
            terms = row["top_terms"]
            if isinstance(terms, str):
                words = [t.strip().strip("'\"") for t in terms.split(",")]
                all_words.extend(words[:10])

        unique_words = set(all_words)
        diversity = len(unique_words) / len(all_words) if all_words else 0

        diversity_results.append({
            "category": category,
            "total_words": len(all_words),
            "unique_words": len(unique_words),
            "diversity": diversity
        })

    return pd.DataFrame(diversity_results)


def calculate_coherence_scores(messages_df, labels_df):
    """
    Calculate topic coherence scores using NPMI metric.
    """
    print("\n" + "=" * 60)
    print("STEP 2: TOPIC COHERENCE SCORES (NPMI)")
    print("=" * 60)

    coherence_results = []

    for category in sorted(messages_df["category"].unique()):
        print(f"\nProcessing {category}...")

        cat_messages = messages_df[messages_df["category"] == category]
        cat_labels = labels_df[labels_df["category"] == category]

        # Tokenize all texts
        all_texts = cat_messages["text"].dropna().tolist()
        texts_tokenized = [tokenize(t) for t in all_texts]
        n_docs = len(texts_tokenized)

        # Build word-document frequency
        word_doc_freq = defaultdict(int)
        for doc_tokens in texts_tokenized:
            for token in doc_tokens:
                word_doc_freq[token] += 1

        print(f"  Documents: {n_docs:,}")
        print(f"  Unique words: {len(word_doc_freq):,}")

        # Calculate coherence per topic
        for _, row in cat_labels.iterrows():
            topic_id = int(row["topic"])
            label = row["label"]
            top_terms = row["top_terms"]

            # Parse top terms
            if isinstance(top_terms, str):
                terms = [t.strip().strip("'\"") for t in top_terms.split(",")]
                terms = terms[:10]  # Top 10 words
            else:
                continue

            # Calculate NPMI coherence
            npmi = calculate_npmi_coherence(terms, texts_tokenized, word_doc_freq, n_docs)

            coherence_results.append({
                "category": category,
                "topic": topic_id,
                "label": label,
                "coherence_npmi": npmi
            })

            print(f"    Topic {topic_id}: NPMI={npmi:.4f} | {label}")

    coherence_df = pd.DataFrame(coherence_results)

    # Calculate diversity
    print("\n" + "-" * 60)
    print("TOPIC DIVERSITY BY CATEGORY")
    print("-" * 60)
    diversity_df = calculate_topic_diversity(labels_df)
    for _, row in diversity_df.iterrows():
        print(f"  {row['category']}: {row['diversity']:.2%} unique ({row['unique_words']}/{row['total_words']} words)")

    # Print summary
    print("\n" + "-" * 60)
    print("COHERENCE SUMMARY BY CATEGORY")
    print("-" * 60)

    for category in sorted(coherence_df["category"].unique()):
        cat_scores = coherence_df[coherence_df["category"] == category]["coherence_npmi"]
        print(f"\n{category}:")
        print(f"  Mean NPMI:     {cat_scores.mean():.4f}")
        print(f"  Std deviation: {cat_scores.std():.4f}")
        print(f"  Min NPMI:      {cat_scores.min():.4f}")
        print(f"  Max NPMI:      {cat_scores.max():.4f}")

    # Print top and bottom topics
    print("\n" + "-" * 60)
    print("TOP 10 MOST COHERENT TOPICS")
    print("-" * 60)
    top_10 = coherence_df.nlargest(10, "coherence_npmi")
    for _, row in top_10.iterrows():
        print(f"  {row['coherence_npmi']:.4f} | {row['category']:20} | {row['label']}")

    print("\n" + "-" * 60)
    print("BOTTOM 10 LEAST COHERENT TOPICS")
    print("-" * 60)
    bottom_10 = coherence_df.nsmallest(10, "coherence_npmi")
    for _, row in bottom_10.iterrows():
        print(f"  {row['coherence_npmi']:.4f} | {row['category']:20} | {row['label']}")

    return coherence_df, diversity_df


def extract_representative_documents(messages_df, labels_df, n_docs=10):
    """
    Extract representative documents for each topic.
    """
    print("\n" + "=" * 60)
    print("STEP 3: REPRESENTATIVE DOCUMENTS")
    print("=" * 60)

    representative_docs = []

    for category in sorted(messages_df["category"].unique()):
        print(f"\nProcessing {category}...")

        cat_messages = messages_df[messages_df["category"] == category]
        cat_labels = labels_df[labels_df["category"] == category]

        for _, label_row in cat_labels.iterrows():
            topic_id = int(label_row["topic"])
            label = label_row["label"]

            # Get messages for this topic
            topic_messages = cat_messages[cat_messages["topic"] == topic_id]["text"].dropna()

            if len(topic_messages) == 0:
                continue

            # Calculate message lengths
            msg_lengths = topic_messages.str.len()

            # Get longer messages (above median length)
            median_len = msg_lengths.median()
            longer_msgs = topic_messages[msg_lengths >= median_len]

            if len(longer_msgs) == 0:
                longer_msgs = topic_messages

            # Sample up to n_docs
            n_sample = min(n_docs, len(longer_msgs))
            sampled = longer_msgs.sample(n=n_sample, random_state=42)

            for i, text in enumerate(sampled):
                # Truncate very long messages
                display_text = text[:500] + "..." if len(text) > 500 else text

                representative_docs.append({
                    "category": category,
                    "topic": topic_id,
                    "label": label,
                    "doc_rank": i + 1,
                    "text": display_text,
                    "full_length": len(text)
                })

        print(f"  Extracted {len([d for d in representative_docs if d['category'] == category])} documents")

    rep_docs_df = pd.DataFrame(representative_docs)

    # Print samples (first 5 topics per category)
    print("\n" + "-" * 60)
    print("SAMPLE REPRESENTATIVE DOCUMENTS (3 per topic, first 5 topics)")
    print("-" * 60)

    for category in sorted(rep_docs_df["category"].unique()):
        print(f"\n{'='*60}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'='*60}")

        cat_docs = rep_docs_df[rep_docs_df["category"] == category]

        for topic_id in sorted(cat_docs["topic"].unique())[:5]:
            topic_docs = cat_docs[cat_docs["topic"] == topic_id]
            label = topic_docs["label"].values[0]

            print(f"\n--- Topic {topic_id}: {label} ---")

            for _, row in topic_docs.head(3).iterrows():
                text = row["text"].replace("\n", " ")[:150]
                print(f"  • {text}...")

    return rep_docs_df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    messages_df, labels_df = load_data()

    # Step 2: Calculate coherence
    coherence_df, diversity_df = calculate_coherence_scores(messages_df, labels_df)

    # Step 3: Extract representative documents
    rep_docs_df = extract_representative_documents(messages_df, labels_df, n_docs=10)

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    coherence_df.to_csv(f"{OUTPUT_DIR}/topic_coherence.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/topic_coherence.csv")

    diversity_df.to_csv(f"{OUTPUT_DIR}/topic_diversity.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/topic_diversity.csv")

    rep_docs_df.to_csv(f"{OUTPUT_DIR}/topic_representative_docs.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/topic_representative_docs.csv")

    # Combine labels with coherence
    summary_df = labels_df.merge(
        coherence_df[["category", "topic", "coherence_npmi"]],
        on=["category", "topic"],
        how="left"
    )
    summary_df.to_csv(f"{OUTPUT_DIR}/topic_labels_with_coherence.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/topic_labels_with_coherence.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
