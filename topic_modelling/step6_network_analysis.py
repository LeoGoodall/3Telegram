"""
Step 6: Topic Network Analysis
Map relationships between topics and channels, and co-occurring topics.
"""

import os
import pandas as pd
import numpy as np

# Paths
MESSAGE_TOPICS_PATH = "/Volumes/One Touch/terrorgram/topic_modelling/step1_labels/telegram_topics_message_topics.csv"
LABELS_PATH = "/Volumes/One Touch/terrorgram/topic_modelling/step1_labels/topic_labels.csv"
OUTPUT_DIR = "/Volumes/One Touch/terrorgram/topic_modelling/step6_network"

INCLUDED_CATEGORIES = {"764", "terrorgram", "white-supremacist"}


def load_data():
    """Load message topics and labels."""
    messages_df = pd.read_csv(MESSAGE_TOPICS_PATH, low_memory=False)
    messages_df['category'] = messages_df['category'].astype(str)
    messages_df = messages_df[messages_df['category'].isin(INCLUDED_CATEGORIES)]
    messages_df = messages_df[messages_df['topic'] >= 0]

    labels_df = pd.read_csv(LABELS_PATH)
    labels_df['category'] = labels_df['category'].astype(str)
    labels_df = labels_df[labels_df['category'].isin(INCLUDED_CATEGORIES)]

    return messages_df, labels_df


def analyze_channel_topic_distribution(messages_df, labels_df):
    """Analyze which topics dominate which channels."""
    results = []

    for category in sorted(messages_df['category'].unique()):
        cat_messages = messages_df[messages_df['category'] == category]
        cat_labels = labels_df[labels_df['category'] == category]
        label_lookup = dict(zip(cat_labels['topic'], cat_labels['label']))

        for channel in cat_messages['channel'].unique():
            channel_msgs = cat_messages[cat_messages['channel'] == channel]
            total = len(channel_msgs)

            if total < 10:
                continue

            topic_counts = channel_msgs['topic'].value_counts()
            dominant_topic = topic_counts.index[0]
            dominant_count = topic_counts.values[0]
            dominant_label = label_lookup.get(dominant_topic, f"Topic {dominant_topic}")

            top3_topics = []
            for t, c in topic_counts.head(3).items():
                lbl = label_lookup.get(t, f"Topic {t}")
                top3_topics.append(f"{lbl[:20]} ({100*c/total:.0f}%)")

            results.append({
                'category': category,
                'channel': channel,
                'total_messages': total,
                'n_topics': len(topic_counts),
                'dominant_topic': dominant_topic,
                'dominant_label': dominant_label,
                'dominant_pct': 100 * dominant_count / total,
                'top3_topics': ' | '.join(top3_topics)
            })

    return pd.DataFrame(results)


def compute_topic_cooccurrence(messages_df, labels_df):
    """Compute topic co-occurrence within channels."""
    results = []

    for category in sorted(messages_df['category'].unique()):
        cat_messages = messages_df[messages_df['category'] == category]
        cat_labels = labels_df[labels_df['category'] == category]

        topics = sorted(cat_labels['topic'].unique())
        n_topics = len(topics)
        cooccur_matrix = np.zeros((n_topics, n_topics))

        for channel in cat_messages['channel'].unique():
            channel_topics = set(cat_messages[cat_messages['channel'] == channel]['topic'].unique())

            for i, t1 in enumerate(topics):
                for j, t2 in enumerate(topics):
                    if t1 in channel_topics and t2 in channel_topics:
                        cooccur_matrix[i, j] += 1

        diag = np.diag(cooccur_matrix)
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_matrix = cooccur_matrix / np.sqrt(np.outer(diag, diag))
            norm_matrix = np.nan_to_num(norm_matrix)

        label_lookup = dict(zip(cat_labels['topic'], cat_labels['label']))

        for i, t1 in enumerate(topics):
            for j, t2 in enumerate(topics):
                if i < j:
                    results.append({
                        'category': category,
                        'topic_1': t1,
                        'topic_2': t2,
                        'label_1': label_lookup.get(t1, f"Topic {t1}"),
                        'label_2': label_lookup.get(t2, f"Topic {t2}"),
                        'cooccur_count': int(cooccur_matrix[i, j]),
                        'cooccur_norm': norm_matrix[i, j]
                    })

    return pd.DataFrame(results)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    messages_df, labels_df = load_data()
    channel_topics_df = analyze_channel_topic_distribution(messages_df, labels_df)
    cooccur_df = compute_topic_cooccurrence(messages_df, labels_df)

    channel_topics_df.to_csv(f"{OUTPUT_DIR}/channel_topic_distribution.csv", index=False)
    cooccur_df.to_csv(f"{OUTPUT_DIR}/topic_cooccurrence.csv", index=False)


if __name__ == "__main__":
    main()
