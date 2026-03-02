"""
Step 4: Temporal Analysis
Track topic prevalence over time to identify emerging/declining topics and event-driven spikes.
"""

import json
import os
import pandas as pd
import numpy as np

# Paths
JSON_DIR = "/Volumes/One Touch/terrorgram/json_output"
MESSAGE_TOPICS_PATH = "/Volumes/One Touch/terrorgram/topic_modelling/step1_labels/telegram_topics_message_topics.csv"
LABELS_PATH = "/Volumes/One Touch/terrorgram/topic_modelling/step1_labels/topic_labels.csv"
OUTPUT_DIR = "/Volumes/One Touch/terrorgram/topic_modelling/step4_temporal"

INCLUDED_CATEGORIES = {"764", "terrorgram", "white-supremacist"}


def extract_timestamps_from_json():
    """Extract timestamps from original JSON files."""
    timestamp_lookup = {}
    files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]

    for fname in files:
        try:
            with open(os.path.join(JSON_DIR, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'messages' not in data:
                continue

            for msg in data['messages']:
                ts = msg.get('timestamp')
                if ts is None or not isinstance(ts, dict):
                    continue

                date_str = ts.get('date')
                if not date_str:
                    continue

                content = msg.get('content')
                if isinstance(content, dict):
                    text = content.get('text', '')
                else:
                    text = msg.get('text', '')

                if isinstance(text, list):
                    parts = []
                    for part in text:
                        if isinstance(part, str):
                            parts.append(part)
                        elif isinstance(part, dict):
                            parts.append(part.get('text', ''))
                    text = ' '.join(parts)

                if not text or not isinstance(text, str):
                    continue

                text_key = text[:100].lower().strip()
                timestamp_lookup[text_key] = date_str

        except Exception:
            continue

    return timestamp_lookup


def load_and_merge_timestamps():
    """Load message topics and merge with timestamps."""
    df = pd.read_csv(MESSAGE_TOPICS_PATH, low_memory=False)
    df['category'] = df['category'].astype(str)
    df = df[df['category'].isin(INCLUDED_CATEGORIES)]
    df = df[df['topic'] >= 0]

    timestamp_lookup = extract_timestamps_from_json()

    dates = []
    for text in df['text']:
        if isinstance(text, str):
            key = text[:100].lower().strip()
            dates.append(timestamp_lookup.get(key))
        else:
            dates.append(None)

    df['date'] = dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    return df


def compute_temporal_trends(df, labels_df):
    """Compute topic prevalence over time."""
    df_dated = df[df['date'].notna()].copy()
    df_dated['year_month'] = df_dated['date'].dt.to_period('M')

    results = []

    for category in sorted(df_dated['category'].unique()):
        cat_df = df_dated[df_dated['category'] == category]
        cat_labels = labels_df[labels_df['category'] == category]

        monthly = cat_df.groupby(['year_month', 'topic']).size().unstack(fill_value=0)
        monthly_prop = monthly.div(monthly.sum(axis=1), axis=0)

        for topic_id in monthly.columns:
            label_row = cat_labels[cat_labels['topic'] == topic_id]
            label = label_row['label'].values[0] if len(label_row) > 0 else f"Topic {topic_id}"

            for period in monthly.index:
                results.append({
                    'category': category,
                    'topic': int(topic_id),
                    'label': label,
                    'year_month': str(period),
                    'count': int(monthly.loc[period, topic_id]),
                    'proportion': float(monthly_prop.loc[period, topic_id])
                })

    return pd.DataFrame(results)


TREND_CUTOFF = "2023-01"  # Fixed cutoff: Pre-2023 vs Post-2023


def identify_trends(temporal_df):
    """Identify emerging and declining topics using a fixed cutoff date."""
    trend_results = []

    for category in temporal_df['category'].unique():
        cat_df = temporal_df[temporal_df['category'] == category]

        for topic in cat_df['topic'].unique():
            topic_df = cat_df[cat_df['topic'] == topic].sort_values('year_month')

            if len(topic_df) < 3:
                continue

            label = topic_df['label'].values[0]

            pre = topic_df[topic_df['year_month'] < TREND_CUTOFF]['proportion']
            post = topic_df[topic_df['year_month'] >= TREND_CUTOFF]['proportion']

            first_half = pre.mean() if len(pre) > 0 else 0
            second_half = post.mean() if len(post) > 0 else 0

            if first_half > 0:
                change_pct = (second_half - first_half) / first_half * 100
            else:
                change_pct = 100 if second_half > 0 else 0

            if change_pct > 50:
                trend = "Emerging"
            elif change_pct < -50:
                trend = "Declining"
            else:
                trend = "Stable"

            trend_results.append({
                'category': category,
                'topic': topic,
                'label': label,
                'first_half_avg': first_half,
                'second_half_avg': second_half,
                'change_pct': change_pct,
                'trend': trend
            })

    return pd.DataFrame(trend_results)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    labels_df = pd.read_csv(LABELS_PATH)
    labels_df['category'] = labels_df['category'].astype(str)
    labels_df['topic'] = labels_df['topic'].astype(int)

    df = load_and_merge_timestamps()
    temporal_df = compute_temporal_trends(df, labels_df)
    trends_df = identify_trends(temporal_df)

    temporal_df.to_csv(f"{OUTPUT_DIR}/topic_temporal.csv", index=False)
    trends_df.to_csv(f"{OUTPUT_DIR}/topic_trends.csv", index=False)


if __name__ == "__main__":
    main()
