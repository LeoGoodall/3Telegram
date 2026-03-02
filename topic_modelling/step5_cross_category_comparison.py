"""
Step 5: Cross-Category Topic Comparison
Analyze topic overlap and similarity across categories.
"""

import os
import pandas as pd
import numpy as np

# Paths
LABELS_PATH = "/Volumes/One Touch/terrorgram/topic_modelling/step1_labels/topic_labels.csv"
ALIGNMENT_PATH = "/Volumes/One Touch/terrorgram/topic_modelling/step1_labels/telegram_topics_topic_alignment.csv"
OUTPUT_DIR = "/Volumes/One Touch/terrorgram/topic_modelling/step5_cross_category"

INCLUDED_CATEGORIES = {"764", "terrorgram", "white-supremacist"}


def load_data():
    """Load topic labels and alignment data."""
    labels_df = pd.read_csv(LABELS_PATH)
    labels_df['category'] = labels_df['category'].astype(str)
    labels_df = labels_df[labels_df['category'].isin(INCLUDED_CATEGORIES)]

    alignment_df = pd.read_csv(ALIGNMENT_PATH)
    alignment_df['category'] = alignment_df['category'].astype(str)
    alignment_df = alignment_df[alignment_df['category'].isin(INCLUDED_CATEGORIES)]

    merged = labels_df.merge(alignment_df, on=['category', 'topic'], how='left')
    return merged



def analyze_meta_topics(df):
    """Analyze cross-category themes using meta_topic clusters."""
    results = []

    for meta_topic in sorted(df['meta_topic'].dropna().unique()):
        if meta_topic == -1:
            continue

        cluster = df[df['meta_topic'] == meta_topic]
        categories = cluster['category'].unique().tolist()
        labels = cluster['label'].tolist()

        results.append({
            'meta_topic': int(meta_topic),
            'n_topics': len(cluster),
            'categories': ', '.join(sorted(categories)),
            'n_categories': len(categories),
            'cross_category': len(categories) > 1,
            'topic_labels': ' | '.join(labels)
        })

    return pd.DataFrame(results)



def compute_category_overlap(labels_df):
    """Compute vocabulary overlap between categories."""
    category_terms = {}

    for category in labels_df['category'].unique():
        cat_df = labels_df[labels_df['category'] == category]
        all_terms = set()

        for terms in cat_df['top_terms']:
            if isinstance(terms, str):
                term_list = [t.strip().strip("'\"").lower() for t in terms.split(',')]
                all_terms.update(term_list)

        category_terms[category] = all_terms

    categories = sorted(category_terms.keys())
    overlap_results = []

    for cat1 in categories:
        for cat2 in categories:
            terms1 = category_terms[cat1]
            terms2 = category_terms[cat2]
            intersection = len(terms1 & terms2)
            union = len(terms1 | terms2)
            jaccard = intersection / union if union > 0 else 0

            overlap_results.append({
                'category_1': cat1,
                'category_2': cat2,
                'intersection': intersection,
                'union': union,
                'jaccard': jaccard
            })

    shared_all = category_terms['764'] & category_terms['terrorgram'] & category_terms['white-supremacist']

    return pd.DataFrame(overlap_results), category_terms, shared_all



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    meta_topics_df = analyze_meta_topics(df)
    overlap_df, category_terms, shared_all = compute_category_overlap(df)

    meta_topics_df.to_csv(f"{OUTPUT_DIR}/meta_topics_analysis.csv", index=False)
    overlap_df.to_csv(f"{OUTPUT_DIR}/category_overlap.csv", index=False)

    shared_df = pd.DataFrame({'term': sorted(shared_all)})
    shared_df.to_csv(f"{OUTPUT_DIR}/shared_vocabulary.csv", index=False)


if __name__ == "__main__":
    main()
