#!/usr/bin/env python3
"""
Transfer entropy between DCM features.

Tests whether changes in one psycholinguistic feature (e.g., outgroup othering)
precede changes in another (e.g., violent language) within each extremist
category, using multivariate transfer entropy (mTE).

Method:
  1. For each category, construct daily time series of the proportion of
     messages exhibiting each DCM feature.
  2. Drop days with fewer than MIN_DAILY_MSGS messages (noisy estimates).
  3. Split the series at calendar gaps into contiguous blocks. Blocks shorter
     than MIN_BLOCK_DAYS are discarded.
  4. Each contiguous block becomes a "replication" in IDTxl's terminology,
     which pools realisations across blocks to increase statistical power.
  5. Multivariate TE (MultivariateTE) with JidtGaussianCMI estimates the
     directed information flow from each source feature to each target feature,
     conditioning on all other features (so indirect paths are accounted for).
  6. Significance is assessed via permutation testing with FDR correction.

Output: dcm_annotations/analysis_results/D_transfer_entropy.csv
"""

import os
import numpy as np
import pandas as pd
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE

DATA_PATH = "/Volumes/One Touch/terrorgram/dcm_annotations/parsed_results/telegram_dcm_annotations.csv"
OUTPUT_DIR = "/Volumes/One Touch/terrorgram/dcm_annotations/analysis_results"

FEATURES = ["identityfusion_bin", "violentlang_bin", "threat_bin", "outgroup_othering_bin"]
FEATURE_LABELS = {
    "identityfusion_bin": "Identity Fusion",
    "violentlang_bin": "Violence Condoning",
    "threat_bin": "Existential Threat",
    "outgroup_othering_bin": "Outgroup Othering",
}

MIN_DAILY_MSGS = 20      # minimum messages per day to keep
MIN_BLOCK_DAYS = 30       # minimum contiguous days per replication
MAX_LAG = 7               # up to 7 days (one full week)
N_PERM = 500              # permutations for significance testing
ALPHA = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_daily_proportions(df_cat):
    """Aggregate binary features into daily proportions. Returns a DataFrame
    indexed by date with one column per feature, or None if insufficient data."""
    df_cat = df_cat.copy()
    df_cat["date"] = pd.to_datetime(df_cat["date"])
    for f in FEATURES:
        df_cat[f] = pd.to_numeric(df_cat[f], errors="coerce")

    daily = df_cat.groupby(df_cat["date"].dt.date).agg(
        n=("date", "size"),
        **{f: (f, "mean") for f in FEATURES}
    )
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    # Drop sparse days
    daily = daily[daily["n"] >= MIN_DAILY_MSGS].drop(columns="n")
    return daily


def split_contiguous_blocks(daily, min_days):
    """Split a date-indexed DataFrame at calendar gaps (>1 day) into
    contiguous blocks of at least min_days length."""
    if len(daily) < min_days:
        return []
    dates = daily.index
    gaps = (dates[1:] - dates[:-1]).days
    split_points = np.where(gaps > 1)[0] + 1
    indices = np.split(np.arange(len(daily)), split_points)
    blocks = [daily.iloc[idx] for idx in indices if len(idx) >= min_days]
    return blocks


def run_te_analysis(blocks, category):
    """Run multivariate TE on contiguous blocks for one category."""
    n_processes = len(FEATURES)

    # Keep only blocks at least half the length of the longest, so short
    # outlier blocks don't force excessive trimming of longer ones.
    blocks = sorted(blocks, key=len, reverse=True)
    longest = len(blocks[0])
    blocks = [b for b in blocks if len(b) >= longest // 2]

    min_len = min(len(b) for b in blocks)
    replication_arrays = []
    for block in blocks:
        arr = block[FEATURES].values[:min_len].T  # shape: (n_processes, min_len)
        replication_arrays.append(arr)

    # Stack into (processes, samples, replications)
    data_array = np.stack(replication_arrays, axis=2)

    data = Data(data_array, dim_order="psr", normalise=True, seed=42)

    settings = {
        "cmi_estimator": "JidtGaussianCMI",
        "max_lag_sources": MAX_LAG,
        "min_lag_sources": 1,
        "max_lag_target": MAX_LAG,
        "n_perm_max_stat": N_PERM,
        "n_perm_min_stat": N_PERM,
        "n_perm_omnibus": N_PERM,
        "alpha_max_stat": ALPHA,
        "alpha_min_stat": ALPHA,
        "alpha_omnibus": ALPHA,
        "fdr_correction": True,
        "verbose": False,
    }

    # With only 1 replication, permute within time rather than across replications
    if len(blocks) == 1:
        settings["permute_in_time"] = True

    mte = MultivariateTE()
    results = mte.analyse_network(settings=settings, data=data)
    return results, data, len(blocks), min_len


def extract_edges(results, category):
    """Extract significant directed edges from IDTxl results."""
    edges = []
    n_processes = len(FEATURES)
    for target in range(n_processes):
        try:
            res = results.get_single_target(target=target, fdr=True)
        except Exception:
            continue

        omnibus_te = res.get("omnibus_te", None)
        omnibus_p = res.get("omnibus_pval", None)
        selected = res.get("selected_vars_sources", [])
        te_vals = res.get("selected_sources_te", [])
        p_vals = res.get("selected_sources_pval", [])

        # Group by source process and aggregate
        source_te = {}
        for i, (proc, lag) in enumerate(selected):
            if proc == target:
                continue
            te_val = te_vals[i] if i < len(te_vals) else 0
            p_val = p_vals[i] if i < len(p_vals) else 1
            if proc not in source_te:
                source_te[proc] = {"te_sum": 0, "min_p": 1, "lags": []}
            source_te[proc]["te_sum"] += te_val
            source_te[proc]["min_p"] = min(source_te[proc]["min_p"], p_val)
            source_te[proc]["lags"].append(lag)

        for source, info in source_te.items():
            edges.append({
                "category": category,
                "source": FEATURE_LABELS[FEATURES[source]],
                "target": FEATURE_LABELS[FEATURES[target]],
                "te": round(info["te_sum"], 6),
                "p_value": info["min_p"],
                "lags": str(sorted(info["lags"])),
                "omnibus_te": round(omnibus_te, 6) if omnibus_te else None,
                "omnibus_p": omnibus_p,
            })

    return edges


# --- Main ---------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, usecols=["category", "date", *FEATURES],
                      dtype={"category": str}, low_memory=False)

    all_edges = []
    categories = sorted(df["category"].dropna().unique())
    for cat in categories:
        print(f"Processing {cat}...")
        df_cat = df[df["category"] == cat]
        daily = build_daily_proportions(df_cat)

        blocks = split_contiguous_blocks(daily, MIN_BLOCK_DAYS)
        print(f"  {len(daily)} days after filtering, {len(blocks)} contiguous block(s)")
        if not blocks:
            print(f"  Skipping {cat}: insufficient contiguous data.")
            continue

        block_lengths = [len(b) for b in blocks]
        print(f"  Block lengths: {block_lengths}")

        results, data, n_reps, samples_per_rep = run_te_analysis(blocks, cat)
        print(f"  Using {n_reps} replication(s) × {samples_per_rep} samples")

        edges = extract_edges(results, cat)
        all_edges.extend(edges)
        print(f"  Found {len(edges)} significant directed edge(s)")

    # Save combined results
    combined = pd.DataFrame(all_edges)
    combined.to_csv(os.path.join(OUTPUT_DIR, "transfer_entropy.csv"), index=False)
    print(f"\nDone. {len(all_edges)} total edges saved to {OUTPUT_DIR}/")
