#!/usr/bin/env python3
"""
Evaluate DCM validation: two human coding rounds + enrichment vs LLM annotations.
All LLM predictions come from dcm_llm_annotations.csv (420 messages, updated prompts).
Human labels: round1+round2 consensus for original 298, enrich_round1+round2 consensus for enrichment.
IRR computed on both original and enrichment samples (both have two rounds).

Usage:
  python dcm_validation_evaluate.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)

BASE_DIR = "/Volumes/One Touch/terrorgram/dcm_validation"
ROUND1_PATH = os.path.join(BASE_DIR, "round1.csv")
ROUND2_PATH = os.path.join(BASE_DIR, "round2.csv")
ENRICHMENT_R1_PATH = os.path.join(BASE_DIR, "enrich_round1.csv")
ENRICHMENT_R2_PATH = os.path.join(BASE_DIR, "enrich_round2.csv")
LLM_PATH = os.path.join(BASE_DIR, "dcm_llm_annotations.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "validation_results.csv")

FEATURES = ["identity_fusion", "outgroup_othering", "existential_threat", "violence_condoning"]


def parse_binary(val):
    s = str(val).strip().lower()
    if s in {"1", "yes", "true", "present"}:
        return 1
    if s in {"0", "no", "false", "absent", "nan", ""}:
        return 0
    for ch in s:
        if ch == "1":
            return 1
        if ch == "0":
            return 0
    return None


def compute_metrics(y_true, y_pred):
    return {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def print_metrics(m, indent=2):
    pad = " " * indent
    print(f"{pad}Accuracy:  {m['accuracy']:.4f}")
    print(f"{pad}Precision: {m['precision']:.4f}")
    print(f"{pad}Recall:    {m['recall']:.4f}")
    print(f"{pad}F1:        {m['f1']:.4f}")


def evaluate():
    # ------------------------------------------------------------------
    # Load all data
    # ------------------------------------------------------------------
    r1 = pd.read_csv(ROUND1_PATH, encoding="utf-8-sig")
    r2 = pd.read_csv(ROUND2_PATH, encoding="utf-8-sig")
    enrich_r1 = pd.read_csv(ENRICHMENT_R1_PATH, encoding="utf-8-sig")
    enrich_r2 = pd.read_csv(ENRICHMENT_R2_PATH, encoding="utf-8-sig")
    llm = pd.read_csv(LLM_PATH, encoding="utf-8-sig")

    for df in [r1, r2, enrich_r1, enrich_r2, llm]:
        df["message_id"] = df["message_id"].astype(str)

    for f in FEATURES:
        llm[f] = llm[f].apply(parse_binary)
    pre_len = len(llm)
    llm = llm.dropna(subset=FEATURES)
    if len(llm) < pre_len:
        print(f"Dropped {pre_len - len(llm)} LLM rows with unparseable responses")

    r1 = r1.dropna(subset=FEATURES)
    r2 = r2.dropna(subset=FEATURES)
    enrich_r1 = enrich_r1.dropna(subset=FEATURES)
    enrich_r2 = enrich_r2.dropna(subset=FEATURES)

    # ------------------------------------------------------------------
    # Original sample: merge round1 + round2 + LLM
    # ------------------------------------------------------------------
    orig = r1[["message_id"] + FEATURES].merge(
        r2[["message_id"] + FEATURES], on="message_id", suffixes=("_r1", "_r2")
    ).merge(
        llm[["message_id"] + FEATURES], on="message_id", suffixes=("", "_llm")
    )
    for f in FEATURES:
        if f in orig.columns and f"{f}_r1" in orig.columns:
            orig = orig.rename(columns={f: f"{f}_llm"})
    print(f"Original sample matched: {len(orig)}")

    # ------------------------------------------------------------------
    # Enrichment sample: merge enrichment_round1 + round2 + LLM
    # ------------------------------------------------------------------
    enrich = enrich_r1[["message_id"] + FEATURES].merge(
        enrich_r2[["message_id"] + FEATURES], on="message_id", suffixes=("_r1", "_r2")
    ).merge(
        llm[["message_id"] + FEATURES], on="message_id", suffixes=("", "_llm")
    )
    for f in FEATURES:
        if f in enrich.columns and f"{f}_r1" in enrich.columns:
            enrich = enrich.rename(columns={f: f"{f}_llm"})
    print(f"Enrichment sample matched: {len(enrich)}")

    # ==================================================================
    # INTERRATER RELIABILITY
    # ==================================================================
    irr_results = []

    for sample_name, sample_df in [("original", orig), ("enrichment", enrich)]:
        print(f"\n{'=' * 60}")
        print(f"INTERRATER RELIABILITY (Round 1 vs Round 2, {sample_name} sample)")
        print("=" * 60)

        all_r1_vals, all_r2_vals = [], []

        for feat in FEATURES:
            r1_vals = sample_df[f"{feat}_r1"].values.astype(int)
            r2_vals = sample_df[f"{feat}_r2"].values.astype(int)
            all_r1_vals.extend(r1_vals)
            all_r2_vals.extend(r2_vals)

            agree = (r1_vals == r2_vals).mean()
            kappa = cohen_kappa_score(r1_vals, r2_vals)

            irr_results.append({
                "sample": sample_name,
                "feature": feat,
                "n": len(r1_vals),
                "pct_agreement": round(agree, 4),
                "cohens_kappa": round(kappa, 4),
            })

            print(f"\n  {feat} (n={len(r1_vals)}):")
            print(f"    % Agreement:   {agree:.4f}")
            print(f"    Cohen's Kappa: {kappa:.4f}")

        all_r1_arr = np.array(all_r1_vals)
        all_r2_arr = np.array(all_r2_vals)
        print(f"\n  OVERALL POOLED (n={len(all_r1_arr)}):")
        print(f"    % Agreement:   {(all_r1_arr == all_r2_arr).mean():.4f}")
        print(f"    Cohen's Kappa: {cohen_kappa_score(all_r1_arr, all_r2_arr):.4f}")

    # ==================================================================
    # Build combined long-format: original consensus + enrichment consensus
    # ==================================================================
    combined_rows = []

    for sample_name, sample_df in [("original", orig), ("enrichment", enrich)]:
        for _, row in sample_df.iterrows():
            for feat in FEATURES:
                v1 = int(row[f"{feat}_r1"])
                v2 = int(row[f"{feat}_r2"])
                if v1 == v2:
                    combined_rows.append({
                        "message_id": row["message_id"],
                        "source": sample_name,
                        "feature": feat,
                        "human": v1,
                        "llm": int(row[f"{feat}_llm"]),
                    })

    combined = pd.DataFrame(combined_rows)

    n_orig = (combined["source"] == "original").sum()
    n_enrich = (combined["source"] == "enrichment").sum()

    # ==================================================================
    # LLM vs HUMAN — Combined
    # ==================================================================
    print(f"\n{'=' * 60}")
    print(f"LLM vs HUMAN (combined: {n_orig} original consensus + {n_enrich} enrichment)")
    print("=" * 60)

    results = []
    for feat in FEATURES:
        sub = combined[combined["feature"] == feat]
        sub_orig = sub[sub["source"] == "original"]
        sub_enrich = sub[sub["source"] == "enrichment"]
        m = compute_metrics(sub["human"].values, sub["llm"].values)
        m["comparison"] = "llm_vs_human_combined"
        m["feature"] = feat
        results.append(m)
        pos_total = (sub["human"] == 1).sum()
        pos_orig = (sub_orig["human"] == 1).sum()
        pos_enrich = (sub_enrich["human"] == 1).sum()
        print(f"\n  --- {feat} (n={m['n']}, positives={pos_total} [{pos_orig} orig + {pos_enrich} enrich]) ---")
        print_metrics(m, indent=4)

    m_all = compute_metrics(combined["human"].values, combined["llm"].values)
    pos_total = (combined["human"] == 1).sum()
    print(f"\n  OVERALL (n={m_all['n']}, positives={pos_total})")
    print_metrics(m_all, indent=4)
    m_all["comparison"] = "llm_vs_human_combined"
    m_all["feature"] = "OVERALL"
    results.append(m_all)

    # ==================================================================
    # LLM vs HUMAN — Original sample only
    # ==================================================================
    orig_only = combined[combined["source"] == "original"]
    print(f"\n{'=' * 60}")
    print(f"LLM vs HUMAN CONSENSUS (original sample only, n={len(orig_only)})")
    print("=" * 60)

    for feat in FEATURES:
        sub = orig_only[orig_only["feature"] == feat]
        m = compute_metrics(sub["human"].values, sub["llm"].values)
        m["comparison"] = "llm_vs_consensus_original"
        m["feature"] = feat
        results.append(m)
        print(f"\n  --- {feat} (n={m['n']}, positives={(sub['human']==1).sum()}) ---")
        print_metrics(m, indent=4)

    m_orig = compute_metrics(orig_only["human"].values, orig_only["llm"].values)
    print(f"\n  OVERALL (n={m_orig['n']})")
    print_metrics(m_orig, indent=4)
    m_orig["comparison"] = "llm_vs_consensus_original"
    m_orig["feature"] = "OVERALL"
    results.append(m_orig)

    # ==================================================================
    # LLM vs HUMAN — Enrichment sample only
    # ==================================================================
    enrich_only = combined[combined["source"] == "enrichment"]
    print(f"\n{'=' * 60}")
    print(f"LLM vs HUMAN (enrichment sample only, n={len(enrich_only)})")
    print("=" * 60)

    for feat in FEATURES:
        sub = enrich_only[enrich_only["feature"] == feat]
        m = compute_metrics(sub["human"].values, sub["llm"].values)
        m["comparison"] = "llm_vs_human_enrichment"
        m["feature"] = feat
        results.append(m)
        print(f"\n  --- {feat} (n={m['n']}, positives={(sub['human']==1).sum()}) ---")
        print_metrics(m, indent=4)

    m_enrich = compute_metrics(enrich_only["human"].values, enrich_only["llm"].values)
    print(f"\n  OVERALL (n={m_enrich['n']})")
    print_metrics(m_enrich, indent=4)
    m_enrich["comparison"] = "llm_vs_human_enrichment"
    m_enrich["feature"] = "OVERALL"
    results.append(m_enrich)

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")

    irr_df = pd.DataFrame(irr_results)
    irr_path = os.path.join(BASE_DIR, "interrater_reliability.csv")
    irr_df.to_csv(irr_path, index=False)
    print(f"Saved: {irr_path}")


if __name__ == "__main__":
    evaluate()
