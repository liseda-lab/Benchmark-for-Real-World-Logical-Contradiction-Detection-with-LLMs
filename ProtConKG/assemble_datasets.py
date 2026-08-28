import pandas as pd
import argparse
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a balanced benchmark dataset")
    parser.add_argument("--input-positive", required=True)
    parser.add_argument("--input-negative", required=True)
    parser.add_argument("--size", type=int, default=2500, help="Target size of the positive set")
    return parser.parse_args()

def extract_subject(triple_str: str) -> str:
    if pd.isna(triple_str):
        return None
    parts = str(triple_str).strip("()").split(";")
    return parts[0].strip() if len(parts) > 0 else None

def extract_relation(triple_str: str) -> str:
    if pd.isna(triple_str):
        return None
    parts = str(triple_str).strip("()").split(";")
    if len(parts) < 2:
        return None
    rel = parts[1].strip()
    # Normalise negative assertions: "NOT enables" → "enables"
    return rel.removeprefix("NOT ").strip()

def get_diverse_sample(df: pd.DataFrame, target_size: int) -> pd.DataFrame:
    """
    Samples rows to maximise coverage of both subjects and relations.
    Iterates in round-robin order over (subject, relation) groups so that
    every combination gets at least one representative before any gets a second.
    If df <= target_size, returns all rows.
    """
    if len(df) <= target_size:
        return df.copy()

    # Build a group key that captures both dimensions
    group_key = list(zip(df["subject"], df["relation"]))
    df = df.copy()
    df["_group"] = group_key

    group_indices = df.groupby("_group").indices
    groups = list(group_indices.keys())
    np.random.shuffle(groups)

    selected_indices = []
    pointers = {g: 0 for g in groups}

    while len(selected_indices) < target_size:
        added_in_round = 0
        for g in groups:
            idx_list = group_indices[g]
            if pointers[g] < len(idx_list):
                selected_indices.append(idx_list[pointers[g]])
                pointers[g] += 1
                added_in_round += 1
            if len(selected_indices) >= target_size:
                break
        if added_in_round == 0:
            break

    return df.iloc[selected_indices].drop(columns=["_group"]).copy()

def main():
    args = parse_args()

    # Load datasets
    positives = pd.read_csv(args.input_positive, sep="\t")
    negatives = pd.read_csv(args.input_negative, sep="\t")

    # Extract subject and relation for diversity logic
    for df in (positives, negatives):
        df["subject"]  = df["relation1"].apply(extract_subject)
        df["relation"] = df["relation1"].apply(extract_relation)

    # Drop rows where the relation is the legacy fallback
    positives = positives[positives["relation"] != "has_function"].reset_index(drop=True)
    negatives = negatives[negatives["relation"] != "has_function"].reset_index(drop=True)

    # 1. Diverse sampling for positives (subject × relation coverage)
    positives_balanced = get_diverse_sample(positives, args.size)
    target_count = len(positives_balanced)

    # 2. Stratified sampling for negatives, matching subject × relation distribution
    positive_counts = positives_balanced.groupby(["subject", "relation"]).size()
    sampled_negatives = []

    for (subject, relation), pos_count in positive_counts.items():
        neg_subset = negatives[
            (negatives["subject"] == subject) &
            (negatives["relation"] == relation)
        ]
        if len(neg_subset) == 0:
            continue
        take_n = min(len(neg_subset), pos_count)
        sampled_negatives.append(neg_subset.sample(n=take_n, random_state=42))

    if sampled_negatives:
        negatives_balanced = pd.concat(sampled_negatives)
    else:
        negatives_balanced = pd.DataFrame(columns=negatives.columns)

    # 3. Fill remaining negatives to reach 1:1 balance
    remaining_needed = target_count - len(negatives_balanced)
    if remaining_needed > 0:
        remaining_pool = negatives.drop(negatives_balanced.index, errors="ignore")
        if not remaining_pool.empty:
            extra = remaining_pool.sample(
                n=min(remaining_needed, len(remaining_pool)),
                random_state=42
            )
            negatives_balanced = pd.concat([negatives_balanced, extra])

    # 4. Final safety trim for negatives
    if len(negatives_balanced) > target_count:
        negatives_balanced = negatives_balanced.sample(n=target_count, random_state=42)

    # Add labels
    positives_balanced["label"] = 1
    negatives_balanced["label"] = 0

    # Combine and shuffle
    final_dataset = pd.concat([positives_balanced, negatives_balanced])
    final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Clean up helper columns
    final_dataset = final_dataset.drop(columns=["subject", "relation"], errors="ignore")

    # Save
    output_path = f"contradictions/{os.path.basename(args.input_positive).replace('.tsv', '_benchmark.tsv')}"
    final_dataset.to_csv(output_path, sep="\t", index=False)

    # ---- Coverage report ----
    pos_b = positives_balanced.copy()
    neg_b = negatives_balanced.copy()

    print(f"\nBenchmark dataset generated: {output_path}")
    print(f"Total size : {len(final_dataset)}  (Positives: {target_count}, Negatives: {len(neg_b)})")
    print(f"\nPositive set coverage:")
    print(f"  Unique subjects  : {pos_b['subject'].nunique()}")
    print(f"  Unique relations : {pos_b['relation'].nunique()} {sorted(pos_b['relation'].dropna().unique())}")
    print(f"\nNegative set coverage:")
    print(f"  Unique subjects  : {neg_b['subject'].nunique()}")
    print(f"  Unique relations : {neg_b['relation'].nunique()} {sorted(neg_b['relation'].dropna().unique())}")

if __name__ == "__main__":
    main()