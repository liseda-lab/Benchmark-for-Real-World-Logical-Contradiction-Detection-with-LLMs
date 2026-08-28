import pandas as pd
import argparse
import os
import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Assemble per-language disjoint contradiction datasets by joining one positive "
            "file and two negative files per language. Negative files are expected to be "
            "half the size of the positive so the final dataset matches the target size. "
            "Produces one TSV per language and one combined TSV."
        )
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing positive and negative TSV files."
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory where assembled TSVs will be written (default: current dir)."
    )
    parser.add_argument(
        "--languages", nargs="+", default=None,
        help=(
            "Languages to assemble. If omitted, all languages found via the positive "
            "prefix are auto-discovered. Example: --languages english spanish chinese"
        )
    )
    parser.add_argument(
        "--positive-prefix", default="disjoint_contradiction_",
        help="Filename prefix for positive example files (default: disjoint_contradiction_)."
    )
    parser.add_argument(
        "--negative-v1-prefix", default="disjoint_non_contradiction_v1_",
        help="Filename prefix for class-based negative example files (default: disjoint_non_contradiction_v1_)."
    )
    parser.add_argument(
        "--negative-v2-prefix", default="disjoint_non_contradiction_v2_",
        help="Filename prefix for instance-based negative example files (default: disjoint_non_contradiction_v2_)."
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle rows within each assembled file (recommended to avoid label ordering bias)."
    )
    return parser.parse_args()


def discover_languages(input_dir: str, prefix: str) -> list[str]:
    """Infer available languages from filenames matching the given prefix."""
    pattern = os.path.join(input_dir, f"{prefix}*.tsv")
    files = glob.glob(pattern)
    languages = []
    for f in sorted(files):
        basename = os.path.basename(f)
        lang = basename[len(prefix):].replace(".tsv", "")
        languages.append(lang)
    return languages


def load_tsv(path: str, label: int) -> pd.DataFrame:
    """Load a TSV file and attach a label column."""
    df = pd.read_csv(path, sep="\t")
    df["label"] = label
    return df


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.languages:
        languages = args.languages
    else:
        languages = discover_languages(args.input_dir, args.positive_prefix)
        if not languages:
            print(
                f"No files matching '{args.positive_prefix}*.tsv' found in '{args.input_dir}'. "
                "Use --languages to specify them explicitly."
            )
            return
        print(f"Discovered languages: {languages}")

    all_frames = []
    assembled  = []

    for lang in languages:
        pos_path      = os.path.join(args.input_dir, f"{args.positive_prefix}{lang}.tsv")
        neg_v1_path    = os.path.join(args.input_dir, f"{args.negative_v1_prefix}{lang}.tsv")
        neg_v2_path = os.path.join(args.input_dir, f"{args.negative_v2_prefix}{lang}.tsv")

        missing = [p for p in (pos_path, neg_v1_path, neg_v2_path) if not os.path.exists(p)]
        if missing:
            print(f"  [{lang}] Skipping — file(s) not found: {missing}")
            continue

        print(f"  [{lang}] Loading positive examples:         {pos_path}")
        pos_df = load_tsv(pos_path, label=1)
        print(f"    {len(pos_df)} positive rows.")

        print(f"  [{lang}] Loading negative v1 examples:    {neg_v1_path}")
        neg_v1_df = load_tsv(neg_v1_path, label=0)
        print(f"    {len(neg_v1_df)} negative v1 rows.")

        print(f"  [{lang}] Loading negative v2 examples: {neg_v2_path}")
        neg_v2_df = load_tsv(neg_v2_path, label=0)
        print(f"    {len(neg_v2_df)} negative v2 rows.")

        total_neg = len(neg_v1_df) + len(neg_v2_df)
        total     = len(pos_df) + total_neg
        print(f"  [{lang}] Total: {len(pos_df)} positive + {total_neg} negative = {total} rows.")

        lang_df = pd.concat([pos_df, neg_v1_df, neg_v2_df], ignore_index=True)

        if args.shuffle:
            lang_df = lang_df.sample(frac=1, random_state=42).reset_index(drop=True)

        disj_type = args.positive_prefix
        out_path = os.path.join(args.output_dir, f"{disj_type}benchmark_{lang}.tsv")
        lang_df.to_csv(out_path, index=False, sep="\t")
        print(f"  [{lang}] Written {len(lang_df)} rows → {out_path}")

        all_frames.append(lang_df)
        assembled.append(lang)

    if not all_frames:
        print("No language datasets were assembled. Exiting.")
        return

    combined_df = pd.concat(all_frames, ignore_index=True)
    if args.shuffle:
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    
    combined_path = os.path.join(args.output_dir, f"{disj_type}benchmark_all.tsv")
    combined_df.to_csv(combined_path, index=False, sep="\t")

    print(f"\nSummary")
    print(f"  Languages assembled : {assembled}")
    print(f"  Total rows          : {len(combined_df)}")
    print(f"  Positive (label=1)  : {(combined_df['label'] == 1).sum()}")
    print(f"  Negative (label=0)  : {(combined_df['label'] == 0).sum()}")
    print(f"  Combined file       : {combined_path}")


if __name__ == "__main__":
    main()