import pandas as pd
import re
import argparse

# Regex to extract QID anywhere in the string
qid_pattern = re.compile(r'\bQ\d+\b')

def filter_qids(df, columns):
    """Keep only rows where specified columns contain a QID."""
    if not columns:
        return df

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    mask = pd.Series(True, index=df.index)

    for col in columns:
        mask &= df[col].apply(lambda x: bool(qid_pattern.search(str(x))))

    return df[mask]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter contradictions to keep only valid QIDs.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--columns", nargs="+", required=True, help="Columns to check")

    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    df_clean = filter_qids(df, columns=args.columns)
    df_clean.drop_duplicates(inplace=True)
    df_clean.to_csv(args.output, index=False)

    print(f"Filtered file saved to {args.output}. {len(df_clean)} rows remain.")