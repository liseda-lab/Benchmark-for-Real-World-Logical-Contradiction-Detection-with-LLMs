import argparse
import csv
import os
import pandas as pd
from deterministic_verbalizer import TripleVerbalizer

# Must match every language name that can appear at the end of a filename.
# Keys are the lowercase language name as it appears in the filename;
# values are BCP-47 codes matching the verbalizer's TEMPLATES keys.
LANGUAGE_CODES = {
    "english":    "en",
    "spanish":    "es",
    "french":     "fr"
}


def detect_language(filename: str) -> str:
    """
    Scan every '_'-separated token in the filename (without extension) and
    return the BCP-47 code for the first token that matches a known language
    name.  Falls back to 'en' with a warning if nothing matches.

    Works for any naming pattern, e.g.:
      dataset_spanish.tsv               -> 'es'
      cardinality_contradiction_french.tsv -> 'fr'
      explicit_dataset_portuguese.tsv   -> 'pt'
    """
    stem   = os.path.splitext(os.path.basename(filename))[0]   # strip extension
    tokens = [t.lower().strip() for t in stem.split("_")]

    for token in tokens:
        if token in LANGUAGE_CODES:
            return token, LANGUAGE_CODES[token]

    return None, "en"


def process_tsv(input_path: str, output_name: str, output_dir: str):

    verbalizer = TripleVerbalizer()

    df = pd.read_csv(input_path, sep="\t", dtype=str).fillna("")

    required_columns = {"relation1", "relation2", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    candidate_lang, lang_code = detect_language(input_path)
    if candidate_lang:
        print(f"Detected language: '{candidate_lang}' -> '{lang_code}'")
    else:
        print(f"Warning: no known language found in filename '{os.path.basename(input_path)}' — defaulting to English.")

    def verbalize_col(col_name: str) -> list:
        return [
            verbalizer.verbalize(row[col_name], language=lang_code)
            for _, row in df.iterrows()
        ]

    df["relation1"] = verbalize_col("relation1")
    df["relation2"] = verbalize_col("relation2")

    if "context" in df.columns:
        df["context"] = verbalize_col("context")

    os.makedirs(output_dir, exist_ok=True)

    # ── Explicit dataset: relation1 + relation2 + context + label ─────────────
    explicit_cols = ["relation1", "relation2"]
    if "context" in df.columns:
        explicit_cols.append("context")
    explicit_cols.append("label")

    explicit_df = df[explicit_cols]
    explicit_path = os.path.join(output_dir, f"explicit_{output_name}")
    explicit_df.to_csv(explicit_path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    print(f"Explicit dataset written to: {explicit_path}")

    # ── Implicit dataset: relation1 + relation2 + label (no context) ──────────
    if "context" in df.columns:
        implicit_cols = ["relation1", "relation2", "label"]
        implicit_df   = df[implicit_cols]
        implicit_path = os.path.join(output_dir, f"implicit_{output_name}")
        implicit_df.to_csv(implicit_path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
        print(f"Implicit dataset written to: {implicit_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Deterministic TSV Triple Verbalizer")
    parser.add_argument("--input",      required=True, help="Input TSV file.")
    parser.add_argument("--output",     required=True, help="Output filename (will be prefixed with explicit_/implicit_).")
    parser.add_argument("--output-dir", default="verbalized_datasets", help="Directory for output files (default: verbalized_datasets).")

    args = parser.parse_args()

    process_tsv(args.input, args.output, args.output_dir)