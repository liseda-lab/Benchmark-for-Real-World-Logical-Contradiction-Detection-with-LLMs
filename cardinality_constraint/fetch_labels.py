import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import time
import urllib.error
import argparse
import os

# ── Wikidata QIDs for the two context concepts ────────────────────────────────
# Q21502838 = "property constraint"
# Q19474404 = "single-value constraint"
PROPERTY_CONSTRAINT_QID    = "P2302"
SINGLE_VALUE_CONSTRAINT_QID = "Q19474404"

LANGUAGE_CODES = {
    "english":    "en",
    "spanish":    "es",
    "chinese":    "zh",
    "japanese":   "ja",
    "french":     "fr",
    "german":     "de",
    "portuguese": "pt",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Wikidata labels in multiple languages for cardinality contradiction rows "
            "and produce one TSV per language with (relation1, relation2, context) triples."
        )
    )
    parser.add_argument("--input",       required=True, help="Path to input CSV (already filtered/downsampled).")
    parser.add_argument("--output-dir",  default=".",   help="Directory where per-language TSVs will be written (default: current dir).")
    parser.add_argument("--nature", default="contradiction", choices=["contradiction", "non_contradiction"], help="Whether the input examples are contradictions or non-contradictions (default: contradiction). Affects the context string in the output.")
    parser.add_argument("--source",      default=None,  help="Path to the full original labeled CSV (with <lang>_label boolean columns). Used to find replacements for rows with missing labels.")
    parser.add_argument(
        "--languages", nargs="+", required=True,
        help=(
            "Languages to generate outputs for. Use friendly names (english, spanish, …) "
            "or BCP-47 codes (en, es, …). Example: --languages english spanish chinese"
        )
    )
    parser.add_argument("--batch-size",  type=int, default=50, help="SPARQL batch size (default: 50).")
    parser.add_argument("--max-retries", type=int, default=5,  help="Max retries per batch on HTTP errors (default: 5).")
    parser.add_argument("--base-delay",  type=int, default=5,  help="Base delay in seconds for exponential backoff (default: 5).")
    return parser.parse_args()


# ── Auxiliar Functions ───────────────────────────────────────────────────────────────────

def resolve_language(lang: str) -> tuple[str, str]:
    """Return (friendly_name, bcp47_code) for a language spec."""
    lower = lang.lower()
    if lower in LANGUAGE_CODES:
        return lower, LANGUAGE_CODES[lower]
    reverse = {v: k for k, v in LANGUAGE_CODES.items()}
    return reverse.get(lower, lower), lower


def normalise_id(raw: str) -> str:
    """Strip a full Wikidata URI to its QID/PID."""
    return str(raw).strip().split("/")[-1]


def collect_entities(df: pd.DataFrame, columns: list[str]) -> set[str]:
    """Return all normalised QIDs/PIDs found across the given columns."""
    entities: set[str] = set()
    for col in columns:
        if col in df.columns:
            for val in df[col].dropna().unique():
                entities.add(normalise_id(str(val)))
    return entities


# ── SPARQL ────────────────────────────────────────────────────────────────────

def fetch_labels_for_language(
    qids: set[str],
    lang_code: str,
    batch_size: int,
    max_retries: int,
    base_delay: int,
) -> dict[str, str]:
    """
    Fetch labels in `lang_code` for all `qids` using wikibase:label with
    English as fallback, so we always receive a human-readable string.
    Returns { normalised_qid: label }.
    """
    endpoint = SPARQLWrapper("https://query.wikidata.org/sparql")
    endpoint.setReturnFormat(JSON)

    qid_list = sorted(qids)
    labels: dict[str, str] = {}
    total_batches = -(-len(qid_list) // batch_size)

    for i in range(0, len(qid_list), batch_size):
        batch     = qid_list[i : i + batch_size]
        batch_num = i // batch_size + 1
        values_str = " ".join(f"wd:{qid}" for qid in batch)

        # First pass: fetch label strictly in the target language (no fallback),
        # consistent with how check_labels.py marks the boolean columns.
        lang_filter = f'LANG(?label) = "{lang_code}"'
        query = f"""
        SELECT ?entity ?label WHERE {{
          VALUES ?entity {{ {values_str} }}
          ?entity rdfs:label ?label .
          FILTER({lang_filter})
        }}
        """

        for attempt in range(max_retries):
            try:
                endpoint.setQuery(query)
                response = endpoint.query().convert()
                for binding in response["results"]["bindings"]:
                    qid   = normalise_id(binding["entity"]["value"])
                    label = binding["label"]["value"]
                    labels[qid] = label
                time.sleep(0.5)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = base_delay * (2 ** attempt)
                    print(f"    429 — retrying in {wait}s (attempt {attempt + 1}/{max_retries})…")
                    time.sleep(wait)
                else:
                    print(f"    HTTP {e.code} on batch {batch_num}: {e}")
                    break
            except urllib.error.URLError as e:
                wait = base_delay * (2 ** attempt)
                print(f"    URL error on batch {batch_num}: {e} — retrying in {wait}s…")
                time.sleep(wait)
        else:
            print(f"    Giving up on batch {batch_num} after {max_retries} attempts.")

        print(
            f"    [{lang_code}] Batch {batch_num}/{total_batches} — "
            f"{min(i + batch_size, len(qid_list))}/{len(qid_list)} entities processed"
        )

    return labels


# ── Statement buider ─────────────────────────────────────────────────────────

def get_label(labels: dict[str, str], raw: str) -> str:
    """Resolve a raw URI or bare QID to its label, falling back to the QID."""
    return labels.get(normalise_id(raw), normalise_id(raw))


def is_bare_id(s: str) -> bool:
    """Return True if the string looks like an unresolved QID or PID."""
    s = s.strip()
    return (s.startswith("Q") or s.startswith("P")) and s[1:].isdigit()


def row_has_full_labels(row: pd.Series, labels: dict[str, str]) -> bool:
    """
    Return True only if every data entity in the row resolved to a real label
    (i.e. none of item / property / value1 / value2 is still a bare QID/PID).
    """
    for col in ("item", "property", "value1", "value2"):
        label = labels.get(normalise_id(str(row[col])), normalise_id(str(row[col])))
        if is_bare_id(label):
            return False
    return True


def build_statements(
    df: pd.DataFrame,
    labels: dict[str, str],
    item_col: str,
    prop_col: str,
    val1_col: str,
    val2_col: str,
    property_constraint_label: str,
    single_value_label: str,
    nature: str,
) -> pd.DataFrame:
    """
    For each row produce:
      relation1 : (item_label; prop_label; val1_label)
      relation2 : (item_label; prop_label; val2_label)
      context   : (prop_label; <property_constraint_label>; <single_value_label>)
    """
    rows = []
    for _, row in df.iterrows():
        label_item = get_label(labels, str(row[item_col]))
        label_prop = get_label(labels, str(row[prop_col]))
        label_val1 = get_label(labels, str(row[val1_col]))
        label_val2 = get_label(labels, str(row[val2_col]))

        if nature == "contradiction":
            rows.append({
                "relation1": f"({label_item}; {label_prop}; {label_val1})",
                "relation2": f"({label_item}; {label_prop}; {label_val2})",
                "context":   f"({label_prop}; {property_constraint_label}; {single_value_label})",
            })
        else:  # non_contradiction
            rows.append({
                "relation1": f"({label_item}; {label_prop}; {label_val1})",
                "relation2": f"({label_item}; {label_prop}; {label_val2})",
                "context":   f"({label_prop}; NOT {property_constraint_label}; {single_value_label})",
            })

    return pd.DataFrame(rows, columns=["relation1", "relation2", "context"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    lang_pairs: list[tuple[str, str]] = [resolve_language(l) for l in args.languages]
    print(f"Languages: { {name: code for name, code in lang_pairs} }")

    print(f"\nLoading input file: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} rows, {len(df.columns)} columns.")

    # Build a set of row fingerprints already in the working set so we never
    # pick a duplicate from the source pool.
    def fingerprint(row) -> tuple:
        return (str(row["item"]), str(row["property"]),
                str(row["value1"]), str(row["value2"]))

    working_fingerprints = set(df.apply(fingerprint, axis=1))

    source_df = None
    if args.source:
        print(f"Loading source pool: {args.source}")
        source_df = pd.read_csv(args.source)
        print(f"  {len(source_df)} rows in source pool.")
    else:
        print("  No --source provided — rows with missing labels will be dropped with a warning.")

    data_columns = ["item", "property", "value1", "value2"]
    os.makedirs(args.output_dir, exist_ok=True)

    for friendly_name, lang_code in lang_pairs:
        print(f"\n{'─' * 60}")
        print(f"Processing '{friendly_name}' ({lang_code})…")

        # Fetch labels for the working set
        entities = collect_entities(df, data_columns)
        entities.add(PROPERTY_CONSTRAINT_QID)
        entities.add(SINGLE_VALUE_CONSTRAINT_QID)
        print(f"  Fetching labels for {len(entities)} unique entities…")

        labels = fetch_labels_for_language(
            entities, lang_code,
            args.batch_size, args.max_retries, args.base_delay,
        )
        print(f"  Fetched {len(labels)} labels.")

        property_constraint_label = labels.get(PROPERTY_CONSTRAINT_QID,     "property constraint")
        single_value_label        = labels.get(SINGLE_VALUE_CONSTRAINT_QID, "single-value constraint")
        print(f"  'property constraint'     → {property_constraint_label!r}")
        print(f"  'single-value constraint' → {single_value_label!r}")

        def make_candidate_iter(friendly: str):
            if source_df is None:
                return iter([])
            lang_col = f"{friendly}_label"
            if lang_col not in source_df.columns:
                print(f"  Warning: '{lang_col}' column not found in source — no replacements available.")
                return iter([])
            pool = source_df[
                source_df[lang_col].astype(str).str.lower() == "true"
            ]
            pool = pool[~pool.apply(fingerprint, axis=1).isin(working_fingerprints)]
            return pool.iterrows()

        candidate_iter = make_candidate_iter(friendly_name)
        used_fingerprints = set(working_fingerprints)  # grows as we pick candidates

        valid_rows    = []
        invalid_count = 0
        unfilled      = 0

        for _, row in df.iterrows():
            if row_has_full_labels(row, labels):
                valid_rows.append(row)
            else:
                invalid_count += 1
                replaced = False

                while not replaced:
                    try:
                        _, candidate = next(candidate_iter)
                        fp = fingerprint(candidate)
                        if fp in used_fingerprints:
                            continue  # skip exact duplicates picked in this run

                        # Fetch labels for the candidate's new entities if needed
                        new_entities = {
                            normalise_id(str(candidate[c]))
                            for c in data_columns
                            if normalise_id(str(candidate[c])) not in labels
                        }
                        if new_entities:
                            new_labels = fetch_labels_for_language(
                                new_entities, lang_code,
                                args.batch_size, args.max_retries, args.base_delay,
                            )
                            labels.update(new_labels)

                        if row_has_full_labels(candidate, labels):
                            valid_rows.append(candidate)
                            used_fingerprints.add(fp)
                            replaced = True

                    except StopIteration:
                        print(f"  Warning: source pool exhausted — could not replace a row with missing labels.")
                        unfilled += 1
                        break

        if invalid_count:
            filled = invalid_count - unfilled
            print(f"  {invalid_count} incomplete row(s) found: {filled} replaced, {unfilled} could not be filled.")
        else:
            print(f"  All rows have full labels — no replacements needed.")

        final_df = pd.DataFrame(valid_rows).reset_index(drop=True)

        # Build and write statements
        statements = build_statements(
            final_df, labels,
            "item", "property", "value1", "value2",
            property_constraint_label, single_value_label,
            args.nature,
        )

        out_path = os.path.join(args.output_dir, f"cardinality_{args.nature}_{friendly_name}_dataset.tsv")
        statements.to_csv(out_path, index=False, sep="\t")
        print(f"  Written {len(statements)} rows → {out_path}")

    print(f"\n{len(lang_pairs)} file(s) written to '{args.output_dir}'.")


if __name__ == "__main__":
    main()