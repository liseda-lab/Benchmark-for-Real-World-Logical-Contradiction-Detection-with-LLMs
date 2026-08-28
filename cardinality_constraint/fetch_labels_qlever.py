import pandas as pd
import requests
import time
import random
import argparse
import os

# ── Wikidata QIDs for the two context concepts ────────────────────────────────
PROPERTY_CONSTRAINT_QID     = "P2302"
SINGLE_VALUE_CONSTRAINT_QID = "Q19474404"

QLEVER_ENDPOINT = "https://qlever.dev/api/wikidata"
QLEVER_HEADERS  = {
    "Accept":     "application/sparql-results+json",
    "User-Agent": "WDContradictionThesis/1.0 (thesis-bot@example.com)",
}

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
            "Fetch Wikidata labels via QLever in multiple languages for cardinality "
            "contradiction rows and produce one TSV per language with "
            "(relation1, relation2, context) triples."
        )
    )
    parser.add_argument("--input",       required=True, help="Path to input CSV (already filtered/downsampled).")
    parser.add_argument("--output-dir",  default=".",   help="Directory where per-language TSVs will be written (default: current dir).")
    parser.add_argument("--source",      default=None,  help="Path to the full original labeled CSV. Used to find replacements for rows with missing labels.")
    parser.add_argument(
        "--languages", nargs="+", required=True,
        help="Languages to generate outputs for. Example: --languages english spanish chinese"
    )
    parser.add_argument("--nature", choices=["positive", "negative"], default="positive", help="Whether the input rows are 'positive' (cardinality contradictions) or 'negative' (non-contradictions). Affects how the context column is formulated.")
    parser.add_argument("--batch-size",  type=int, default=100, help="Batch size for QLever queries (default: 100, QLever handles larger batches well).")
    parser.add_argument("--max-retries", type=int, default=5,   help="Max retries per batch on errors (default: 5).")
    parser.add_argument("--base-delay",  type=int, default=2,   help="Base delay in seconds for exponential backoff (default: 2).")
    return parser.parse_args()


# ── Auxiliar Functions ───────────────────────────────────────────────────────────────────

def resolve_language(lang: str) -> tuple[str, str]:
    lower = lang.lower()
    if lower in LANGUAGE_CODES:
        return lower, LANGUAGE_CODES[lower]
    reverse = {v: k for k, v in LANGUAGE_CODES.items()}
    return reverse.get(lower, lower), lower


def normalise_id(raw: str) -> str:
    return str(raw).strip().split("/")[-1]


def collect_entities(df: pd.DataFrame, columns: list[str]) -> set[str]:
    entities: set[str] = set()
    for col in columns:
        if col in df.columns:
            for val in df[col].dropna().unique():
                entities.add(normalise_id(str(val)))
    return entities


def is_bare_id(s: str) -> bool:
    s = s.strip()
    return (s.startswith("Q") or s.startswith("P")) and s[1:].isdigit()


def get_label(labels: dict[str, str], raw: str) -> str:
    return labels.get(normalise_id(raw), normalise_id(raw))


def row_has_full_labels(row: pd.Series, labels: dict[str, str]) -> bool:
    for col in ("item", "property", "value1", "value2"):
        label = labels.get(normalise_id(str(row[col])), normalise_id(str(row[col])))
        if is_bare_id(label):
            return False
    return True


# ── QLever SPARQL ─────────────────────────────────────────────────────────────

def fetch_labels_for_language(
    qids: set[str],
    lang_code: str,
    batch_size: int,
    max_retries: int,
    base_delay: int,
) -> dict[str, str]:
    """
    Fetch rdfs:label in `lang_code` for all `qids` via QLever.
    Returns { normalised_qid: label }.
    """
    qid_list      = sorted(qids)
    labels        = {}
    total_batches = -(-len(qid_list) // batch_size)

    for i in range(0, len(qid_list), batch_size):
        batch      = qid_list[i : i + batch_size]
        batch_num  = i // batch_size + 1
        values_str = " ".join(
            f"<http://www.wikidata.org/entity/{qid}>" for qid in batch
        )

        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?entity ?label WHERE {{
          VALUES ?entity {{ {values_str} }}
          ?entity rdfs:label ?label .
          FILTER(LANG(?label) = "{lang_code}")
        }}
        """

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    QLEVER_ENDPOINT,
                    data={"query": query},
                    headers=QLEVER_HEADERS,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                for binding in data.get("results", {}).get("bindings", []):
                    qid   = normalise_id(binding["entity"]["value"])
                    label = binding["label"]["value"]
                    labels[qid] = label
                time.sleep(0.3 + random.uniform(0, 0.3))
                break
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                if status == 429:
                    wait = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"    429 — retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})…")
                    time.sleep(wait)
                else:
                    wait = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"    HTTP {status} on batch {batch_num} — retrying in {wait:.1f}s…")
                    time.sleep(wait)
            except requests.RequestException as e:
                wait = base_delay * (2 ** attempt) + random.uniform(0, 2)
                print(f"    Request error on batch {batch_num}: {e} — retrying in {wait:.1f}s…")
                time.sleep(wait)
        else:
            print(f"    Giving up on batch {batch_num} after {max_retries} attempts.")

        print(
            f"    [{lang_code}] Batch {batch_num}/{total_batches} — "
            f"{min(i + batch_size, len(qid_list))}/{len(qid_list)} entities processed"
        )

    return labels


# ── Statement builder ─────────────────────────────────────────────────────────

def build_statements(
    df: pd.DataFrame,
    labels: dict[str, str],
    property_constraint_label: str,
    single_value_label: str,
    nature: str,
) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        label_item = get_label(labels, str(row["item"]))
        label_prop = get_label(labels, str(row["property"]))
        label_val1 = get_label(labels, str(row["value1"]))
        label_val2 = get_label(labels, str(row["value2"]))

        if nature == "positive":
            rows.append({
                "relation1": f"({label_item}; {label_prop}; {label_val1})",
                "relation2": f"({label_item}; {label_prop}; {label_val2})",
                "context":   f"({label_prop}; {property_constraint_label}; {single_value_label})",
            })
        else:
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
            pool = source_df[source_df[lang_col].astype(str).str.lower() == "true"]
            pool = pool[~pool.apply(fingerprint, axis=1).isin(working_fingerprints)]
            return pool.iterrows()

        candidate_iter    = make_candidate_iter(friendly_name)
        used_fingerprints = set(working_fingerprints)
        valid_rows        = []
        invalid_count     = 0
        unfilled          = 0

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
                            continue

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
                        print("  Warning: source pool exhausted — could not replace a row with missing labels.")
                        unfilled += 1
                        break

        if invalid_count:
            filled = invalid_count - unfilled
            print(f"  {invalid_count} incomplete row(s) found: {filled} replaced, {unfilled} could not be filled.")
        else:
            print("  All rows have full labels — no replacements needed.")

        final_df   = pd.DataFrame(valid_rows).reset_index(drop=True)
        statements = build_statements(final_df, labels, property_constraint_label, single_value_label, args.nature)

        nat_text = "contradiction" if args.nature == "positive" else "non_contradiction"
        out_path = os.path.join(args.output_dir, f"cardinality_{nat_text}_{friendly_name}.tsv")
        statements.to_csv(out_path, index=False, sep="\t")
        print(f"  Written {len(statements)} rows → {out_path}")

    print(f"\n{len(lang_pairs)} file(s) written to '{args.output_dir}'.")


if __name__ == "__main__":
    main()