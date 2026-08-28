import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import time
import urllib.error
import argparse

LANGUAGE_CODES = {
    "english": "en",
    "spanish": "es",
    "chinese": "zh",
    "french": "fr",
    "german": "de",
    "portuguese": "pt",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Check Wikidata label availability per language for QID columns in a CSV."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument(
        "--columns", nargs="+", required=True,
        help="Column names in the CSV that contain QIDs (e.g. --columns item property value1 value2)."
    )
    parser.add_argument(
        "--languages", nargs="+", required=True,
        help=(
            "Languages to check. Use either language names (english, spanish, chinese) "
            "or BCP-47 codes (en, es, zh). "
            "Example: --languages english spanish chinese"
        )
    )
    return parser.parse_args()


def resolve_language_code(lang: str) -> tuple[str, str]:
    """
    Returns (column_suffix, language_code).
    Accepts either a friendly name ('english') or a BCP-47 code ('en').
    """
    lower = lang.lower()
    if lower in LANGUAGE_CODES:
        return lower, LANGUAGE_CODES[lower]
    # Assume it's already a BCP-47 code; use it directly as suffix too
    return lower, lower


def collect_qids(df: pd.DataFrame, columns: list[str]) -> set:
    """Gather all unique QIDs from the specified columns."""
    qids = set()
    for col in columns:
        if col not in df.columns:
            print(f"Warning: column '{col}' not found in input file — skipping.")
            continue
        for val in df[col].dropna().unique():
            val = str(val).strip()
            if val.startswith("Q") or val.startswith("P") or "wikidata.org" in val:
                qids.add(val)
    return qids


def normalise_qid(raw: str) -> str:
    """Strip a full URI down to its QID/PID."""
    return raw.split("/")[-1]


def fetch_labels_for_languages(
    qids: set,
    language_codes: list[str],
    batch_size: int = 50,
    max_retries: int = 5,
    base_delay: int = 5,
) -> dict[str, dict[str, bool]]:
    """
    Returns a dict:
        { normalised_qid: { lang_code: True/False, ... }, ... }

    True  → the entity has a proper label (not just the QID echoed back) in that language.
    False → no label found.
    """
    endpoint = SPARQLWrapper("https://query.wikidata.org/sparql")
    endpoint.setReturnFormat(JSON)

    qid_list = [normalise_qid(q) for q in qids]

    # Initialise result: all False
    results_map: dict[str, dict[str, bool]] = {
        qid: {lang: False for lang in language_codes} for qid in qid_list
    }

    for i in range(0, len(qid_list), batch_size):
        batch = qid_list[i : i + batch_size]
        values_str = " ".join(f"wd:{qid}" for qid in batch)

        # Build a SPARQL query that requests labels in all target languages at once.
        # We use rdfs:label with FILTER to avoid wikibase:label auto-fallback behaviour,
        # so we only get a True when the label actually exists in that language.
        # Blazegraph does not support plain string literals inside FILTER(...IN(...)),
        # so we build an explicit OR chain using LANG() = "xx" comparisons instead.
        lang_filter = " || ".join(f'LANG(?label) = "{lc}"' for lc in language_codes)
        query = f"""
        SELECT ?entity ?label ?lang WHERE {{
          VALUES ?entity {{ {values_str} }}
          ?entity rdfs:label ?label .
          BIND(LANG(?label) AS ?lang)
          FILTER({lang_filter})
        }}
        """

        for attempt in range(max_retries):
            try:
                endpoint.setQuery(query)
                response = endpoint.query().convert()

                for binding in response["results"]["bindings"]:
                    uri  = binding["entity"]["value"]
                    lang = binding["lang"]["value"]
                    qid  = normalise_qid(uri)
                    if qid in results_map and lang in results_map[qid]:
                        results_map[qid][lang] = True

                time.sleep(0.5)
                break

            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = base_delay * (2 ** attempt)
                    print(f"  429 Too Many Requests — retrying in {wait}s (attempt {attempt + 1}/{max_retries})…")
                    time.sleep(wait)
                else:
                    print(f"  HTTP error {e.code} on batch {i//batch_size + 1}: {e}")
                    break

            except urllib.error.URLError as e:
                wait = base_delay * (2 ** attempt)
                print(f"  URL error on batch {i//batch_size + 1}: {e} — retrying in {wait}s…")
                time.sleep(wait)

            except Exception as e:
                wait = base_delay * (2 ** attempt)
                print(f"  Unexpected error on batch {i//batch_size + 1}: {e} — retrying in {wait}s…")
                time.sleep(wait)

        else:
            print(f"  Failed to fetch batch {i//batch_size + 1} (starting at {batch[0]}) after {max_retries} attempts — skipping.")

        print(
            f"  Processed batch {i//batch_size + 1} / {-(-len(qid_list)//batch_size)}"
            f" ({min(i + batch_size, len(qid_list))}/{len(qid_list)} QIDs)"
        )

    return results_map


def row_has_label(row, columns: list[str], label_map: dict, lang_code: str) -> bool:
    """
    Returns True only if *every* QID present in the specified columns of this row
    has a label in the given language.
    """
    for col in columns:
        raw = row.get(col)
        if pd.isna(raw):
            continue
        qid = normalise_qid(str(raw).strip())
        if not label_map.get(qid, {}).get(lang_code, False):
            return False
    return True


def main():
    args = parse_args()

    # Resolve language names → (suffix, code) pairs
    lang_pairs: list[tuple[str, str]] = [resolve_language_code(l) for l in args.languages]
    lang_codes  = [code   for _, code   in lang_pairs]
    lang_labels = [suffix for suffix, _ in lang_pairs]

    print(f"Loading input file: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} rows, {len(df.columns)} columns.")

    print(f"Collecting QIDs from columns: {args.columns}")
    qids = collect_qids(df, args.columns)
    print(f"  {len(qids)} unique QIDs found.")

    print(f"Fetching labels for languages: {lang_codes} …")
    label_map = fetch_labels_for_languages(qids, lang_codes)
    print("   Done fetching labels.")

    # Add one boolean column per language
    for suffix, code in zip(lang_labels, lang_codes):
        col_name = f"{suffix}_label"
        df[col_name] = df.apply(
            lambda row: row_has_label(row, args.columns, label_map, code),
            axis=1,
        )
        true_count = df[col_name].sum()
        print(f"  '{col_name}': {true_count}/{len(df)} rows have all labels in '{code}'.")

    print(f"Writing output to: {args.output}")
    df.to_csv(args.output, index=False)
    print("Done!")


if __name__ == "__main__":
    main()