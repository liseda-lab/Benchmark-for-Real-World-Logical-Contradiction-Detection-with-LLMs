import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import time
import random
import urllib.error
import argparse
import os
import json

# ── Wikidata QIDs for the two context concepts ────────────────────────────────
SUBCLASS_OF_QID  = "P279"
INSTANCE_OF_QID  = "Q21503252"

LANGUAGE_CODES = {
    "english":    "en",
    "spanish":    "es",
    "chinese":    "zh",
    "japanese":   "ja",
    "french":     "fr",
    "german":     "de",
    "portuguese": "pt",
}


DATA_COLUMNS = ["disjoint_entity1", "disjoint_entity2", "violating_subclass"]

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Wikidata labels in multiple languages for disjoint contradiction rows "
            "and produce one TSV per language with (relation1, relation2, context) triples."
        )
    )
    parser.add_argument("--input",       required=True, help="Path to input CSV (already filtered/downsampled).")
    parser.add_argument("--output-dir",  default="./contradictions", help="Directory where per-language TSVs will be written.")
    parser.add_argument("--source",      default=None,  help="Path to the full original labeled CSV. Used to find replacement rows when labels are missing.")
    parser.add_argument(
        "--languages", nargs="+", required=True,
        help="Languages to generate outputs for. Example: --languages english spanish chinese"
    )
    parser.add_argument("--disjointeness-type", default="class", choices=["class", "instance"],
                        help="Whether contradictions are about classes or instances (default: class).")
    parser.add_argument("--nature-type", default="positive", choices=["positive", "negative"],
                        help="Whether examples are positive (violating) or negative (default: positive).")
    parser.add_argument("--batch-size",  type=int, default=50)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--base-delay",  type=int, default=5)
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def fingerprint(row) -> tuple:
    return tuple(str(row[c]) for c in DATA_COLUMNS)


def is_bare_id(s: str) -> bool:
    s = s.strip()
    return (s.startswith("Q") or s.startswith("P")) and s[1:].isdigit()


def get_label(labels: dict[str, str], raw: str) -> str:
    return labels.get(normalise_id(raw), normalise_id(raw))


def row_has_full_labels(row: pd.Series, labels: dict[str, str]) -> bool:
    for col in DATA_COLUMNS:
        label = labels.get(normalise_id(str(row[col])), normalise_id(str(row[col])))
        if is_bare_id(label):
            return False
    return True


# ── SPARQL ────────────────────────────────────────────────────────────────────

def fetch_labels_for_language(
    qids: set[str],
    lang_code: str,
    batch_size: int,
    max_retries: int,
    base_delay: int,
) -> dict[str, str]:
    endpoint = SPARQLWrapper("https://query.wikidata.org/sparql")
    endpoint.setReturnFormat(JSON)
    endpoint.addCustomHttpHeader(
        "User-Agent",
        "WDContradictionThesis/1.0 (thesis-bot@example.com)"
    )

    qid_list      = sorted(qids)
    labels        = {}
    total_batches = -(-len(qid_list) // batch_size)

    for i in range(0, len(qid_list), batch_size):
        batch      = qid_list[i : i + batch_size]
        batch_num  = i // batch_size + 1
        values_str = " ".join(f"wd:{qid}" for qid in batch)

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
                time.sleep(1.5 + random.uniform(0, 1))
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"    429 — retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})…")
                    time.sleep(wait)
                else:
                    print(f"    HTTP {e.code} on batch {batch_num}: {e}")
                    break
            except urllib.error.URLError as e:
                wait = base_delay * (2 ** attempt) + random.uniform(0, 2)
                print(f"    URL error on batch {batch_num}: {e} — retrying in {wait:.1f}s…")
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
    relation_label: str,
    disjoint_with_label: str,
    nature_type: str,
) -> pd.DataFrame:
    """
    For each row produce:
      relation1 : (subj_label; relation_label; disj1_label)
      relation2 : (subj_label; relation_label; disj2_label)
      context   : (disj1_label; disjoint_with_label; disj2_label)

    For negative examples the context relation is prefixed with NOT.
    """
    context_relation = disjoint_with_label if nature_type == "positive" else f"NOT {disjoint_with_label}"

    rows = []
    for _, row in df.iterrows():
        label_subj  = get_label(labels, str(row["violating_subclass"]))
        label_disj1 = get_label(labels, str(row["disjoint_entity1"]))
        label_disj2 = get_label(labels, str(row["disjoint_entity2"]))

        rows.append({
            "relation1": f"({label_subj}; {relation_label}; {label_disj1})",
            "relation2": f"({label_subj}; {relation_label}; {label_disj2})",
            "context":   f"({label_disj1}; {context_relation}; {label_disj2})",
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

    if args.disjointeness_type == "instance":
        DATA_COLUMNS[2] = "violating_instance"
    
    working_fingerprints = set(df.apply(fingerprint, axis=1))

    source_df = None
    if args.source:
        print(f"Loading source pool: {args.source}")
        source_df = pd.read_csv(args.source)
        print(f"  {len(source_df)} rows in source pool.")
    else:
        print("  No --source provided — rows with missing labels will be dropped with a warning.")

    concept_qid = SUBCLASS_OF_QID if args.disjointeness_type == "class" else INSTANCE_OF_QID
    
    os.makedirs(args.output_dir, exist_ok=True)

    for friendly_name, lang_code in lang_pairs:
        print(f"\n{'─' * 60}")
        print(f"Processing '{friendly_name}' ({lang_code})…")

        # ── Step 1: fetch labels for the working set ─────────────────────────
        entities = collect_entities(df, DATA_COLUMNS)
        entities.add(concept_qid)
        print(f"  Fetching labels for {len(entities)} unique entities…")

        labels = fetch_labels_for_language(
            entities, lang_code,
            args.batch_size, args.max_retries, args.base_delay,
        )
        print(f"  Fetched {len(labels)} labels.")

        relation_label      = labels.get(concept_qid, "subclass of" if args.disjointeness_type == "class" else "instance of")
        disjoint_with_label = json.load(open("disjoint_with_label.json"))[lang_code]["disjoint_with_label"]
        print(f"  'relation label'      → {relation_label!r}")
        print(f"  'disjoint with label' → {disjoint_with_label!r}")

        # ── Step 2: validate rows, swap in source candidates for bad ones ─────
        def make_candidate_iter(friendly: str):
            if source_df is None:
                return iter([])
            lang_col = f"{friendly}_label"
            if lang_col not in source_df.columns:
                print(f"  Warning: '{lang_col}' not found in source — no replacements available.")
                return iter([])
            pool = source_df[source_df[lang_col].astype(str).str.lower() == "true"]
            pool = pool[~pool.apply(fingerprint, axis=1).isin(working_fingerprints)]
            return pool.iterrows()

        candidate_iter = make_candidate_iter(friendly_name)
        used_fingerprints = set(working_fingerprints)

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
                            continue

                        # Fetch labels for any new entities in the candidate
                        new_entities = {
                            normalise_id(str(candidate[c]))
                            for c in DATA_COLUMNS
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

        # ── Step 3: build and write statements ───────────────────────────────
        statements = build_statements(
            final_df, labels,
            relation_label, disjoint_with_label,
            args.nature_type,
        )

        out_path = os.path.join(args.output_dir, f"disjoint_{args.nature_type}_{friendly_name}.tsv")
        statements.to_csv(out_path, index=False, sep="\t")
        print(f"  Written {len(statements)} rows → {out_path}")

    print(f"\nDone! {len(lang_pairs)} file(s) written to '{args.output_dir}'.")


if __name__ == "__main__":
    main()