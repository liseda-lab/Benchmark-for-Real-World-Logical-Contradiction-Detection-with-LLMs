from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import csv
import os
import time

# Configuration
CONTRADICTION_FILE = "./contradictions/downsampled_cardinality_contradictions.csv"
MULTIVALUE_PROPERTY_FILE = "filtered_multivalue_properties.txt"
OUTPUT_FILE = "./contradictions/non_cardinality_contradictions.csv"
WAIT_SECONDS = 0.5
MAX_RESULTS_PER_SUBJECT = 10


def load_positive_subjects(file_path):
    """Extract the unique set of subject IDs from the positive contradiction file."""
    print(f"Loading positive subjects from {file_path}...")
    df = pd.read_csv(file_path)

    if "item" not in df.columns:
        raise ValueError("Contradiction file must have an 'item' column.")

    # Items are stored as full URIs or QIDs — normalise to QID
    subjects = df["item"].dropna().unique().tolist()
    subjects = [s.split("/")[-1] if s.startswith("http") else s for s in subjects]
    subjects = [s for s in subjects if s.startswith("Q")]

    print(f"Found {len(subjects)} unique subjects in positive examples.")
    return subjects


def load_multivalue_properties(file_path):
    """Load the list of known multi-value properties."""
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["property_id", "property_label"]
    )
    return df["property_id"].tolist()


def load_already_processed(output_file):
    """Return set of subject QIDs already fully queried."""
    if not os.path.exists(output_file):
        return set()
    try:
        df = pd.read_csv(output_file)
        return set(df["item"].unique().tolist())
    except Exception as e:
        print(f"Could not load existing output file: {e}")
        return set()


def fetch_non_contradictions_for_subject(subject_qid, properties):
    """
    For a single subject, query all multi-value properties in one SPARQL call
    using a VALUES block on the property side. Returns up to
    MAX_RESULTS_PER_SUBJECT rows in total across all properties.
    """
    values_block = " ".join(f"wdt:{p}" for p in properties)

    query = f"""
    SELECT DISTINCT ?prop ?object1 ?object2 WHERE {{
      VALUES ?prop {{ {values_block} }}
      wd:{subject_qid} ?prop ?object1, ?object2.
      FILTER(?object1 != ?object2)
    }}
    LIMIT {MAX_RESULTS_PER_SUBJECT}
    """

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setMethod("POST")
    sparql.addCustomHttpHeader("User-Agent", "NonContradictionFetcher/1.0")

    try:
        results = sparql.query().convert()
        rows = []
        for result in results["results"]["bindings"]:
            # prop is returned as a full URI e.g. http://www.wikidata.org/prop/direct/P106
            prop_uri = result["prop"]["value"]
            prop_id = "P" + prop_uri.split("/P")[-1]
            obj1 = result["object1"]["value"]
            obj2 = result["object2"]["value"]
            rows.append({
                "property": prop_id,
                "item": subject_qid,
                "value1": obj1,
                "value2": obj2,
            })
        return rows

    except Exception as e:
        print(f"Error querying {subject_qid}: {e}")
        return []


def collect_non_contradictions(subjects, properties, output_file):
    already_processed = load_already_processed(output_file)

    file_exists = os.path.exists(output_file)
    write_header = not file_exists or os.stat(output_file).st_size == 0

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["property", "item", "value1", "value2"])

        total_saved = 0

        for idx, subject_qid in enumerate(subjects, 1):
            if subject_qid in already_processed:
                continue

            print(f"[{idx}/{len(subjects)}] Querying {subject_qid}...")

            rows = fetch_non_contradictions_for_subject(subject_qid, properties)

            for row in rows:
                writer.writerow([row["property"], row["item"], row["value1"], row["value2"]])

            if rows:
                total_saved += len(rows)
                f.flush()

            already_processed.add(subject_qid)
            print(f"{len(rows)} rows saved (total so far: {total_saved})")

            time.sleep(WAIT_SECONDS)

    print(f"\nFinished process. Total non-contradictions saved: {total_saved} → {output_file}")


def main():
    subjects = load_positive_subjects(CONTRADICTION_FILE)
    properties = load_multivalue_properties(MULTIVALUE_PROPERTY_FILE)
    collect_non_contradictions(subjects, properties, OUTPUT_FILE)


if __name__ == "__main__":
    main()