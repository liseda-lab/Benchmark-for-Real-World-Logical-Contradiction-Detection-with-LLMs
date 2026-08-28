import pandas as pd
import csv
import time
import os
from SPARQLWrapper import SPARQLWrapper, JSON

# Configuration
VIOLATION_FILE = "violation_results.csv"
CONTRADICTION_FILE = "./contradictions/cardinality_contradictions.csv"
WAIT_SECONDS = 10
MAX_CONTRADICTIONS_PER_PROPERTY = 5

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

def analyze_violations(file_path):
    try:
        df = pd.read_csv(file_path)
        if "property" not in df.columns or "violation_count" not in df.columns:
            raise ValueError("CSV file is missing required columns.")

        df["violation_count"] = pd.to_numeric(df["violation_count"], errors="coerce").fillna(0).astype(int)
        properties_with_violations = df[df["violation_count"] > 0]
        print(f"Properties with violations: {len(properties_with_violations)}")
        print(f"Total number of violations: {properties_with_violations['violation_count'].sum()}")
        return properties_with_violations["property"].tolist()

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def load_existing_contradictions(file_path):
    """Returns a set of property IDs that have already been processed."""
    if not os.path.exists(file_path):
        return set()
    try:
        df = pd.read_csv(file_path)
        return set(df["property"].unique().tolist())
    except Exception as e:
        print(f"Warning: Could not load existing contradictions file. {e}")
        return set()

def get_example_contradictions(prop_id, limit=3):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query = f"""
    SELECT DISTINCT ?item ?value1 ?value2 WHERE {{
      ?item wdt:{prop_id} ?value1, ?value2.
      FILTER(?value1 != ?value2)
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        contradictions = []
        for row in bindings:
            contradictions.append({
                "property": prop_id,
                "item": row["item"]["value"],
                "value1": row["value1"]["value"],
                "value2": row["value2"]["value"]
            })
        return contradictions
    except Exception as e:
        print(f"Error querying property {prop_id}: {e}")
        return []

def collect_contradiction_examples(violating_props, output_file):
    already_processed = load_existing_contradictions(output_file)

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if os.stat(output_file).st_size == 0:
            writer.writerow(["property", "item", "value1", "value2"])  # write header if empty

        for prop in violating_props:
            if prop in already_processed:
                print(f"Skipping {prop}, already recorded.")
                continue

            print(f"Querying up to {MAX_CONTRADICTIONS_PER_PROPERTY} contradictions for {prop}...")
            contradictions = get_example_contradictions(prop, MAX_CONTRADICTIONS_PER_PROPERTY)

            if contradictions:
                for c in contradictions:
                    writer.writerow([c["property"], c["item"], c["value1"], c["value2"]])
                print(f"Saved {len(contradictions)} contradictions for {prop}")
            else:
                print(f"No contradiction found for {prop}")

            time.sleep(WAIT_SECONDS)

def main():
    violating_props = analyze_violations(VIOLATION_FILE)
    if violating_props:
        collect_contradiction_examples(violating_props, CONTRADICTION_FILE)

if __name__ == "__main__":
    main()

