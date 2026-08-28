import time
import csv
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

# Settings
INPUT_FILE = "filtered_svc_properties.txt"
OUTPUT_FILE = "violation_results.csv"
WAIT_SECONDS = 10  # Time between SPARQL queries

# Wikidata SPARQL endpoint
endpoint_url = "https://query.wikidata.org/sparql"

def get_properties_from_file(file_path):
    """Read property IDs from the filtered .txt file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.split('\t')[0].strip() for line in f if line.strip()]

def load_existing_results(output_file):
    """Load previously completed property IDs from the CSV output."""
    try:
        df = pd.read_csv(output_file)
        return set(df["property"].tolist())
    except FileNotFoundError:
        return set()

def save_result(prop_id, count):
    """Append the result for a property to the CSV file."""
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([prop_id, count])

def query_property_violation_count(prop_id):
    """Run a SPARQL query to count how many items have >1 value for the given property."""
    sparql = SPARQLWrapper(endpoint_url)
    query = f"""
    SELECT (COUNT(*) AS ?violationCount) WHERE {{
      {{
        SELECT ?item (COUNT(?value) AS ?valueCount) WHERE {{
          ?item wdt:{prop_id} ?value.
        }}
        GROUP BY ?item
        HAVING (COUNT(?value) > 1)
      }}
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        if bindings:
            count_str = bindings[0]["violationCount"]["value"]
            return int(count_str)
        else:
            return 0
    except Exception as e:
        print(f"Error querying {prop_id}: {e}")
        return None

def main():
    all_properties = get_properties_from_file(INPUT_FILE)
    already_checked = load_existing_results(OUTPUT_FILE)

    # Initialize the output file with a header if it's not already created
    if not already_checked:
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["property", "violation_count"])

    for prop_id in all_properties:
        if prop_id in already_checked:
            print(f"Skipping already processed: {prop_id}")
            continue

        print(f"Checking {prop_id}...")
        count = query_property_violation_count(prop_id)

        if count is not None:
            save_result(prop_id, count)
            print(f"Saved: {prop_id}, violations: {count}")
        else:
            print(f"Failed to get result for {prop_id}")

        time.sleep(WAIT_SECONDS)

if __name__ == "__main__":
    main()
