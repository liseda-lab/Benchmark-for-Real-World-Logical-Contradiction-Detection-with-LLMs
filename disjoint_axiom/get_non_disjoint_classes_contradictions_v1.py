import os
import time
import pandas as pd
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib.error

# ---------- CONFIGURATION ----------
DISJOINT_FILE = "disjoint_pairs_summary.csv"
POSITIVE_FILE = "./contradictions/disjoint_contradictions.csv"
OUTPUT_FILE = "./contradictions/non_contradictions_explicit_v1.csv"

NEGATIVES_PER_SIDE = 30000
RETRY_LIMIT = 5
BASE_BACKOFF = 5
DELAY_BETWEEN_QUERIES = 1
# -----------------------------------

# Load disjoint pairs
disjoint_df = pd.read_csv(DISJOINT_FILE)
disjoint_pairs = list(
    {(row["entity1"], row["entity2"]) for _, row in disjoint_df.iterrows()}
)

# Load positives to avoid accidental overlap
positive_df = pd.read_csv(POSITIVE_FILE)
positive_subclasses = set(positive_df["violating_subclass"].tolist())

# Setup SPARQL endpoint
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)

def query_with_retry(query):
    sparql.setQuery(query)
    for attempt in range(RETRY_LIMIT):
        try:
            return sparql.query().convert()
        except urllib.error.HTTPError as e:
            wait = BASE_BACKOFF * (2 ** attempt)
            if e.code in [429, 500, 502, 503, 504]:
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded.")

negatives = []

for e1, e2 in tqdm(disjoint_pairs, desc="Processing disjoint pairs"):

    # ==========================================================
    # SIDE 1: subclass under e1 but NOT under e2
    # ==========================================================
    query = f"""
    SELECT DISTINCT ?subclass WHERE {{
      ?subclass wdt:P279+ <{e1}> .
      FILTER NOT EXISTS {{
        ?subclass wdt:P279+ <{e2}> .
      }}
    }}
    LIMIT {NEGATIVES_PER_SIDE}
    """

    try:
        results = query_with_retry(query)

        for result in results["results"]["bindings"]:
            subclass = result["subclass"]["value"]

            # Avoid overlap with positive contradictions
            if subclass in positive_subclasses:
                continue

            negatives.append({
                "superclass": e1,
                "disjoint_other": e2,
                "subclass": subclass
            })

    except Exception as e:
        print(f"Error processing pair {e1}, {e2} (side 1): {e}")

    time.sleep(DELAY_BETWEEN_QUERIES)

    # ==========================================================
    # SIDE 2: subclass under e2 but NOT under e1
    # ==========================================================
    query = f"""
    SELECT DISTINCT ?subclass WHERE {{
      ?subclass wdt:P279+ <{e2}> .
      FILTER NOT EXISTS {{
        ?subclass wdt:P279+ <{e1}> .
      }}
    }}
    LIMIT {NEGATIVES_PER_SIDE}
    """

    try:
        results = query_with_retry(query)

        for result in results["results"]["bindings"]:
            subclass = result["subclass"]["value"]

            if subclass in positive_subclasses:
                continue

            negatives.append({
                "superclass": e2,
                "disjoint_other": e1,
                "subclass": subclass
            })

    except Exception as e:
        print(f"Error processing pair {e1}, {e2} (side 2): {e}")

    time.sleep(DELAY_BETWEEN_QUERIES)

# Save output
if negatives:
    df_out = pd.DataFrame(negatives)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df_out)} non-contradictions to {OUTPUT_FILE}")
else:
    print("No negatives found.")
