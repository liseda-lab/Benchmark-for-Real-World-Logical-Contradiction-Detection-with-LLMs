import os
import time
import pandas as pd
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib.error

# ---------- CONFIGURATION ----------
INPUT_FILE = "disjoint_pairs.csv"
SUMMARY_FILE = "disjoint_pairs_summary.csv"
OUTPUT_FILE = "./contradictions/disjoint_contradictions.csv"
BATCH_SIZE = 20  # how many disjoint pairs to check at once
RETRY_LIMIT = 5
BASE_BACKOFF = 5  # seconds
# -----------------------------------

# Load disjoint pairs
disjoint_df = pd.read_csv(INPUT_FILE)

# Load existing contradictions if resuming
if os.path.exists(OUTPUT_FILE):
    contradictions_df = pd.read_csv(OUTPUT_FILE)
    seen = set(
        (row["disjoint_entity1"], row["disjoint_entity2"], row["violating_subclass"])
        for _, row in contradictions_df.iterrows()
    )
    contradictions = contradictions_df.to_dict(orient="records")
    print(f"Resuming with {len(contradictions)} existing contradiction results.")
else:
    contradictions = []
    seen = set()

# Load existing summary if resuming
if os.path.exists(SUMMARY_FILE):
    summary_df = pd.read_csv(SUMMARY_FILE)
    pair_counts = {(row["entity1"], row["entity2"]): row["contradiction_count"] 
                   for _, row in summary_df.iterrows()}
    summary_list = summary_df.to_dict(orient="records")
    processed_pairs = set((row["entity1"], row["entity2"]) for _, row in summary_df.iterrows())
    print(f"Resuming with {len(summary_list)} existing summary results.")
else:
    summary_list = []
    pair_counts = {}
    processed_pairs = set()

# Setup SPARQL
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)

# Helper: run query with exponential backoff
def query_with_retry(query):
    sparql.setQuery(query)
    for attempt in range(RETRY_LIMIT):
        try:
            return sparql.query().convert()
        except urllib.error.HTTPError as e:
            wait = BASE_BACKOFF * (2 ** attempt)
            if e.code == 429:
                print(f"429 Too Many Requests. Retrying in {wait}s...")
                time.sleep(wait)
            elif e.code in [500, 502, 503, 504]:
                print(f"Server error {e.code}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded.")

# Helper: create SPARQL VALUES batch
def make_values_block(pairs):
    block = " ".join(f"( <{e1}> <{e2}> )" for e1, e2 in pairs)
    return f"VALUES (?e1 ?e2) {{ {block} }}"

# Main loop
all_pairs = list(
    {(row["entity1"], row["entity2"]) for _, row in disjoint_df.iterrows()}
)

# Filter out already processed pairs
remaining_pairs = [pair for pair in all_pairs if pair not in processed_pairs]
print(f"Total pairs: {len(all_pairs)}, Remaining: {len(remaining_pairs)}")

batches = [remaining_pairs[i:i + BATCH_SIZE] for i in range(0, len(remaining_pairs), BATCH_SIZE)]

for batch in tqdm(batches, desc="Processing disjoint batches"):
    values_block = make_values_block(batch)
    query = f"""
    SELECT DISTINCT ?e1 ?e2 ?subclass WHERE {{
      {values_block}
      ?subclass wdt:P279+ ?e1 .
      ?subclass wdt:P279+ ?e2 .
    }}
    """

    try:
        results = query_with_retry(query)
        
        # 🎯 HERE: Count results per pair in this batch
        batch_pair_counts = {}
        
        # Initialize all pairs in this batch with 0 count
        for e1, e2 in batch:
            batch_pair_counts[(e1, e2)] = 0
        
        # Count actual results
        for result in results["results"]["bindings"]:
            e1 = result["e1"]["value"]
            e2 = result["e2"]["value"]
            subclass = result["subclass"]["value"]

            # Increment count for this pair
            pair_key = (e1, e2)
            batch_pair_counts[pair_key] += 1

            # Store individual contradiction
            key = (e1, e2, subclass)
            if key not in seen:
                contradictions.append({
                    "disjoint_entity1": e1,
                    "disjoint_entity2": e2,
                    "violating_subclass": subclass
                })
                seen.add(key)
        
        # 🎯 HERE: Add batch results to summary
        for (e1, e2), count in batch_pair_counts.items():
            summary_list.append({
                "entity1": e1,
                "entity2": e2,
                "contradiction_count": count
            })
            processed_pairs.add((e1, e2))
        
        # Print batch statistics
        total_batch_results = sum(batch_pair_counts.values())
        pairs_with_results = len([c for c in batch_pair_counts.values() if c > 0])
        print(f"Batch: {len(batch)} pairs, {total_batch_results} total results, "
              f"{pairs_with_results} pairs with contradictions")
        
    except Exception as e:
        print(f"Error processing batch starting with {batch[0]}: {e}")
        continue

    # Save progress for both files
    if contradictions:
        pd.DataFrame(contradictions).to_csv(OUTPUT_FILE, index=False)
    
    if summary_list:
        pd.DataFrame(summary_list).to_csv(SUMMARY_FILE, index=False)

print("Done. Files saved:")
print(f"Contradictions: {OUTPUT_FILE}")
print(f"Summary: {SUMMARY_FILE}")
print(f"Total contradictions: {len(contradictions)}")

# Print final statistics
if summary_list:
    summary_df = pd.DataFrame(summary_list)
    print(f"\nFinal Statistics:")
    print(f"   Total pairs processed: {len(summary_df)}")
    print(f"   Pairs with contradictions: {len(summary_df[summary_df['contradiction_count'] > 0])}")
    print(f"   Pairs without contradictions: {len(summary_df[summary_df['contradiction_count'] == 0])}")
    print(f"   Total contradictions found: {summary_df['contradiction_count'].sum()}")
    print(f"   Max contradictions per pair: {summary_df['contradiction_count'].max()}")
    print(f"   Average contradictions per pair: {summary_df['contradiction_count'].mean():.2f}")