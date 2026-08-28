import os
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

# Output file path
output_file = "disjoint_pairs.csv"

# Initialize SPARQL
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)

# SPARQL query and entire pipeline adapted from the Disjointness Violations in Wikidata paper
query = """
SELECT DISTINCT ?class ?e1 ?e2 WHERE {
  ?class p:P2738 ?l .
  MINUS { ?l wikibase:rank wikibase:DeprecatedRank . }
  ?l pq:P11260 ?e1 .
  ?l pq:P11260 ?e2 .
  FILTER ( str(?e1) < str(?e2) )
}
ORDER BY ?class
"""

# Run query
sparql.setQuery(query)

try:
    print("Running SPARQL query...")
    results = sparql.query().convert()

    # Load existing data if available
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        existing_records = set(tuple(row) for row in existing_df[["class", "entity1", "entity2"]].values)
        print(f"Resuming from {len(existing_records)} previously saved records.")
    else:
        existing_df = pd.DataFrame(columns=["class", "entity1", "entity2"])
        existing_records = set()

    # Store new results
    new_data = []
    count = 0
    save_interval = 100

    for result in results["results"]["bindings"]:
        cls = result["class"]["value"]
        e1 = result["e1"]["value"]
        e2 = result["e2"]["value"]
        key = (cls, e1, e2)

        if key not in existing_records:
            new_data.append({"class": cls, "entity1": e1, "entity2": e2})
            existing_records.add(key)
            count += 1

        # Save progress every N items
        if count > 0 and count % save_interval == 0:
            print(f"Saving progress after {count} new records...")
            temp_df = pd.DataFrame(new_data)
            combined_df = pd.concat([existing_df, temp_df], ignore_index=True)
            combined_df.to_csv(output_file, index=False)
            existing_df = combined_df
            new_data = []

    # Final save
    if new_data:
        print(f"Saving final batch of {len(new_data)} new records...")
        temp_df = pd.DataFrame(new_data)
        final_df = pd.concat([existing_df, temp_df], ignore_index=True)
        final_df.to_csv(output_file, index=False)

    print(f"Done. Total records: {len(existing_records)}")
except Exception as e:
    print("An error occurred:", e)
