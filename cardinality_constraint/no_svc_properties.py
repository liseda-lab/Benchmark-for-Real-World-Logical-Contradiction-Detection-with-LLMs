from SPARQLWrapper import SPARQLWrapper, JSON
import time

def fetch_multi_value_properties(limit=5000, output_file="multivalue_properties.txt"):
    print("Fetching properties without single-value constraint...")

    query = f"""
    SELECT DISTINCT ?prop ?propLabel WHERE {{
      ?prop a wikibase:Property.

      FILTER NOT EXISTS {{
        ?prop p:propertyConstraint [
          pq:constraintQ wd:Q19474404
        ]
      }}

      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en".
        ?prop rdfs:label ?propLabel.
      }}
    }}
    LIMIT {limit}
    """

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(query)
    sparql.setMethod("POST")
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", "MultiValuePropBot/1.0")

    results = sparql.query().convert()

    props = []
    for result in results["results"]["bindings"]:
        prop_uri = result["prop"]["value"]
        prop_label = result["propLabel"]["value"]
        prop_id = prop_uri.split("/")[-1]
        props.append((prop_id, prop_label))

    print(f"Retrieved {len(props)} properties without single-value constraint.")

    with open(output_file, "w", encoding="utf-8") as f:
        for prop_id, label in props:
            f.write(f"{prop_id}\t{label}\n")

    print(f"Results written to: {output_file}")

# Run the script
if __name__ == "__main__":
    fetch_multi_value_properties()



