from SPARQLWrapper import SPARQLWrapper, JSON

# SPARQL endpoint
endpoint_url = "https://query.wikidata.org/sparql"

# SPARQL query to fetch properties with a single-value constraint
query = """
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX wd: <http://www.wikidata.org/entity/>

SELECT DISTINCT ?property ?propertyLabel WHERE {
  ?property a wikibase:Property ;
            p:P2302 [ ps:P2302 wd:Q19474404 ] .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
ORDER BY ?property
"""

def fetch_single_value_properties():
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    props = []
    for item in results["results"]["bindings"]:
        prop_uri = item["property"]["value"]        # e.g. "http://www.wikidata.org/entity/P569"
        prop_id = prop_uri.rsplit('/', 1)[-1]       # e.g. "P569"
        label = item["propertyLabel"]["value"]      # human-readable name
        props.append(f"{prop_id}\t{label}")
    return props

def save_to_txt(props, filename="single_value_properties.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for line in props:
            f.write(line + "\n")

if __name__ == "__main__":
    props = fetch_single_value_properties()
    save_to_txt(props)
    print(f"Saved {len(props)} properties to single_value_properties.txt")




