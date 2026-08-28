from rdflib import URIRef
from rdflib.namespace import RDF, OWL
from collections import defaultdict
import pandas as pd
import re

from owldataloader_thesis import OwlDataLoader

# ---------- Macros ----------



# ---------- GO Namespace → Relation Mapping ----------

GO_BRANCH_RELATION = {
    "BP": "has_biological_process",
    "MF": "enables",
    "CC": "located_in",
}


# ---------- Utilities ----------

def get_label(loader, uri: str) -> str:
    return loader.label_map.get(uri, uri.split("/")[-1])

def get_protein_label(uniprot_id: str, id_mapping: pd.DataFrame) -> str:
    # Look up the ID
    subset = id_mapping[id_mapping['From'].str.strip() == uniprot_id.strip()]
    if subset.empty:
        print(f"Warning: {uniprot_id} not found in mapping")
        return uniprot_id

    # Get the full string
    label = subset['Protein names'].iloc[0].strip()

    # Stop at the first " ("
    if " (" in label:
        label = label.split(" (")[0].strip()

    # Stop at the first " ["
    if " [" in label:
        label = label.split(" [")[0].strip()

    return label

def get_ontology(uri: URIRef):
    s = str(uri)
    if "GO_" in s:
        return "GO"
    if "HP_" in s:
        return "HPO"
    return None

def get_go_branch(loader: OwlDataLoader, uri: URIRef):
    return loader.go_branch_map.get(str(uri), None)

def get_relation_for_go_term(loader: OwlDataLoader, uri: URIRef) -> str:
    """
    Returns the appropriate OBO relation string for a GO term
    based on its namespace (BP, MF, CC).
    Falls back to 'has_function' if the term is not found in any branch.
    """
    branch = get_go_branch(loader, uri)
    return GO_BRANCH_RELATION.get(branch, "has_function")

def build_hierarchy_path(subclass_map, parent: URIRef, child: URIRef):
    path = []

    def dfs(current, target, visited):
        visited.add(current)
        if current == target:
            path.append(current)
            return True
        for c in subclass_map.get(current, []):
            if c not in visited and dfs(c, target, visited):
                path.append(current)
                return True
        return False

    dfs(parent, child, set())
    return " -> ".join(str(p) for p in reversed(path))


# ---------- Contradiction Detection ----------

def get_direct_contradictions(loader: OwlDataLoader, id_mapping: pd.DataFrame):
    pos_set = set(loader.pos_statement)
    direct_conflicts = set()
    rows = []

    for subj, obj in loader.neg_statement:
        if (subj, obj) in pos_set:
            direct_conflicts.add((subj, obj))

            obj_uri = URIRef(obj)
            relation = get_relation_for_go_term(loader, obj_uri)

            entity1 = get_protein_label(subj.split("/")[-1], id_mapping)
            entity2 = get_label(loader, obj)
            rows.append({
                "relation1": f"({entity1}; {relation}; {entity2})",
                "relation2": f"({entity1}; NOT {relation}; {entity2})"
            })

    return len(direct_conflicts), rows, direct_conflicts

def get_non_contradiction_pairs_go_branch(loader: OwlDataLoader, direct_conflict_set, id_mapping: pd.DataFrame):
    results = []
    count = 0

    pos_dict = defaultdict(set)
    neg_dict = defaultdict(set)

    # Index statements by protein
    for s, o in loader.pos_statement:
        pos_dict[s].add(o)

    for s, o in loader.neg_statement:
        neg_dict[s].add(o)

    for protein in pos_dict:

        if protein not in neg_dict:
            continue

        for pos_obj in pos_dict[protein]:
            for neg_obj in neg_dict[protein]:

                # Skip direct contradictions
                if (protein, pos_obj) in direct_conflict_set:
                    continue

                pos_uri = URIRef(pos_obj)
                neg_uri = URIRef(neg_obj)

                # Must both be GO terms
                if get_ontology(pos_uri) != "GO" or get_ontology(neg_uri) != "GO":
                    continue

                # Same GO branch (BP / MF / CC)
                if get_go_branch(loader, pos_uri) != get_go_branch(loader, neg_uri):
                    continue

                # Skip hierarchical contradictions
                if pos_uri in loader.subclass_map.get(neg_uri, set()):
                    continue
                if neg_uri in loader.subclass_map.get(pos_uri, set()):
                    continue

                count += 1

                pos_relation = get_relation_for_go_term(loader, pos_uri)
                neg_relation = get_relation_for_go_term(loader, neg_uri)

                entity = get_protein_label(protein.split("/")[-1], id_mapping)
                pos_term = get_label(loader, pos_obj)
                neg_term = get_label(loader, neg_obj)

                results.append({
                    "relation1": f"({entity}; {pos_relation}; {pos_term})",
                    "relation2": f"({entity}; NOT {neg_relation}; {neg_term})"
                })

    return count, results

def get_hierarchical_contradictions(loader: OwlDataLoader, direct_conflict_set, id_mapping: pd.DataFrame):
    results = []
    count = 0

    for neg_subj, neg_obj in loader.neg_statement:
        for pos_subj, pos_obj in loader.pos_statement:

            # Same entity
            if neg_subj != pos_subj:
                continue

            # Skip direct contradictions
            if (neg_subj, pos_obj) in direct_conflict_set:
                continue

            neg_uri = URIRef(neg_obj)
            pos_uri = URIRef(pos_obj)

            # Same ontology
            if get_ontology(neg_uri) != get_ontology(pos_uri):
                continue

            # Hierarchical contradiction: pos ⊑ neg
            if pos_uri in loader.subclass_map.get(neg_uri, set()):
                count += 1
                path = build_hierarchy_path(loader.subclass_map, pos_uri, neg_uri)

                pos_relation = get_relation_for_go_term(loader, pos_uri)
                neg_relation = get_relation_for_go_term(loader, neg_uri)

                entity1 = get_protein_label(neg_subj.split("/")[-1], id_mapping)
                pos_term = get_label(loader, pos_obj)
                neg_term = get_label(loader, neg_obj)
                results.append({
                    "relation1": f"({entity1}; NOT {neg_relation}; {neg_term})",
                    "relation2": f"({entity1}; {pos_relation}; {pos_term})",
                    "context": f"({pos_term}; subClassOf; {neg_term})",
                    "Hierarchy_Path": " -> ".join(
                        get_label(loader, p) for p in path.split(" -> ")
                    )
                })

    return count, results

def get_hierarchical_neg(loader, direct_conflict_set, id_mapping):
    results = []
    count = 0

    for neg_subj, neg_obj in loader.neg_statement:
        for pos_subj, pos_obj in loader.pos_statement:

            # Same entity
            if neg_subj != pos_subj:
                continue

            # Skip direct contradictions
            if (neg_subj, pos_obj) in direct_conflict_set:
                continue

            # Child (neg) → Parent (pos)
            neg_uri = URIRef(neg_obj)
            pos_uri = URIRef(pos_obj)

            # Same ontology
            if get_ontology(neg_uri) != get_ontology(pos_uri):
                continue

            # Hierarchical positive statements: neg ⊑ pos
            if neg_uri in loader.subclass_map.get(pos_uri, set()):
                count += 1
                path = build_hierarchy_path(loader.subclass_map, neg_uri, pos_uri)

                pos_relation = get_relation_for_go_term(loader, pos_uri)
                neg_relation = get_relation_for_go_term(loader, neg_uri)

                entity1 = get_protein_label(neg_subj.split("/")[-1], id_mapping)
                pos_term = get_label(loader, pos_obj)
                neg_term = get_label(loader, neg_obj)
                results.append({
                    "relation1": f"({entity1}; NOT {neg_relation}; {neg_term})",
                    "relation2": f"({entity1}; {pos_relation}; {pos_term})",
                    "context": f"({neg_term}; subClassOf; {pos_term})",
                    "Hierarchy_Path": " -> ".join(
                        get_label(loader, p) for p in path.split(" -> ")
                    )
                })

    return count, results

# ---------- Main ----------

def main():
    owl_loader = OwlDataLoader(
        "./ontologies",
        "config_schema.tsv",
        "config_data.tsv"
    )
    id_mapping = pd.read_csv("idmapping_2026_03_09.tsv", sep="\t")

    d_count, d_rows, d_set = get_direct_contradictions(owl_loader, id_mapping)
    pd.DataFrame(d_rows).to_csv("./contradictions/direct_contradictions.tsv", sep="\t", index=False)
    print("Direct contradictions identified and saved")

    h_count, h_rows = get_hierarchical_contradictions(owl_loader, d_set, id_mapping)
    pd.DataFrame(h_rows).to_csv("./contradictions/hierarchical_contradictions.tsv", sep="\t", index=False)
    print("Hierarchical contradictions identified and saved")

    nc_count, nc_rows = get_non_contradiction_pairs_go_branch(owl_loader, d_set, id_mapping)
    pd.DataFrame(nc_rows).to_csv("./contradictions/non_direct_contradictions.tsv", sep="\t", index=False)
    print("Non Direct contradiction pairs identified and saved")

    h_neg_count, h_neg_rows = get_hierarchical_neg(owl_loader, d_set, id_mapping)
    pd.DataFrame(h_neg_rows).to_csv("./contradictions/non_hierarchical_contradictions.tsv", sep="\t", index=False)
    print("Non Hierarchical contradictory statements identified and saved")

    g = owl_loader.graph
    print(f"Triples: {len(g)}")
    print(f"Classes: {len(set(g.subjects(RDF.type, OWL.Class)))}")
    print(f"Individuals: {len(set(g.subjects(RDF.type, OWL.NamedIndividual)))}")
    print(f"Direct contradictions: {d_count}")
    print(f"Direct non contradictions: {nc_count}")
    print(f"Hierarchical contradictions: {h_count}")
    print(f"Hierarchical non contradictions: {h_neg_count}")


if __name__ == "__main__":
    main()