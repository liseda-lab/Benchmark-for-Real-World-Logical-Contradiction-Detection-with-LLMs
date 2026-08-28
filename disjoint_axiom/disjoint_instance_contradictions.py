import csv
import time
from pathlib import Path

import pandas as pd
import requests

ENDPOINT = "https://qlever.dev/api/wikidata"
HEADERS = {"Accept": "application/sparql-results+json"}

RATE_LIMIT_SLEEP = 0.1
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds (exponential backoff multiplier)
OUTPUT_CSV = "contradictions_instances/instance_disjoint_contradictions.csv"
DISJOINT_PAIRS_CSV = "disjoint_pairs.csv"


def load_disjoint_pairs(csv_path: str) -> pd.DataFrame:
    """Load disjoint class pairs into a DataFrame from disjoint_pairs.csv."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find disjoint pairs file: {csv_path}")

    df = pd.read_csv(path)
    required_cols = {"entity1", "entity2"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s) in {csv_path}: {', '.join(sorted(missing))}"
        )

    # Keep only the pair columns used by this script and drop empty rows.
    df = df[["entity1", "entity2"]].rename(
        columns={"entity1": "disjoint_entity1", "entity2": "disjoint_entity2"}
    )
    df = df.dropna(subset=["disjoint_entity1", "disjoint_entity2"])
    df["disjoint_entity1"] = df["disjoint_entity1"].astype(str).str.strip()
    df["disjoint_entity2"] = df["disjoint_entity2"].astype(str).str.strip()
    df = df[(df["disjoint_entity1"] != "") & (df["disjoint_entity2"] != "")]

    return df


def main() -> None:
    disjoint_classes_df = load_disjoint_pairs(DISJOINT_PAIRS_CSV)
    all_results = []

    for idx, row in disjoint_classes_df.iterrows():
        c = row["disjoint_entity1"]
        d = row["disjoint_entity2"]

        query = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        SELECT DISTINCT ?instance WHERE {{
          ?instance wdt:P31/wdt:P279+ <{c}> .
          ?instance wdt:P31/wdt:P279+ <{d}> .
        }}
        LIMIT 50000
        """

        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                resp = requests.post(
                    ENDPOINT,
                    data={"query": query},
                    headers=HEADERS,
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()
                bindings = data.get("results", {}).get("bindings", [])

                total = len(disjoint_classes_df)
                step = idx + 1
                if bindings:
                    print(
                        f"[{step}/{total}] Found {len(bindings)} contradictions for pair ({c}, {d})",
                        flush=True,
                    )
                    for item in bindings:
                        result_entry = {
                            "disjoint_entity1": c,
                            "disjoint_entity2": d,
                            "violating_instance": item["instance"]["value"],
                        }
                        all_results.append(result_entry)
                        print(f"   -> Instance: {result_entry['violating_instance']}", flush=True)
                else:
                    print(f"[{step}/{total}] No contradictions for pair.", flush=True)

                break
            except Exception as exc:
                attempt += 1
                print(
                    f"[{idx + 1}] Request error (attempt {attempt}/{MAX_RETRIES}): {exc}",
                    flush=True,
                )
                time.sleep(RETRY_BACKOFF**attempt)

        if RATE_LIMIT_SLEEP:
            time.sleep(RATE_LIMIT_SLEEP)

    print("\n--- Final Report ---")
    print(f"Total instances found violating disjointness: {len(all_results)}")

    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["violating_instance", "disjoint_entity1", "disjoint_entity2"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved contradictions to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
