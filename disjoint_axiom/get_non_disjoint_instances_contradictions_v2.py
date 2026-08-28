import pandas as pd
import requests
import time
import random
import os
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


# ── Setup helpers (same pattern as script 2) ──────────────────────────────────

def create_robust_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "Accept": "application/sparql-results+json",
        "User-Agent": "InstanceSuperclassBot/1.0 (research@example.com)"
    })
    return session


def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                data = json.load(f)
            tqdm.write(f"📁 Loaded checkpoint from {checkpoint_file}")
            tqdm.write(f"✅ Previously processed: {data['processed_count']} items")
            tqdm.write(f"📊 Results so far: {len(data['results'])}")
            return data
        except Exception as e:
            tqdm.write(f"❌ Error loading checkpoint: {e}")
            return None
    return None


def save_checkpoint(checkpoint_file, processed_count, results, skipped, processed_rows, superclass_cache):
    checkpoint_data = {
        "processed_count": processed_count,
        "results": results,
        "skipped": skipped,
        "processed_rows": processed_rows,
        "superclass_cache": superclass_cache,
    }
    try:
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        tqdm.write(f"💾 Checkpoint saved: {processed_count} items processed")
    except Exception as e:
        tqdm.write(f"❌ Error saving checkpoint: {e}")


def load_instance_contradictions(csv_path):
    df = pd.read_csv(csv_path)
    return df[["disjoint_entity1", "disjoint_entity2", "violating_instance"]].dropna()


def load_disjoint_class_map(csv_path):
    """Optional guard file: entity1,entity2,class — known disjoint-class assertions."""
    df = pd.read_csv(csv_path)
    class_map = {}
    for _, row in df.iterrows():
        e1, e2, cls = row["entity1"], row["entity2"], row["class"]
        class_map[(e1, e2)] = cls
        class_map[(e2, e1)] = cls
    return class_map


def get_single_subclassof_uri(entity_uri, session, max_retries=3):
    """
    Get ONE superclass (P279) of entity_uri.
    NOTE: this is called on the CLASS (entity1/entity2), never on the
    instance — individuals don't carry P279 edges. This is the key
    translation from script 2 (which called it on violating_subclass,
    since there X was itself a class).
    """
    qid = entity_uri.split("/")[-1]
    query = f"""
    SELECT ?superclass WHERE {{
      wd:{qid} wdt:P279 ?superclass .
    }}
    LIMIT 1
    """
    url = "https://query.wikidata.org/sparql"

    for attempt in range(max_retries):
        try:
            response = session.get(url, params={"query": query}, timeout=(10, 30))
            if response.status_code == 200:
                bindings = response.json()["results"]["bindings"]
                if bindings:
                    return bindings[0]["superclass"]["value"]
                tqdm.write(f"⚠️  No superclass found for {qid}")
                return None
            else:
                tqdm.write(f"🌐 HTTP {response.status_code} for {qid}")
                if response.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                return None
        except requests.exceptions.ConnectionError:
            tqdm.write(f"🔌 Connection error for {qid} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + random.uniform(0, 1))
            else:
                tqdm.write(f"❌ Failed to fetch superclass for {qid} after {max_retries} attempts")
                return None
        except requests.exceptions.Timeout:
            tqdm.write(f"⏰ Timeout for {qid} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
        except Exception as e:
            tqdm.write(f"💥 Unexpected error for {qid}: {e}")
            return None
    return None


# ── Core logic ─────────────────────────────────────────────────────────────

def generate_valid_instance_pairs(contradictions_csv, disjoint_class_csv, output_csv,
                                   checkpoint_interval=10):
    tqdm.write("🚀 Starting to generate valid instance-level pairs...")

    checkpoint_file = "instance_processing_checkpoint.json"
    checkpoint_data = load_checkpoint(checkpoint_file)

    tqdm.write("📂 Loading data...")
    contradiction_rows = load_instance_contradictions(contradictions_csv)
    disjoint_class_map = (
        load_disjoint_class_map(disjoint_class_csv) if disjoint_class_csv else {}
    )
    tqdm.write(f"📋 Loaded {len(contradiction_rows)} contradiction rows")
    tqdm.write(f"🗂️  Loaded {len(disjoint_class_map)} disjoint class mappings")

    if checkpoint_data:
        results = checkpoint_data["results"]
        skipped = checkpoint_data["skipped"]
        processed_count = checkpoint_data["processed_count"]
        processed_rows = set(checkpoint_data["processed_rows"])
        superclass_cache = checkpoint_data.get("superclass_cache", {})
        tqdm.write(f"🔄 Resuming from checkpoint: {processed_count} items already processed")
    else:
        results = []
        skipped = []
        processed_count = 0
        processed_rows = set()
        superclass_cache = {}
        tqdm.write("🆕 Starting fresh processing...")

    session = create_robust_session()
    total_items = len(contradiction_rows)
    remaining_items = total_items - processed_count

    try:
        with tqdm(total=remaining_items, desc="Processing rows", unit="row",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

            for _, row in contradiction_rows.iterrows():
                entity1 = row["disjoint_entity1"]
                entity2 = row["disjoint_entity2"]
                instance = row["violating_instance"]

                row_key = f"{entity1}_{entity2}_{instance}"
                if row_key in processed_rows:
                    continue

                # Pick which of the two disjoint classes to broaden.
                # The instance must be a P31 member of that entity for the
                # transitive "instance is also an S" claim to hold, but since
                # the input row already tells us `instance` was flagged
                # against this pair, we broaden whichever entity we choose
                # and keep the OTHER as the (unchanged) pairing partner —
                # this keeps the two choices coupled, unlike script 2's
                # decoupled random.choice bug.
                source_entity, kept_entity = random.choice(
                    [(entity1, entity2), (entity2, entity1)]
                )

                qid = instance.split("/")[-1]
                pbar.set_description(f"Processing {qid}")

                if source_entity in superclass_cache:
                    superclass = superclass_cache[source_entity]
                else:
                    superclass = get_single_subclassof_uri(source_entity, session)
                    superclass_cache[source_entity] = superclass
                    time.sleep(1.0)  # polite delay only on an actual network call

                if not superclass:
                    skipped.append(instance)
                    processed_rows.add(row_key)
                    processed_count += 1
                    pbar.update(1)
                    if processed_count % checkpoint_interval == 0:
                        save_checkpoint(checkpoint_file, processed_count, results,
                                         skipped, list(processed_rows), superclass_cache)
                    continue

                # Safety guard: don't assert non-disjointness if the fetched
                # superclass IS the known disjoint class for this pair
                # (mirrors script 2's collision check).
                known_disjoint_class = disjoint_class_map.get((entity1, entity2))
                if known_disjoint_class and superclass == known_disjoint_class:
                    tqdm.write(f"⏭️  Skipping {qid}: superclass matches known disjoint class")
                    skipped.append(instance)
                    processed_rows.add(row_key)
                    processed_count += 1
                    pbar.update(1)
                    if processed_count % checkpoint_interval == 0:
                        save_checkpoint(checkpoint_file, processed_count, results,
                                         skipped, list(processed_rows), superclass_cache)
                    continue

                # Avoid a trivial/degenerate output where the broadened
                # superclass is identical to the entity we're pairing it with.
                if superclass == kept_entity:
                    skipped.append(instance)
                    processed_rows.add(row_key)
                    processed_count += 1
                    pbar.update(1)
                    continue

                results.append({
                    "disjoint_entity1": superclass,
                    "disjoint_entity2": kept_entity,
                    "violating_instance": instance
                })

                processed_rows.add(row_key)
                processed_count += 1
                pbar.update(1)
                pbar.set_postfix({"Valid": len(results), "Skipped": len(skipped)})

                if processed_count % checkpoint_interval == 0:
                    save_checkpoint(checkpoint_file, processed_count, results,
                                     skipped, list(processed_rows), superclass_cache)
                    if results:
                        pd.DataFrame(results).to_csv(
                            output_csv.replace(".csv", "_temp.csv"), index=False
                        )

    except KeyboardInterrupt:
        tqdm.write("\n⚠️  Interrupted. Saving progress...")
        save_checkpoint(checkpoint_file, processed_count, results, skipped,
                         list(processed_rows), superclass_cache)
        if results:
            out = output_csv.replace(".csv", "_interrupted.csv")
            pd.DataFrame(results).to_csv(out, index=False)
            tqdm.write(f"💾 Results saved to {out}")
        return

    except Exception as e:
        tqdm.write(f"\n💥 Unexpected error: {e}")
        save_checkpoint(checkpoint_file, processed_count, results, skipped,
                         list(processed_rows), superclass_cache)
        if results:
            out = output_csv.replace(".csv", "_error.csv")
            pd.DataFrame(results).to_csv(out, index=False)
            tqdm.write(f"💾 Results saved to {out}")
        raise

    finally:
        session.close()

    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(output_csv, index=False)
        tqdm.write(f"✅ Saved {len(df_out)} valid instance-level pairs to: {output_csv}")
        temp_output = output_csv.replace(".csv", "_temp.csv")
        if os.path.exists(temp_output):
            os.remove(temp_output)
    else:
        tqdm.write("⚠️  No valid instance-level pairs generated.")

    if skipped:
        with open("no_superclass_instances.txt", "w") as f:
            for uri in skipped:
                f.write(uri + "\n")
        tqdm.write(f"📝 Logged {len(skipped)} skipped instances to: no_superclass_instances.txt")

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        tqdm.write(f"🎉 Done. Cleaned up checkpoint file: {checkpoint_file}")


if __name__ == "__main__":
    generate_valid_instance_pairs(
        "./contradictions_instances/filtered_instance_disjoint_contradictions.csv",
        "disjoint_pairs_with_contradictions.csv",   # set to None to skip the guard
        "./contradictions_instances/non_contradictions_instances_v2.csv",
        checkpoint_interval=10
    )