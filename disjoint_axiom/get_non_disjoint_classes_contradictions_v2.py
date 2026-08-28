import pandas as pd
import requests
import time
import random
import os
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

def create_robust_session():
    """Create a requests session with retry strategy and robust settings."""
    session = requests.Session()
    
    # Define retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    # Mount adapter with retry strategy
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set timeout and headers
    session.headers.update({
        "Accept": "application/sparql-results+json",
        "User-Agent": "SubclassBot/1.0 (research@example.com)"
    })
    
    return session

def load_checkpoint(checkpoint_file):
    """Load checkpoint data if it exists."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            tqdm.write(f"📁 Loaded checkpoint from {checkpoint_file}")
            tqdm.write(f"✅ Previously processed: {data['processed_count']} items")
            tqdm.write(f"📊 Results so far: {len(data['results'])}")
            return data
        except Exception as e:
            tqdm.write(f"❌ Error loading checkpoint: {e}")
            return None
    return None

def save_checkpoint(checkpoint_file, processed_count, results, no_superclass_entities, processed_entities):
    """Save current progress to checkpoint file."""
    checkpoint_data = {
        'processed_count': processed_count,
        'results': results,
        'no_superclass_entities': no_superclass_entities,
        'processed_entities': processed_entities
    }
    
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        tqdm.write(f"💾 Checkpoint saved: {processed_count} items processed")
    except Exception as e:
        tqdm.write(f"❌ Error saving checkpoint: {e}")

def load_disjoint_contradictions(csv_path):
    df = pd.read_csv(csv_path)
    return df[["disjoint_entity1", "disjoint_entity2", "violating_subclass"]].dropna()

def load_disjoint_class_map(csv_path):
    df = pd.read_csv(csv_path)
    class_map = {}
    for _, row in df.iterrows():
        entity1 = row["entity1"]
        entity2 = row["entity2"]
        cls = row["class"]
        class_map[(entity1, entity2)] = cls
        class_map[(entity2, entity1)] = cls  # allow lookup in either direction
    return class_map

def get_single_subclassof_uri(entity_uri, session, max_retries=3):
    """Get superclass with robust error handling and retries."""
    qid = entity_uri.split("/")[-1]
    query = f"""
    SELECT ?superclass WHERE {{
      wd:{qid} wdt:P279 ?superclass .
    }}
    LIMIT 1
    """
    url = 'https://query.wikidata.org/sparql'
    
    for attempt in range(max_retries):
        try:
            response = session.get(
                url, 
                params={'query': query}, 
                timeout=(10, 30)  # (connection, read) timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                bindings = data['results']['bindings']
                if bindings:
                    return bindings[0]['superclass']['value']
                else:
                    tqdm.write(f"⚠️  No superclass found for {qid}")
                    return None
            else:
                tqdm.write(f"🌐 HTTP {response.status_code} for {qid}")
                if response.status_code == 429:  # Rate limited
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None
                
        except requests.exceptions.ConnectionError as e:
            tqdm.write(f"🔌 Connection error for {qid} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt + random.uniform(0, 1)
                tqdm.write(f"⏳ Waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
            else:
                tqdm.write(f"❌ Failed to fetch superclass for {qid} after {max_retries} attempts")
                return None
                
        except requests.exceptions.Timeout as e:
            tqdm.write(f"⏰ Timeout error for {qid} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
                
        except Exception as e:
            tqdm.write(f"💥 Unexpected error for {qid}: {e}")
            return None
    
    return None

def generate_valid_subclass_pairs(contradictions_csv, disjoint_class_csv, output_csv, checkpoint_interval=10):
    tqdm.write("🚀 Starting to generate valid subclass pairs...")
    
    # Define checkpoint file
    checkpoint_file = "subclass_processing_checkpoint.json"
    
    # Load checkpoint if it exists
    checkpoint_data = load_checkpoint(checkpoint_file)
    
    tqdm.write("📂 Loading data...")
    contradiction_rows = load_disjoint_contradictions(contradictions_csv)
    disjoint_class_map = load_disjoint_class_map(disjoint_class_csv)
    tqdm.write(f"📋 Loaded {len(contradiction_rows)} contradiction rows")
    tqdm.write(f"🗂️  Loaded {len(disjoint_class_map)} disjoint class mappings")

    # Initialize variables
    if checkpoint_data:
        results = checkpoint_data['results']
        no_superclass_entities = checkpoint_data['no_superclass_entities']
        processed_count = checkpoint_data['processed_count']
        processed_entities = set(checkpoint_data['processed_entities'])
        start_from = processed_count
        tqdm.write(f"🔄 Resuming from checkpoint: {processed_count} items already processed")
    else:
        results = []
        no_superclass_entities = []
        processed_count = 0
        processed_entities = set()
        start_from = 0
        tqdm.write("🆕 Starting fresh processing...")

    # Create robust session
    session = create_robust_session()
    
    # Calculate remaining items for progress bar
    total_items = len(contradiction_rows)
    remaining_items = total_items - start_from
    
    try:
        # Create progress bar
        with tqdm(total=remaining_items, 
                 desc="Processing entities", 
                 unit="entity",
                 initial=0,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            for index, row in contradiction_rows.iterrows():
                entity1 = row["disjoint_entity1"]
                entity2 = row["disjoint_entity2"]
                violating_subclass = row["violating_subclass"]

                # Skip if already processed
                row_key = f"{entity1}_{entity2}_{violating_subclass}"
                if row_key in processed_entities:
                    continue

                qid = violating_subclass.split("/")[-1]
                pbar.set_description(f"Processing {qid}")
                
                superclass = get_single_subclassof_uri(violating_subclass, session)
                
                if not superclass:
                    no_superclass_entities.append(violating_subclass)
                    processed_entities.add(row_key)
                    processed_count += 1
                    pbar.update(1)
                    
                    # Save checkpoint
                    if processed_count % checkpoint_interval == 0:
                        save_checkpoint(checkpoint_file, processed_count, results, no_superclass_entities, list(processed_entities))
                        # Also save intermediate results
                        if results:
                            df_temp = pd.DataFrame(results)
                            temp_output = output_csv.replace('.csv', '_temp.csv')
                            df_temp.to_csv(temp_output, index=False)
                    continue

                disjoint_class = disjoint_class_map.get((entity1, entity2))
                if disjoint_class and superclass == disjoint_class:
                    tqdm.write(f"⏭️  Skipping {qid}: superclass matches disjoint class")
                    no_superclass_entities.append(violating_subclass)
                    processed_entities.add(row_key)
                    processed_count += 1
                    pbar.update(1)
                    
                    # Save checkpoint
                    if processed_count % checkpoint_interval == 0:
                        save_checkpoint(checkpoint_file, processed_count, results, no_superclass_entities, list(processed_entities))
                        # Also save intermediate results
                        if results:
                            df_temp = pd.DataFrame(results)
                            temp_output = output_csv.replace('.csv', '_temp.csv')
                            df_temp.to_csv(temp_output, index=False)
                    continue

                disjoint_entity2 = random.choice([entity1, entity2])

                results.append({
                    "disjoint_entity1": superclass,
                    "disjoint_entity2": disjoint_entity2,
                    "violating_subclass": violating_subclass
                })
                
                processed_entities.add(row_key)
                processed_count += 1
                pbar.update(1)
                
                # Update stats in progress bar
                pbar.set_postfix({
                    'Valid': len(results),
                    'Skipped': len(no_superclass_entities)
                })

                # Save checkpoint every N items
                if processed_count % checkpoint_interval == 0:
                    save_checkpoint(checkpoint_file, processed_count, results, no_superclass_entities, list(processed_entities))
                    # Also save intermediate results
                    if results:
                        df_temp = pd.DataFrame(results)
                        temp_output = output_csv.replace('.csv', '_temp.csv')
                        df_temp.to_csv(temp_output, index=False)

                # Polite delay between requests
                time.sleep(1.0)

    except KeyboardInterrupt:
        tqdm.write("\n⚠️  Process interrupted by user. Saving current progress...")
        save_checkpoint(checkpoint_file, processed_count, results, no_superclass_entities, list(processed_entities))
        if results:
            df_temp = pd.DataFrame(results)
            temp_output = output_csv.replace('.csv', '_interrupted.csv')
            df_temp.to_csv(temp_output, index=False)
            tqdm.write(f"💾 Results saved to {temp_output}")
        return
    
    except Exception as e:
        tqdm.write(f"\n💥 Unexpected error occurred: {e}")
        tqdm.write("💾 Saving current progress...")
        save_checkpoint(checkpoint_file, processed_count, results, no_superclass_entities, list(processed_entities))
        if results:
            df_temp = pd.DataFrame(results)
            temp_output = output_csv.replace('.csv', '_error.csv')
            df_temp.to_csv(temp_output, index=False)
            tqdm.write(f"💾 Results saved to {temp_output}")
        raise
    
    finally:
        # Close session
        session.close()

    # Save final results
    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(output_csv, index=False)
        tqdm.write(f"✅ Saved {len(df_out)} valid subclass relations to: {output_csv}")
        
        # Clean up temporary files
        temp_output = output_csv.replace('.csv', '_temp.csv')
        if os.path.exists(temp_output):
            os.remove(temp_output)
            tqdm.write(f"🧹 Cleaned up temporary file: {temp_output}")
    else:
        tqdm.write("⚠️  No valid subclass relations found.")

    # Save skipped entities
    if no_superclass_entities:
        with open("no_superclass_entities.txt", "w") as f:
            for uri in no_superclass_entities:
                f.write(uri + "\n")
        tqdm.write(f"📝 Logged {len(no_superclass_entities)} skipped entities to: no_superclass_entities.txt")

    # Clean up checkpoint file on successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        tqdm.write(f"🎉 Processing completed successfully. Cleaned up checkpoint file: {checkpoint_file}")

if __name__ == "__main__":
    generate_valid_subclass_pairs(
        "./contradictions/disjoint_contradictions.csv",
        "disjoint_pairs_with_contradictions.csv",
        "./contradictions/non_contradictions_explicit_v2.csv",
        checkpoint_interval=10  # Save checkpoint every 10 processed items
    )