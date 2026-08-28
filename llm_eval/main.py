import argparse
import pandas as pd
import os
import torch
from tqdm import tqdm
from llm_evaluator import ContradictionEvaluator

LANGUAGE_CODES = {
    "english":    "en",
    "spanish":    "es",
    "french":     "fr"
}

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Evaluation Script")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--wrapper", required=True, choices=["openai", "huggingface"])
    parser.add_argument("--api-key", help="Environment variable name for API key")
    parser.add_argument("--statement-columns", nargs="+", required=True)
    parser.add_argument("--label-column", default="label", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()

def detect_language(filename: str) -> str:
    stem   = os.path.splitext(os.path.basename(filename))[0]   # strip extension
    tokens = [t.lower().strip() for t in stem.split("_")]

    for token in tokens:
        if token in LANGUAGE_CODES:
            return token, LANGUAGE_CODES[token]

    return None, "en"

def extract_model_dir(output_filename: str) -> str:
    """
    Extracts the model name from the output filename to use as a subdirectory.
    Convention: <model_name>_<rest_of_name>.tsv
    e.g. 'qwen3.5-9b_explicit_cardinality_english_output.tsv' -> 'qwen3.5-9b'
    The extracted name is title-cased for readability, e.g. 'Qwen3.5-9B'.
    """
    stem = os.path.splitext(os.path.basename(output_filename))[0]
    model_token = stem.split("_")[0]          # e.g. 'qwen3.5-9b'

    # Title-case each dash-separated segment: 'qwen3.5-9b' -> 'Qwen3.5-9B'
    parts = model_token.split("-")
    formatted = "-".join(
        p.upper() if p.replace(".", "").isdigit() or p[-1:].isalpha() and p[:-1].replace(".", "").isdigit()
        else p.capitalize()
        for p in parts
    )
    return formatted

def get_output_paths(output_filename: str, predictions_root: str, metrics_root: str):
    """
    Resolves the full paths for the prediction TSV and metrics TXT,
    creating model-specific subdirectories if they don't exist.

    Structure:
        models_predictions/
            <ModelName>/
                <output_filename>
        models_metrics/
            <ModelName>/
                metrics_<output_filename>.txt
    """
    model_dir = extract_model_dir(output_filename)

    pred_dir = os.path.join(predictions_root, model_dir)
    metrics_dir = os.path.join(metrics_root, model_dir)

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    pred_path    = os.path.join(pred_dir, output_filename)
    metrics_name = f"metrics_{output_filename.replace('.tsv', '.txt')}"
    metrics_path = os.path.join(metrics_dir, metrics_name)

    return pred_path, metrics_path

def load_dataset(path, statement_columns, label_column):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # This is much faster than iterrows for 2M rows
    subset = df[statement_columns + [label_column]].to_dict('records')
    
    dataset = []
    for row in subset:
        dataset.append({
            "statements": [str(row[col]) for col in statement_columns],
            "label": str(row[label_column]).strip().lower()
        })
    return df, dataset

def main():
    import torch
    args = parse_args()
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)

    candidate_lang, lang_code = detect_language(args.input)
    if candidate_lang:
        print(f"Detected language: '{candidate_lang}' -> '{lang_code}'")
    else:
        print(f"Warning: no known language found in filename '{os.path.basename(args.input)}' — defaulting to English.")

    # Initialize model
    if args.wrapper == "openai":
        api_key = os.getenv(args.api_key)
        if api_key is None:
            raise ValueError(f"Environment variable {args.api_key} not found.")
        evaluator = ContradictionEvaluator(
            model_name=args.model,
            model_wrapper="openai",
            api_key=api_key,
            thinking=args.enable_thinking,
            language=lang_code
        )
    else:
        evaluator = ContradictionEvaluator(
            model_name=args.model,
            model_wrapper="huggingface",
            thinking=args.enable_thinking,
            language=lang_code
        )

    # Load dataset
    df, dataset = load_dataset(args.input, args.statement_columns, args.label_column)

    # Resolve model-scoped output paths
    output_path, metrics_path = get_output_paths(
        output_filename=args.output,
        predictions_root="models_predictions",
        metrics_root="models_metrics"
    )
    print(f"Predictions -> {output_path}")
    print(f"Metrics     -> {metrics_path}")

    # --- HIGH-SPEED CHECKPOINTING LOGIC ---
    if os.path.exists(output_path):
        print("Resuming from existing file...")
        # Only load the prediction column to find the resume point quickly
        df_resume = pd.read_csv(output_path, sep="\t", usecols=["prediction"], low_memory=False)
        start_idx = df_resume["prediction"].notna().sum()
        del df_resume
    else:
        start_idx = 0
        # Create file with headers only
        empty_df = df.iloc[:0].copy()
        empty_df["prediction"] = None
        empty_df["raw_output"] = None
        empty_df.to_csv(output_path, sep="\t", index=False)

    print(f"Starting evaluation from index {start_idx}...")
    
    i = start_idx
    pbar = tqdm(total=len(dataset), initial=start_idx, desc="Evaluating")

    while i < len(dataset):
        batch = dataset[i : i + args.batch_size]

        try:
            if args.wrapper == "huggingface":
                # OOM Protection loop
                while True:
                    try:
                        prompts = evaluator.build_prompt_batch([s["statements"] for s in batch])
                        raw_outputs = evaluator.model.generate(prompts, enable_thinking=evaluator.thinking)
                        break
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"⚠ OOM: Reducing batch size...")
                            import torch
                            torch.cuda.empty_cache()
                            args.batch_size //= 2
                            if args.batch_size <= 0: raise RuntimeError("Batch size 0")
                            batch = batch[:args.batch_size]
                        else: raise e
            else:
                raw_outputs = []
                for sample in batch:
                    prompt = evaluator.build_prompt_batch([sample["statements"]])[0]
                    raw_outputs.append(evaluator.model.generate(prompt, enable_thinking=evaluator.thinking))

            predictions = [evaluator.parse_output(o) for o in raw_outputs]

            # Prepare slice for APPENDING
            # This is much faster than overwriting 2.1M rows every time
            checkpoint_slice = df.iloc[i : i + len(predictions)].copy()
            checkpoint_slice["prediction"] = predictions
            checkpoint_slice["raw_output"] = raw_outputs
            
            checkpoint_slice.to_csv(output_path, sep="\t", index=False, header=False, mode='a')

            i += len(predictions)
            pbar.update(len(predictions))

        except Exception as e:
            print(f"Critical error at index {i}: {e}")
            break

    pbar.close()

    # --- COMPUTE FINAL METRICS ---
    print("\nReading completed file for final metrics...")
    df_final = pd.read_csv(output_path, sep="\t", low_memory=False)
    
    # Drop rows that haven't been processed if the script was interrupted
    df_final = df_final[df_final["prediction"].notna()]
    
    y_true = df_final[args.label_column].tolist()
    y_pred = df_final["prediction"].tolist()

    metrics = evaluator.compute_metrics(y_true, y_pred)

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k.capitalize()}: {v:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    # Save metrics text file
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    main()