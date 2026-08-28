import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Downsample a CSV to a target size, filtering by language label columns "
            "and maximising coverage of distinct disjoint class pairs. "
            "Optionally bias selection toward pairs that overlap with a reference "
            "positive examples file."
        )
    )
    parser.add_argument("--input",   required=True, help="Path to input CSV file.")
    parser.add_argument("--output",  required=True, help="Path to output CSV file.")
    parser.add_argument(
        "--languages", nargs="+", required=True,
        help=(
            "Language suffixes to filter on. Each must match a '<lang>_label' column "
            "in the file. Example: --languages english spanish chinese"
        )
    )
    parser.add_argument(
        "--size", type=int, default=2500,
        help="Target number of rows in the output (default: 2500)."
    )
    parser.add_argument(
        "--entity1-column", default="disjoint_entity1",
        help="Name of the first disjoint class column (default: 'disjoint_entity1')."
    )
    parser.add_argument(
        "--entity2-column", default="disjoint_entity2",
        help="Name of the second disjoint class column (default: 'disjoint_entity2')."
    )
    parser.add_argument(
        "--positives", default=None,
        help=(
            "Path to the downsampled positive examples CSV. When provided, negative "
            "rows whose entities overlap with the positive set are prioritised."
        )
    )
    parser.add_argument(
        "--positives-columns", nargs="+", default=["disjoint_entity1", "disjoint_entity2", "violating_subclass"],
        help=(
            "Columns in the positives file to extract reference entities from "
            "(default: disjoint_entity1 disjoint_entity2 violating_subclass)."
        )
    )
    return parser.parse_args()


def normalise_id(raw: str) -> str:
    return str(raw).strip().split("/")[-1]


def filter_by_languages(df: pd.DataFrame, languages: list[str]) -> pd.DataFrame:
    """Keep only rows where every requested language label column is True."""
    missing = [f"{l}_label" for l in languages if f"{l}_label" not in df.columns]
    if missing:
        raise ValueError(
            f"The following label columns were not found in the file: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    mask = pd.Series(True, index=df.index)
    for lang in languages:
        mask &= df[f"{lang}_label"].astype(bool)
    return df[mask].copy()


def make_pair_key(row, col1: str, col2: str) -> tuple:
    """Canonical sorted pair key so (A,B) and (B,A) are the same."""
    a = normalise_id(str(row[col1]))
    b = normalise_id(str(row[col2]))
    return (a, b) if a <= b else (b, a)


def load_positive_entities(positives_path: str, columns: list[str]) -> set[str]:
    """Return the set of normalised entity IDs that appear in the positive file."""
    pos_df = pd.read_csv(positives_path)
    entities = set()
    for col in columns:
        if col in pos_df.columns:
            for val in pos_df[col].dropna():
                entities.add(normalise_id(str(val)))
    print(f"  Loaded {len(entities)} unique entities from positive examples.")
    return entities


def downsample_max_pair_coverage(
    df: pd.DataFrame,
    target: int,
    col1: str,
    col2: str,
    positive_entities: set[str] | None = None,
) -> pd.DataFrame:
    """
    Select up to `target` rows maximising coverage of distinct pairs.

    When `positive_entities` is provided the pool is split into two tiers:
      - Tier 1 (priority): rows where at least one entity appears in the
        positive set. These are round-robin sampled first.
      - Tier 2 (remainder): all other rows, used to fill any remaining slots.

    Within each tier the same round-robin-by-pair strategy is applied.
    """
    if len(df) <= target:
        print(
            f"  Filtered dataset ({len(df)} rows) is already within "
            f"target ({target}). No downsampling needed."
        )
        return df

    for col in (col1, col2):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. "
                f"Use --entity1-column / --entity2-column to specify the correct names.\n"
                f"Available columns: {list(df.columns)}"
            )

    df = df.copy()
    df["_pair_key"] = df.apply(lambda r: make_pair_key(r, col1, col2), axis=1)

    if positive_entities:
        def overlaps(row) -> bool:
            return (
                normalise_id(str(row[col1])) in positive_entities or
                normalise_id(str(row[col2])) in positive_entities
            )
        mask = df.apply(overlaps, axis=1)
        tier1 = df[mask].copy()
        tier2 = df[~mask].copy()
        print(f"  Tier 1 (overlaps with positives): {len(tier1)} rows")
        print(f"  Tier 2 (no overlap):              {len(tier2)} rows")
    else:
        tier1 = df.copy()
        tier2 = pd.DataFrame(columns=df.columns)

    def round_robin(pool: pd.DataFrame, slots: int) -> list:
        if pool.empty or slots == 0:
            return []
        groups = {
            pair: group.sample(frac=1, random_state=42).reset_index(drop=True)
            for pair, group in pool.groupby("_pair_key")
        }
        selected = []
        round_num = 0
        while len(selected) < slots:
            added = 0
            for pair, group in groups.items():
                if round_num < len(group):
                    selected.append(group.iloc[round_num])
                    added += 1
                    if len(selected) == slots:
                        break
            if added == 0:
                break
            round_num += 1
        return selected

    # Fill from tier 1 first, then tier 2 for any remaining slots
    selected = round_robin(tier1, target)
    remaining = target - len(selected)
    if remaining > 0 and not tier2.empty:
        print(f"  Filling {remaining} remaining slot(s) from tier 2…")
        selected += round_robin(tier2, remaining)

    result = pd.DataFrame(selected).reset_index(drop=True)
    result = result.drop(columns=["_pair_key"])
    return result


def fill_with_relaxed_languages(
    df: pd.DataFrame,
    already_selected: pd.DataFrame,
    languages: list[str],
    remaining: int,
    col1: str,
    col2: str,
    positive_entities: set[str] | None,
) -> pd.DataFrame:
    """
    Fill `remaining` slots by progressively relaxing the language filter,
    dropping the least-covered language first each time.
    Only rows not already in `already_selected` are considered.
    """
    selected_idx = set(already_selected.index)

    lang_counts = {
        lang: int(df[f"{lang}_label"].astype(str).str.lower().eq("true").sum())
        for lang in languages
    }
    langs_by_coverage = sorted(languages, key=lambda l: lang_counts[l])
    print(f"  Language coverage (ascending): { {l: lang_counts[l] for l in langs_by_coverage} }")

    extra_frames = []
    slots_left   = remaining
    relaxed      = list(languages)

    while slots_left > 0 and len(relaxed) > 0:
        mask = pd.Series(True, index=df.index)
        for lang in relaxed:
            mask &= df[f"{lang}_label"].astype(str).str.lower().eq("true")
        candidates = df[mask & ~df.index.isin(selected_idx)]

        if len(candidates) > 0:
            batch = downsample_max_pair_coverage(
                candidates, slots_left, col1, col2, positive_entities
            )
            print(
                f"  Relaxed to [{', '.join(relaxed)}]: "
                f"{len(candidates)} candidates → taking {len(batch)} rows."
            )
            extra_frames.append(batch)
            selected_idx.update(batch.index)
            slots_left -= len(batch)

        if slots_left == 0:
            break

        dropped = langs_by_coverage.pop(0)
        relaxed.remove(dropped)
        print(f"  Still need {slots_left} rows — dropping '{dropped}' from language filter.")

    if slots_left > 0:
        print(f"  Warning: could not fill {slots_left} remaining slot(s) even after relaxing all language filters.")

    return pd.concat(extra_frames, ignore_index=True) if extra_frames else pd.DataFrame(columns=df.columns)

def main():
    args = parse_args()

    print(f"Loading input file: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} rows, {len(df.columns)} columns.")

    print(f"Filtering rows where all language labels are True: {args.languages}")
    filtered = filter_by_languages(df, args.languages)
    print(f"  {len(filtered)} rows remain after filtering.")

    positive_entities = None
    if args.positives:
        print(f"Loading positive examples for entity overlap: {args.positives}")
        positive_entities = load_positive_entities(args.positives, args.positives_columns)

    if len(filtered) >= args.size:
        pair_keys_before = filtered.apply(
            lambda r: make_pair_key(r, args.entity1_column, args.entity2_column), axis=1
        )
        print(f"  Distinct disjoint pairs before downsampling: {pair_keys_before.nunique()}")
        print(f"Downsampling to {args.size} rows with maximum disjoint pair coverage…")
        result = downsample_max_pair_coverage(
            filtered, args.size,
            args.entity1_column, args.entity2_column,
            positive_entities=positive_entities,
        )
    else:
        shortfall = args.size - len(filtered)
        lang_str  = ", ".join(args.languages)
        print(
            f"\n  Warning: only {len(filtered)} rows have labels in [{lang_str}]. "
            f"Need {shortfall} more — filling by relaxing language filter."
        )
        core = downsample_max_pair_coverage(
            filtered, args.size,
            args.entity1_column, args.entity2_column,
            positive_entities=positive_entities,
        )
        extra = fill_with_relaxed_languages(
            df, core, args.languages, args.size - len(core),
            args.entity1_column, args.entity2_column,
            positive_entities,
        )
        result = pd.concat([core, extra], ignore_index=True)

    pair_keys_after = result.apply(
        lambda r: make_pair_key(r, args.entity1_column, args.entity2_column), axis=1
    )
    print(f"  Distinct disjoint pairs in output: {pair_keys_after.nunique()}")
    print(f"  Final row count: {len(result)}")

    result.to_csv(args.output, index=False)
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()