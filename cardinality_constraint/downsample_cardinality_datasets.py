import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Downsample a CSV to a target size, filtering by language label columns "
            "and maximising coverage of unique properties. If fewer rows than the target "
            "have all requested languages, the remainder is filled with rows that cover "
            "as many languages as possible (dropping the least-covered language first)."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
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
        "--property-column", default="property",
        help="Name of the column containing property PIDs (default: 'property')."
    )
    return parser.parse_args()


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


def downsample_max_property_coverage(
    df: pd.DataFrame,
    target: int,
    property_col: str,
) -> pd.DataFrame:
    """
    Select up to `target` rows while maximising coverage of unique property values
    using a round-robin strategy across property groups.
    """
    if len(df) <= target:
        return df

    if property_col not in df.columns:
        print(f"  Warning: property column '{property_col}' not found — falling back to random sampling.")
        return df.sample(n=target, random_state=42).reset_index(drop=True)

    groups = {
        prop: group.sample(frac=1, random_state=42).reset_index(drop=True)
        for prop, group in df.groupby(property_col)
    }

    selected = []
    round_num = 0
    while len(selected) < target:
        added_this_round = 0
        for prop, group in groups.items():
            if round_num < len(group):
                selected.append(group.iloc[round_num])
                added_this_round += 1
                if len(selected) == target:
                    break
        if added_this_round == 0:
            break
        round_num += 1

    return pd.DataFrame(selected).reset_index(drop=True)


def fill_with_relaxed_languages(
    df: pd.DataFrame,
    already_selected: pd.DataFrame,
    languages: list[str],
    remaining: int,
    property_col: str,
) -> pd.DataFrame:
    """
    Fill `remaining` slots by progressively relaxing the language filter,
    dropping the least-covered language first each time.

    Only rows not already in `already_selected` are considered.
    Returns a DataFrame of the extra rows to append.
    """
    # Build a fingerprint set of already-selected rows to avoid duplicates.
    selected_idx = set(already_selected.index)

    # Sort languages by how many rows they cover (ascending) so we drop the
    # least-available language first when relaxing.
    lang_counts = {
        lang: int(df[f"{lang}_label"].astype(str).str.lower().eq("true").sum())
        for lang in languages
    }
    langs_by_coverage = sorted(languages, key=lambda l: lang_counts[l])
    print(f"  Language coverage (ascending): { {l: lang_counts[l] for l in langs_by_coverage} }")

    extra_frames = []
    slots_left   = remaining
    relaxed      = list(languages)  # start with all languages required

    while slots_left > 0 and len(relaxed) > 0:
        # Filter the full df by the current relaxed language set,
        # then exclude already-selected rows.
        mask = pd.Series(True, index=df.index)
        for lang in relaxed:
            mask &= df[f"{lang}_label"].astype(str).str.lower().eq("true")
        candidates = df[mask & ~df.index.isin(selected_idx)]

        if len(candidates) > 0:
            batch = downsample_max_property_coverage(candidates, slots_left, property_col)
            print(
                f"  Relaxed to [{', '.join(relaxed)}]: "
                f"{len(candidates)} candidates → taking {len(batch)} rows."
            )
            extra_frames.append(batch)
            selected_idx.update(batch.index)
            slots_left -= len(batch)

        if slots_left == 0:
            break

        # Drop the least-covered language and try again
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
    print(f"  {len(filtered)} rows remain after strict language filtering.")

    if len(filtered) >= args.size:
        # Happy path: enough rows with all languages — just downsample normally.
        unique_props = filtered[args.property_column].nunique() if args.property_column in filtered.columns else "N/A"
        print(f"  Unique properties before downsampling: {unique_props}")
        print(f"Downsampling to {args.size} rows with maximum property coverage…")
        result = downsample_max_property_coverage(filtered, args.size, args.property_column)

    else:
        # Not enough rows with all languages — take all of them, then fill the gap
        # by relaxing the language constraint one language at a time.
        shortfall = args.size - len(filtered)
        lang_str  = ", ".join(args.languages)
        print(
            f"\n  Warning: only {len(filtered)} rows have labels in [{lang_str}]. "
            f"Need {shortfall} more — filling by relaxing language filter."
        )

        core = downsample_max_property_coverage(filtered, args.size, args.property_column)
        extra = fill_with_relaxed_languages(
            df, core, args.languages, args.size - len(core), args.property_column
        )
        result = pd.concat([core, extra], ignore_index=True)

    if args.property_column in result.columns:
        print(f"  Unique properties in output: {result[args.property_column].nunique()}")
    print(f"  Final row count: {len(result)}")

    result.to_csv(args.output, index=False)
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()