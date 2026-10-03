"""Fill missing (NaN) values in the Track 2 fraud dataset.

Usage:
    # Training data: learn the fill values, save them, and clean the file.
    python preprocess.py data/Track_2_Training_Dataset.csv data/Track_2_Training_Dataset_clean.csv

    # Test / new data: reuse the fill values learned from training (no leakage).
    python preprocess.py data/test.csv data/test_clean.csv --stats imputation_stats.json

Imputation rules (applied in this order, repeated for up to MAX_CYCLES cycles).
All modes/medians are learned from the observed (non-missing) training values only.
  1. merchant_category     <- mode of merchant_category for the same transaction_hour
  2. country               <- mode of country for the same merchant_category
  3. transaction_channel   <- mode of transaction_channel for the same merchant_category
  4. transactions_last_24h <- median transactions_last_24h for the same transactions_last_1h
  5. spend_last_24h        <- transactions_last_24h * median spend-per-transaction (24h)
  6. account_age           <- median account_age for the same country
  7. new_device            <- mode of new_device for the same country
  8. transactions_last_1h  <- median transactions_last_1h for the same transaction_hour
Any value whose group is missing or unseen falls back to the whole-column mode/median.

New columns:
  <col>_was_missing      1 if <col> was NaN in the raw data, else 0 (one per imputed column)
  avg_spend_per_txn_24h  spend_last_24h / transactions_last_24h
                         (falls back to transaction_amount when transactions_last_24h is 0)
"""
import argparse
import json

import pandas as pd

MAX_CYCLES = 3
UNTOUCHED_COLS = ["id", "transaction_amount", "transaction_hour", "fraud"]
IMPUTED_COLS = ["merchant_category", "country", "transaction_channel", "transactions_last_24h",
                "spend_last_24h", "account_age", "new_device", "transactions_last_1h"]
INT_COLS = ["transactions_last_24h", "transactions_last_1h", "account_age", "new_device"]

# (target, group-by column, statistic) for the group-based rules, in rule order.
GROUP_RULES = [
    ("merchant_category", "transaction_hour", "mode"),
    ("country", "merchant_category", "mode"),
    ("transaction_channel", "merchant_category", "mode"),
    ("transactions_last_24h", "transactions_last_1h", "median"),
    ("account_age", "country", "median"),
    ("new_device", "country", "mode"),
    ("transactions_last_1h", "transaction_hour", "median"),
]


def _mode(s):
    """Most common value (ties -> first sorted)."""
    return s.mode().iloc[0]


def fit_stats(df):
    """Learn every fill value from the observed (non-missing) rows of the training data."""
    stats = {"groups": {}, "global": {}}
    for target, by, how in GROUP_RULES:
        d = df.dropna(subset=[target, by])
        g = d.groupby(by)[target].agg(_mode if how == "mode" else "median")
        stats["groups"][target] = [[k.item() if hasattr(k, "item") else k,
                                    v.item() if hasattr(v, "item") else v] for k, v in g.items()]
    for col in IMPUTED_COLS:
        s = df[col].dropna()
        # Categoricals use the column mode; numbers use the column median (rules 6/7 fallback).
        stats["global"][col] = _mode(s) if s.dtype == object or str(s.dtype) == "str" \
            else float(s.median())
    d = df.dropna(subset=["spend_last_24h", "transactions_last_24h"])
    d = d[d["transactions_last_24h"] > 0]
    stats["median_spend_per_txn_24h"] = float(
        (d["spend_last_24h"] / d["transactions_last_24h"]).median())
    return stats


def fill(df, target, values):
    """Fill NaNs in `target` from `values` (Series or scalar). Returns #cells filled."""
    missing = df[target].isna()
    df.loc[missing, target] = values[missing] if isinstance(values, pd.Series) else values
    return int(missing.sum() - df[target].isna().sum())


def fill_group(df, target, by, stats):
    return fill(df, target, df[by].map(dict(map(tuple, stats["groups"][target]))))


def run_cycle(df, stats, last_cycle):
    filled = {
        "1. merchant_category (mode by hour)":
            fill_group(df, "merchant_category", "transaction_hour", stats),
        "2. country (mode by merchant)":
            fill_group(df, "country", "merchant_category", stats),
        "3. transaction_channel (mode by merchant)":
            fill_group(df, "transaction_channel", "merchant_category", stats),
        "4. transactions_last_24h (median for same last-1h count)":
            fill_group(df, "transactions_last_24h", "transactions_last_1h", stats),
        "5. spend_last_24h (24h count x median spend per txn)":
            fill(df, "spend_last_24h",
                 df["transactions_last_24h"] * stats["median_spend_per_txn_24h"]),
        "6. account_age (median by country)":
            fill_group(df, "account_age", "country", stats),
        "7. new_device (mode by country)":
            fill_group(df, "new_device", "country", stats),
        "8. transactions_last_1h (median by hour)":
            fill_group(df, "transactions_last_1h", "transaction_hour", stats),
    }
    # Rules 6/7 fallback (and an unseen-group fallback for the others), only once the
    # group-by columns have had every chance to be filled in earlier cycles.
    if last_cycle or df[IMPUTED_COLS].isna().sum().sum() == 0:
        for col in IMPUTED_COLS:
            n = fill(df, col, stats["global"][col])
            if n:
                filled[f"   {col} (whole-column fallback)"] = n
    return filled


def report_anomalies(df):
    print("\n=== Anomaly check on untouched columns ===")
    problems = []
    for col in UNTOUCHED_COLS:
        if col not in df.columns:
            if col != "fraud":  # test data may legitimately have no label
                problems.append(f"expected column '{col}' is missing from the dataset")
        elif df[col].isna().any():
            problems.append(f"'{col}' has {int(df[col].isna().sum())} missing values")
    if "id" in df and df["id"].duplicated().any():
        problems.append(f"{int(df['id'].duplicated().sum())} duplicate ids")
    if "transaction_amount" in df and (df["transaction_amount"] <= 0).any():
        problems.append(f"{int((df['transaction_amount'] <= 0).sum())} non-positive amounts")
    if "transaction_hour" in df and (~df["transaction_hour"].dropna().between(0, 23)).any():
        problems.append("transaction_hour outside 0-23")
    if "fraud" in df and not set(df["fraud"].dropna().unique()) <= {0, 1}:
        problems.append("fraud contains values other than 0/1")
    for p in problems:
        print(f"  ANOMALY: {p}")
    if not problems:
        print("  None found: the untouched columns are complete and valid.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("src", nargs="?", default="data/Track_2_Training_Dataset.csv")
    ap.add_argument("dst", nargs="?", default="data/Track_2_Training_Dataset_clean.csv")
    ap.add_argument("--stats", help="reuse fill values from this JSON instead of learning them")
    ap.add_argument("--save-stats", default="imputation_stats.json",
                    help="where to save learned fill values (training mode only)")
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    print(f"Loaded {len(df)} rows x {df.shape[1]} columns from {args.src}")
    report_anomalies(df)

    if args.stats:
        with open(args.stats) as f:
            stats = json.load(f)
        print(f"\nUsing fill values learned from training data ({args.stats})")
    else:
        stats = fit_stats(df)
        with open(args.save_stats, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nLearned fill values from this file and saved them to {args.save_stats}")

    print("\nMissing values before imputation:")
    print(df.isna().sum()[lambda s: s > 0].to_string())
    for col in IMPUTED_COLS:
        df[f"{col}_was_missing"] = df[col].isna().astype(int)

    for cycle in range(1, MAX_CYCLES + 1):
        before = int(df.isna().sum().sum())
        print(f"\n=== Currently on cycle {cycle} of {MAX_CYCLES} (missing cells at start: {before}) ===")
        for rule, n in run_cycle(df, stats, last_cycle=cycle == MAX_CYCLES).items():
            if n:
                print(f"  {rule}: filled {n}")
        after = int(df.isna().sum().sum())
        print(f"  Cycle {cycle} done: {before - after} cells filled, {after} still missing")
        if after == 0:
            print("  No missing cells left, so no further cycles are needed.")
            break

    # New feature: average amount spent per transaction over the last 24h.
    zero = df["transactions_last_24h"] == 0
    df["avg_spend_per_txn_24h"] = (df["spend_last_24h"] / df["transactions_last_24h"]).where(
        ~zero, df["transaction_amount"])
    print(f"\nCreated avg_spend_per_txn_24h "
          f"({int(zero.sum())} rows with transactions_last_24h == 0 use transaction_amount instead)")

    for col in INT_COLS:
        if df[col].notna().all():
            df[col] = df[col].round().astype(int)

    remaining = df.isna().sum()[lambda s: s > 0]
    print("\n=== Final result ===")
    if remaining.empty:
        print("No missing (NaN) cells remain in the dataset.")
    else:
        print(f"{int(remaining.sum())} missing cells remain after {MAX_CYCLES} cycles:")
        print(remaining.to_string())

    df.to_csv(args.dst, index=False)
    print(f"Saved cleaned dataset to {args.dst}")


if __name__ == "__main__":
    main()
