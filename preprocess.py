"""Fill missing (NaN) values in the Track 2 training dataset.

Usage:
    python preprocess.py [input_csv] [output_csv]

Imputation rules (applied in this order, repeated for up to MAX_CYCLES cycles):
  1. merchant_category     <- mode of merchant_category for the same transaction_hour
  2. country               <- mode of country for the same merchant_category
  3. transaction_channel   <- mode of transaction_channel for the same merchant_category
  4. transactions_last_24h <- transactions_last_1h
  5. spend_last_24h        <- transaction_amount * transactions_last_1h
  6. account_age           <- median account_age for the same country
                              (fallback: median of the whole column)
  7. new_device            <- mode of new_device for the same country
                              (fallback: median of the whole column)
  8. transactions_last_1h  <- median transactions_last_1h for the same transaction_hour

New feature:
  avg_spend_per_txn_24h = spend_last_24h / transactions_last_24h
  (falls back to transaction_amount when transactions_last_24h is 0)
"""
import sys

import pandas as pd

MAX_CYCLES = 3
UNTOUCHED_COLS = ["id", "transaction_amount", "transaction_hour", "fraud"]


def group_mode(df, target, by):
    """Most common value of `target` within each `by` group (ties -> first sorted)."""
    return df.dropna(subset=[target, by]).groupby(by)[target].agg(lambda s: s.mode().iloc[0])


def group_median(df, target, by):
    return df.dropna(subset=[target, by]).groupby(by)[target].median()


def fill_from_group(df, target, by, stats):
    """Fill NaNs in `target` by looking up the row's `by` value in `stats`. Returns #filled."""
    missing = df[target].isna()
    df.loc[missing, target] = df.loc[missing, by].map(stats)
    return int(missing.sum() - df[target].isna().sum())


def fill_with_value(df, target, values):
    missing = df[target].isna()
    df.loc[missing, target] = values[missing] if isinstance(values, pd.Series) else values
    return int(missing.sum() - df[target].isna().sum())


def run_cycle(df):
    filled = {}
    filled["1. merchant_category (mode by hour)"] = fill_from_group(
        df, "merchant_category", "transaction_hour",
        group_mode(df, "merchant_category", "transaction_hour"))
    filled["2. country (mode by merchant)"] = fill_from_group(
        df, "country", "merchant_category", group_mode(df, "country", "merchant_category"))
    filled["3. transaction_channel (mode by merchant)"] = fill_from_group(
        df, "transaction_channel", "merchant_category",
        group_mode(df, "transaction_channel", "merchant_category"))
    filled["4. transactions_last_24h (= last 1h)"] = fill_with_value(
        df, "transactions_last_24h", df["transactions_last_1h"])
    filled["5. spend_last_24h (= amount x last 1h)"] = fill_with_value(
        df, "spend_last_24h", df["transaction_amount"] * df["transactions_last_1h"])

    n = fill_from_group(df, "account_age", "country", group_median(df, "account_age", "country"))
    filled["6. account_age (median by country)"] = n
    filled["6. account_age (column median fallback)"] = fill_with_value(
        df, "account_age", df["account_age"].median())

    n = fill_from_group(df, "new_device", "country", group_mode(df, "new_device", "country"))
    filled["7. new_device (mode by country)"] = n
    filled["7. new_device (column median fallback)"] = fill_with_value(
        df, "new_device", df["new_device"].median())

    filled["8. transactions_last_1h (median by hour)"] = fill_from_group(
        df, "transactions_last_1h", "transaction_hour",
        group_median(df, "transactions_last_1h", "transaction_hour"))
    return filled


def report_anomalies(df):
    print("\n=== Anomaly check on untouched columns ===")
    problems = 0
    for col in UNTOUCHED_COLS:
        if col not in df.columns:
            print(f"  ANOMALY: expected column '{col}' is missing from the dataset")
            problems += 1
            continue
        n = int(df[col].isna().sum())
        if n:
            print(f"  ANOMALY: '{col}' has {n} missing values")
            problems += 1
    if df["id"].duplicated().any():
        print(f"  ANOMALY: {int(df['id'].duplicated().sum())} duplicate ids")
        problems += 1
    if (df["transaction_amount"] <= 0).any():
        print(f"  ANOMALY: {int((df['transaction_amount'] <= 0).sum())} non-positive amounts")
        problems += 1
    if (~df["transaction_hour"].between(0, 23)).any():
        print("  ANOMALY: transaction_hour outside 0-23")
        problems += 1
    if not set(df["fraud"].dropna().unique()) <= {0, 1}:
        print("  ANOMALY: fraud contains values other than 0/1")
        problems += 1
    if not problems:
        print("  None found: id, transaction_amount, transaction_hour and fraud are complete and valid.")


def main(src="data/Track_2_Training_Dataset.csv", dst="data/Track_2_Training_Dataset_clean.csv"):
    df = pd.read_csv(src)
    print(f"Loaded {len(df)} rows x {df.shape[1]} columns from {src}")
    report_anomalies(df)

    print("\nMissing values before imputation:")
    print(df.isna().sum()[lambda s: s > 0].to_string())

    for cycle in range(1, MAX_CYCLES + 1):
        before = int(df.isna().sum().sum())
        print(f"\n=== Currently on cycle {cycle} of {MAX_CYCLES} (missing cells at start: {before}) ===")
        for rule, n in run_cycle(df).items():
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

    for col in ["transactions_last_24h", "transactions_last_1h", "account_age", "new_device"]:
        if df[col].notna().all():
            df[col] = df[col].astype(int)

    remaining = df.isna().sum()[lambda s: s > 0]
    print("\n=== Final result ===")
    if remaining.empty:
        print("No missing (NaN) cells remain in the dataset.")
    else:
        print(f"{int(remaining.sum())} missing cells remain after {MAX_CYCLES} cycles:")
        print(remaining.to_string())

    df.to_csv(dst, index=False)
    print(f"Saved cleaned dataset to {dst}")
    return df


if __name__ == "__main__":
    main(*sys.argv[1:3])
