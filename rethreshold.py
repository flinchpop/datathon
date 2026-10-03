"""Re-cut the 0/1 `fraud` column of an existing submission without retraining.

Usage: python rethreshold.py submission.csv out.csv [--rule binary_pr_auc|f1] [--threshold T]
  submission.csv must have a probability column (fraud_probability by default; see --proba-col).

The probability column is copied unchanged, so submitting the output alongside the
original is a clean test of which column the leaderboard scores: if the score moves,
it is reading the 0/1 column.
"""
import argparse

import pandas as pd

from validation import plugin_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--proba-col", default="fraud_probability")
    ap.add_argument("--label-col", default="fraud")
    ap.add_argument("--rule", choices=["binary_pr_auc", "f1"], default="binary_pr_auc")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    proba = df[args.proba_col]
    before = int(df[args.label_col].sum()) if args.label_col in df else None
    if args.threshold is None:
        threshold, exp = plugin_threshold(proba, args.rule)
        note = (f"plug-in {args.rule}: expected precision {exp['precision']:.2f}, recall {exp['recall']:.2f}, "
                f"{args.rule} {exp[args.rule]:.3f} (assumes the probabilities are calibrated)")
    else:
        threshold, note = args.threshold, "fixed"
    df[args.label_col] = (proba >= threshold).astype(int)
    df.to_csv(args.output, index=False)
    print(f"{args.input}: flagged {before} -> {int(df[args.label_col].sum())} rows "
          f"(threshold {threshold:.3f}, {note}) -> {args.output}")


if __name__ == "__main__":
    main()
