#!/usr/bin/env python3
"""Validate the tab-separated interaction format expected by the experiments."""

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"user_id", "item_id", "skill_id", "correct"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv, sep="\t")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    responses = set(pd.to_numeric(frame["correct"], errors="raise").unique())
    if not responses.issubset({0, 1}):
        raise ValueError(f"correct must be binary, found {sorted(responses)}")
    for column in ["user_id", "item_id", "skill_id"]:
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or (values < 0).any():
            raise ValueError(f"{column} must contain non-negative integer IDs")

    if "split" in frame.columns:
        expected = {"train", "valid", "test"}
        observed = set(frame["split"].dropna().unique())
        if observed != expected:
            raise ValueError(f"Expected split values {sorted(expected)}, found {sorted(observed)}")
        memberships = frame.groupby("user_id")["split"].nunique()
        if (memberships > 1).any():
            raise ValueError("A user occurs in more than one split")

    print(f"rows={len(frame):,}")
    print(f"users={frame['user_id'].nunique():,}")
    print(f"items={frame['item_id'].nunique():,}")
    print(f"skills={frame['skill_id'].nunique():,}")
    print(f"correct_rate={frame['correct'].mean():.6f}")


if __name__ == "__main__":
    main()
