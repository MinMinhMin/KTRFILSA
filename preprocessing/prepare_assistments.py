#!/usr/bin/env python3
"""Prepare the ASSISTments 2009/2017 files used by the paper experiments."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


SPECS = {
    "ASSISTMENT2009": {
        "raw_name": "skill_builder_data_corrected.csv",
        "encoding": "ISO-8859-1",
        "user": "user_id",
        "item": "problem_id",
        "skill": "skill_id",
        "time": "order_id",
        "tie_breakers": ["problem_id"],
        "fallback_dirs": ["ASSISTMENT 2009"],
    },
    "ASSISTMENT2017": {
        "raw_name": "anonymized_full_release_competition_dataset.csv",
        "encoding": "utf-8",
        "user": "studentId",
        "item": "problemId",
        "skill": "skill",
        "time": "startTime",
        "tie_breakers": ["action_num", "problemId"],
        "fallback_dirs": [],
    },
}


def sorted_values(values):
    values = pd.Series(values).astype(str).unique().tolist()
    return sorted(values)


def split_users(users, seed):
    shuffled = np.asarray(users).copy()
    np.random.RandomState(seed).shuffle(shuffled)
    n_train = int(0.8 * len(shuffled))
    n_valid = int(0.1 * len(shuffled))
    mapping = {value: "train" for value in shuffled[:n_train]}
    mapping.update({value: "valid" for value in shuffled[n_train : n_train + n_valid]})
    mapping.update({value: "test" for value in shuffled[n_train + n_valid :]})
    return mapping


def resolve_raw_path(dataset_root, dataset, raw_path):
    if raw_path is not None:
        return raw_path

    spec = SPECS[dataset]
    candidates = [dataset_root / dataset / spec["raw_name"]]
    candidates.extend(dataset_root / dirname / spec["raw_name"] for dirname in spec["fallback_dirs"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing raw file. Checked:\n" + "\n".join(f"- {candidate}" for candidate in candidates)
    )


def optional_columns(dataset):
    if dataset == "ASSISTMENT2009":
        return {
            "source_skill_name": "skill_name",
            "source_order_id": "order_id",
            "source_original": "original",
            "source_attempt_count": "attempt_count",
            "source_hint_count": "hint_count",
            "source_ms_first_response": "ms_first_response",
            "source_template_id": "template_id",
        }
    return {
        "source_start_time": "startTime",
        "source_action_num": "action_num",
        "source_original": "original",
        "source_scaffold": "scaffold",
    }


def jsonable_mapping(mapping):
    return {str(key): int(value) for key, value in mapping.items()}


def prepare(
    dataset,
    raw_path,
    output_dir,
    seed,
    min_interactions,
    all_actions=False,
    keep_nan_skills=False,
    drop_duplicate_order=False,
    drop_duplicate_user_time=False,
):
    spec = SPECS[dataset]
    frame = pd.read_csv(raw_path, encoding=spec["encoding"], low_memory=False)
    required = {spec["user"], spec["item"], spec["skill"], spec["time"], "correct", "original"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Raw file is missing columns: {sorted(missing)}")

    frame = frame[frame["correct"].isin([0, 1])].copy()
    frame = frame[frame[spec["user"]].notna()]
    frame = frame[frame[spec["item"]].notna()]
    frame = frame[frame[spec["time"]].notna()]
    if not all_actions:
        frame = frame[frame["original"] == 1]
    if keep_nan_skills:
        frame[spec["skill"]] = frame[spec["skill"]].fillna("__MISSING__")
        if "skill_name" in frame.columns:
            frame["skill_name"] = frame["skill_name"].fillna("__MISSING__")
    else:
        frame = frame[frame[spec["skill"]].notna()]

    frame["correct"] = frame["correct"].astype(int)
    frame[spec["time"]] = pd.to_numeric(frame[spec["time"]], errors="coerce")
    frame = frame[frame[spec["time"]].notna()].copy()
    if "action_num" in frame.columns:
        frame["action_num"] = pd.to_numeric(frame["action_num"], errors="coerce").fillna(0)
    for column in ("attempt_count", "hint_count", "ms_first_response"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    source_sort_columns = [spec["user"], spec["time"]]
    source_sort_columns.extend(column for column in spec["tie_breakers"] if column in frame.columns)
    frame = frame.sort_values(source_sort_columns, kind="stable").reset_index(drop=True)

    if dataset == "ASSISTMENT2009" and drop_duplicate_order:
        frame = frame.drop_duplicates([spec["user"], spec["time"]], keep="first")
    if dataset == "ASSISTMENT2017" and drop_duplicate_user_time:
        frame = frame.drop_duplicates([spec["user"], spec["time"]], keep="first")

    frame = frame.groupby(spec["user"], sort=False).filter(
        lambda group: len(group) >= min_interactions
    )
    if frame.empty:
        raise ValueError("No rows remain after filtering. Relax filters and try again.")

    users = sorted_values(frame[spec["user"]])
    items = sorted_values(frame[spec["item"]])
    skills = sorted_values(frame[spec["skill"]])
    user_map = {value: index for index, value in enumerate(users)}
    item_map = {value: index for index, value in enumerate(items)}
    skill_map = {value: index for index, value in enumerate(skills)}
    split_map = split_users(users, seed)
    time_values = pd.to_numeric(frame[spec["time"]], errors="raise")

    result = pd.DataFrame(
        {
            "user_id": frame[spec["user"]].astype(str).map(user_map).astype(int),
            "item_id": frame[spec["item"]].astype(str).map(item_map).astype(int),
            "timestamp": (time_values - time_values.min()).astype(np.int64),
            "correct": frame["correct"].astype(int),
            "skill_id": frame[spec["skill"]].astype(str).map(skill_map).astype(int),
            "split": frame[spec["user"]].astype(str).map(split_map),
            "source_user_id": frame[spec["user"]],
            "source_item_id": frame[spec["item"]],
            "source_skill_id": frame[spec["skill"]].astype(str),
        }
    )
    for target, source in optional_columns(dataset).items():
        if source in frame.columns:
            result[target] = frame[source].to_numpy()

    if dataset == "ASSISTMENT2009":
        sort_columns = ["user_id", "timestamp", "item_id"]
    else:
        sort_columns = ["user_id", "timestamp"]
        if "source_action_num" in result.columns:
            sort_columns.append("source_action_num")
        sort_columns.append("item_id")
    result = result.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "preprocessed_df.csv", sep="\t", index=False)
    rows = result["item_id"].to_numpy()
    cols = result["skill_id"].to_numpy()
    q_matrix = sparse.csr_matrix(
        (np.ones(len(result), dtype=np.float32), (rows, cols)),
        shape=(len(items), len(skills)),
    )
    q_matrix.data[:] = 1.0
    sparse.save_npz(output_dir / "q_mat.npz", q_matrix)
    with (output_dir / "question_skill_rel.pkl").open("wb") as handle:
        pickle.dump(q_matrix, handle)

    summary = {
        "dataset": dataset,
        "raw_file": str(raw_path),
        "all_actions": bool(all_actions),
        "original_only": not bool(all_actions),
        "keep_nan_skills": bool(keep_nan_skills),
        "drop_duplicate_order": bool(drop_duplicate_order),
        "drop_duplicate_user_time": bool(drop_duplicate_user_time),
        "seed": seed,
        "min_user_interactions": min_interactions,
        "n_interactions": int(len(result)),
        "n_users": int(result["user_id"].nunique()),
        "n_items": int(result["item_id"].nunique()),
        "n_skills": int(result["skill_id"].nunique()),
        "correct_rate": float(result["correct"].mean()),
        "split_interactions": result["split"].value_counts().sort_index().astype(int).to_dict(),
        "split_users": result.groupby("split")["user_id"].nunique().sort_index().astype(int).to_dict(),
        "split_correct_rate": result.groupby("split")["correct"].mean().sort_index().astype(float).to_dict(),
    }
    (output_dir / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    maps = {
        "dataset": dataset,
        "raw_file": str(raw_path),
        "user_map": jsonable_mapping(user_map),
        "item_map": jsonable_mapping(item_map),
        "skill_map": jsonable_mapping(skill_map),
    }
    (output_dir / f"{dataset.lower()}_id_maps.json").write_text(
        json.dumps(maps, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--seed", type=int, default=12405)
    parser.add_argument("--min-interactions", type=int, default=5)
    parser.add_argument(
        "--all-actions",
        action="store_true",
        help="Keep non-original/scaffold-like rows too. Default keeps only original == 1 rows.",
    )
    parser.add_argument(
        "--keep-nan-skills",
        action="store_true",
        help="Keep rows without skills by assigning them to a synthetic __MISSING__ skill.",
    )
    parser.add_argument(
        "--drop-duplicate-order",
        action="store_true",
        help="ASSISTMENT2009 only: drop duplicated rows with the same source user and order_id.",
    )
    parser.add_argument(
        "--drop-duplicate-user-time",
        action="store_true",
        help="ASSISTMENT2017 only: drop duplicated rows with the same source student and startTime.",
    )
    args = parser.parse_args()
    raw_path = resolve_raw_path(args.dataset_root, args.dataset, args.raw)
    output_dir = args.output_dir or args.dataset_root / args.dataset
    prepare(
        args.dataset,
        raw_path,
        output_dir,
        args.seed,
        args.min_interactions,
        all_actions=args.all_actions,
        keep_nan_skills=args.keep_nan_skills,
        drop_duplicate_order=args.drop_duplicate_order,
        drop_duplicate_user_time=args.drop_duplicate_user_time,
    )


if __name__ == "__main__":
    main()
