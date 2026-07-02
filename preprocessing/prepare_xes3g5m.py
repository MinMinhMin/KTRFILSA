import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm


def parse_list(value, cast=str):
    if value is None or value == "":
        return []
    out = []
    for item in str(value).split(","):
        item = item.strip()
        if item == "":
            continue
        out.append(int(float(item)) if cast is int else item)
    return out


def sequence_files(base_dir, level):
    base_dir = Path(base_dir)
    if level == "question":
        return [
            (base_dir / "question_level" / "train_valid_sequences_quelevel.csv", "train_valid"),
            (base_dir / "question_level" / "test_quelevel.csv", "test"),
        ]
    if level == "kc":
        return [
            (base_dir / "kc_level" / "train_valid_sequences.csv", "train_valid"),
            (base_dir / "kc_level" / "test.csv", "test"),
        ]
    raise ValueError("level must be 'question' or 'kc'")


def iter_valid_tokens(path, source_split):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        has_selectmasks = "selectmasks" in (reader.fieldnames or [])
        for row in reader:
            uid = row["uid"]
            questions = parse_list(row.get("questions"), int)
            concepts = parse_list(row.get("concepts"), str)
            responses = parse_list(row.get("responses"), int)
            timestamps = parse_list(row.get("timestamps"), int)
            selectmasks = parse_list(row.get("selectmasks"), int) if has_selectmasks else []
            raw_len = max(len(questions), len(concepts), len(responses), len(timestamps), len(selectmasks))
            for idx in range(raw_len):
                q = questions[idx] if idx < len(questions) else -1
                c = concepts[idx] if idx < len(concepts) else "-1"
                r = responses[idx] if idx < len(responses) else -1
                t = timestamps[idx] if idx < len(timestamps) else -1
                m = selectmasks[idx] if idx < len(selectmasks) else 1
                if m != 1 or q < 0 or c == "-1" or r not in (0, 1):
                    continue
                yield {
                    "source_user_id": uid,
                    "source_item_id": q,
                    "source_skill_id": c,
                    "timestamp": t,
                    "correct": r,
                    "source_split": source_split,
                }


def collect_vocab(files):
    users, items, skills, train_valid_users = set(), set(), set(), set()
    counts = {"train_valid": 0, "test": 0}
    for path, source_split in files:
        for token in tqdm(iter_valid_tokens(path, source_split), desc=f"scan {source_split}"):
            users.add(token["source_user_id"])
            items.add(token["source_item_id"])
            skills.add(token["source_skill_id"])
            if source_split == "train_valid":
                train_valid_users.add(token["source_user_id"])
            counts[source_split] += 1
    return users, items, skills, train_valid_users, counts


def sort_mixed(values):
    return sorted(values, key=lambda x: int(x) if str(x).isdigit() else str(x))


def make_valid_users(train_valid_users, valid_ratio, seed):
    users = np.array(sort_mixed(train_valid_users))
    rng = np.random.RandomState(seed)
    rng.shuffle(users)
    n_valid = max(1, int(len(users) * valid_ratio)) if len(users) > 1 else 0
    return set(users[:n_valid])


def write_preprocessed(files, output_path, user_map, item_map, skill_map, valid_users):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "user_id", "item_id", "timestamp", "correct", "skill_id",
                "split", "source_split", "source_user_id", "source_item_id", "source_skill_id",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for path, source_split in files:
            for token in tqdm(iter_valid_tokens(path, source_split), desc=f"write {source_split}"):
                split = "test" if source_split == "test" else ("valid" if token["source_user_id"] in valid_users else "train")
                writer.writerow(
                    {
                        "user_id": user_map[token["source_user_id"]],
                        "item_id": item_map[token["source_item_id"]],
                        "timestamp": token["timestamp"],
                        "correct": token["correct"],
                        "skill_id": skill_map[token["source_skill_id"]],
                        "split": split,
                        "source_split": source_split,
                        "source_user_id": token["source_user_id"],
                        "source_item_id": token["source_item_id"],
                        "source_skill_id": token["source_skill_id"],
                    }
                )


def save_q_matrix(output_dir, item_map, skill_map, preprocessed_path):
    q_mat = sparse.lil_matrix((len(item_map), len(skill_map)), dtype=np.float32)
    for chunk in pd.read_csv(preprocessed_path, sep="\t", usecols=["item_id", "skill_id"], chunksize=500000):
        q_mat[chunk["item_id"].to_numpy(), chunk["skill_id"].to_numpy()] = 1.0
    q_mat = q_mat.tocsr()
    sparse.save_npz(output_dir / "q_mat.npz", q_mat)
    with (output_dir / "question_skill_rel.pkl").open("wb") as f:
        pickle.dump(q_mat, f)


def save_mappings(output_dir, user_map, item_map, skill_map, counts, valid_users):
    with (output_dir / "xes3g5m_id_maps.json").open("w") as f:
        json.dump(
            {
                "user_map": user_map,
                "item_map": {str(k): v for k, v in item_map.items()},
                "skill_map": skill_map,
                "counts": counts,
                "valid_source_users": sort_mixed(valid_users),
            },
            f,
        )


def main():
    parser = argparse.ArgumentParser(description="Prepare XES3G5M for CL4KT using the official train/test split.")
    parser.add_argument("--base_dir", default="dataset/XES3G5M")
    parser.add_argument("--level", default="question", choices=["question", "kc"])
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=12405)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    files = sequence_files(base_dir, args.level)
    missing = [str(path) for path, _ in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing XES3G5M files: {missing}")

    users, items, skills, train_valid_users, counts = collect_vocab(files)
    valid_users = make_valid_users(train_valid_users, args.valid_ratio, args.seed)
    user_map = {uid: idx for idx, uid in enumerate(sort_mixed(users))}
    item_map = {item: idx for idx, item in enumerate(sorted(items))}
    skill_map = {skill: idx for idx, skill in enumerate(sorted(skills))}

    output_path = base_dir / "preprocessed_df.csv"
    write_preprocessed(files, output_path, user_map, item_map, skill_map, valid_users)
    save_q_matrix(base_dir, item_map, skill_map, output_path)
    save_mappings(base_dir, user_map, item_map, skill_map, counts, valid_users)

    df = pd.read_csv(output_path, sep="\t", usecols=["user_id", "item_id", "skill_id", "correct", "split"])
    print("Saved:", output_path)
    print("Level:", args.level)
    print("# Users:", df["user_id"].nunique())
    print("# Items:", df["item_id"].nunique())
    print("# Skills:", df["skill_id"].nunique())
    print("# Interactions:", len(df))
    print("Split counts:")
    print(df.groupby("split").size().to_string())
    print("User counts by split:")
    print(df.groupby("split")["user_id"].nunique().to_string())
    print("Correct rate by split:")
    print(df.groupby("split")["correct"].mean().to_string())


if __name__ == "__main__":
    main()
