import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm


def load_seq_len(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return int(config["train_config"]["seq_len"]), config["train_config"]["sequence_option"]


def entropy(values):
    if len(values) == 0:
        return np.nan
    counts = pd.Series(values).value_counts(normalize=True)
    return float(-(counts * np.log2(counts)).sum())


def longest_streak(values, target):
    best = 0
    current = 0
    for value in values:
        if value == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def linear_slope(values):
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def expanding_linear_slope(values):
    y = np.asarray(values, dtype=np.float64)
    n = np.arange(1, len(y) + 1, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64)
    sum_x = np.cumsum(x)
    sum_y = np.cumsum(y)
    sum_xy = np.cumsum(x * y)
    sum_x2 = np.cumsum(x * x)
    denom = n * sum_x2 - sum_x * sum_x
    slope = np.divide(
        n * sum_xy - sum_x * sum_y,
        denom,
        out=np.zeros_like(y, dtype=np.float64),
        where=denom > 0,
    )
    return slope


def choose_sequence(user_df, seq_len, sequence_option):
    if sequence_option == "recent":
        return user_df.tail(seq_len).copy()
    if sequence_option == "early":
        return user_df.head(seq_len).copy()
    raise ValueError("sequence_option must be recent or early")


def skill_difficulty_table(df):
    train_df = df[df["split"] == "train"] if "split" in df.columns else df
    table = (
        train_df.groupby("skill_id")["correct"]
        .agg(skill_correct_rate="mean", skill_count="size")
        .reset_index()
    )
    table["skill_difficulty"] = 1.0 - table["skill_correct_rate"]
    return table


def add_distance_to_centroid(assignments, clustering_dir):
    latent_path = clustering_dir / "latent_features.npz"
    model_path = clustering_dir / "kmeans_model.pkl"
    if not latent_path.exists() or not model_path.exists():
        assignments["distance_to_centroid"] = np.nan
        return assignments

    bundle = joblib.load(model_path)
    kmeans = bundle["kmeans"] if isinstance(bundle, dict) else bundle
    scaler = bundle.get("scaler") if isinstance(bundle, dict) else None
    latent = np.load(latent_path, allow_pickle=True)
    features = latent["features"]
    cluster_features = scaler.transform(features) if scaler is not None else features
    labels = assignments["cluster"].to_numpy()
    centers = kmeans.cluster_centers_[labels]
    assignments = assignments.copy()
    assignments["distance_to_centroid"] = np.linalg.norm(cluster_features - centers, axis=1)
    return assignments


def is_timestep_assignments(assignments):
    return {"user_id", "step_index", "cluster"}.issubset(assignments.columns)


def prefix_entropy(values):
    counts = {}
    out = []
    for value in values:
        counts[value] = counts.get(value, 0) + 1
        total = sum(counts.values())
        probs = np.asarray(list(counts.values()), dtype=np.float64) / float(total)
        out.append(float(-(probs * np.log2(probs)).sum()))
    return out


def prefix_unique_counts(values):
    seen = set()
    out = []
    for value in values:
        seen.add(value)
        out.append(len(seen))
    return out


def prefix_duplicate_ratio(values):
    seen = set()
    dup_count = 0
    out = []
    for idx, value in enumerate(values, start=1):
        if value in seen:
            dup_count += 1
        seen.add(value)
        out.append(dup_count / float(idx))
    return out


def prefix_median(values):
    out = []
    prefix = []
    for value in values:
        if not pd.isna(value):
            prefix.append(float(value))
        out.append(float(np.median(prefix)) if prefix else np.nan)
    return out


def prefix_p95(values):
    out = []
    prefix = []
    for value in values:
        if not pd.isna(value):
            prefix.append(float(value))
        out.append(float(np.percentile(prefix, 95)) if prefix else np.nan)
    return out


def build_selected_step_table(df, seq_len, sequence_option):
    test_df = df[df["split"] == "test"].copy() if "split" in df.columns else df.copy()
    test_df["_row_order"] = np.arange(len(test_df))
    rows = []
    user_groups = list(test_df.groupby("user_id", sort=True))
    for user_id, user_df in tqdm(user_groups, desc="Selecting per-user sequences", unit="user"):
        seq = choose_sequence(user_df.sort_values("_row_order"), seq_len, sequence_option).copy()
        if seq.empty:
            continue
        seq["step_index"] = np.arange(len(seq))
        seq["user_id"] = user_id
        rows.append(seq)
    if not rows:
        return pd.DataFrame(columns=["user_id", "step_index"])
    return pd.concat(rows, ignore_index=True)


def build_timestep_features(
    df,
    assignments,
    skill_stats,
    seq_len,
    sequence_option,
    timestamp_scale=1000.0,
    timestamp_unit="ms",
):
    skill_stats = skill_stats.set_index("skill_id")
    difficulty = skill_stats["skill_difficulty"]
    hard_threshold = float(difficulty.quantile(0.75))
    easy_threshold = float(difficulty.quantile(0.25))

    raw_steps = build_selected_step_table(df, seq_len, sequence_option)
    raw_cols = [
        col
        for col in [
            "user_id",
            "step_index",
            "timestamp",
            "item_id",
            "skill_id",
            "source_item_id",
            "source_skill_id",
        ]
        if col in raw_steps.columns
    ]
    features = assignments.copy()
    if raw_cols:
        raw_steps = raw_steps[raw_cols].rename(
            columns={
                "item_id": "raw_item_id",
                "skill_id": "raw_skill_id",
                "source_item_id": "raw_source_item_id",
                "source_skill_id": "raw_source_skill_id",
            }
        )
        features = features.merge(raw_steps, on=["user_id", "step_index"], how="left")

    if "skill_id" not in features.columns and "raw_skill_id" in features.columns:
        features["skill_id"] = features["raw_skill_id"]
    if "item_id" not in features.columns and "raw_item_id" in features.columns:
        features["item_id"] = features["raw_item_id"]

    features["correct"] = pd.to_numeric(features["correct"], errors="coerce")
    features["skill_difficulty"] = pd.to_numeric(features["skill_id"], errors="coerce").map(difficulty)
    global_difficulty = float(difficulty.mean()) if len(difficulty) else 0.0
    features["skill_difficulty"] = features["skill_difficulty"].fillna(global_difficulty)
    features["is_hard_skill"] = (features["skill_difficulty"] >= hard_threshold).astype(float)
    features["is_easy_skill"] = (features["skill_difficulty"] <= easy_threshold).astype(float)

    rows = []
    user_groups = list(features.sort_values(["user_id", "step_index"]).groupby("user_id", sort=True))
    for user_id, user_df in tqdm(user_groups, desc="Building timestep pedagogy features", unit="user"):
        user_df = user_df.copy()
        correct = user_df["correct"].astype(float)
        user_df["num_interactions"] = np.arange(1, len(user_df) + 1)
        user_df["mean_correct"] = correct.expanding().mean().to_numpy()
        user_df["previous_correct_rate"] = correct.shift(1).expanding().mean().to_numpy()
        user_df["previous_correct_rate"] = user_df["previous_correct_rate"].fillna(0.5)
        user_df["rolling_5_correct_rate"] = correct.rolling(5, min_periods=1).mean().to_numpy()
        user_df["correct_slope"] = expanding_linear_slope(correct.to_numpy())
        user_df["correct_std"] = correct.expanding().std().fillna(0.0).to_numpy()
        user_df["first_half_correct_rate"] = user_df["previous_correct_rate"]
        user_df["second_half_correct_rate"] = user_df["mean_correct"]
        user_df["correct_rate_delta"] = user_df["mean_correct"] - user_df["previous_correct_rate"]
        user_df["longest_correct_streak"] = [
            longest_streak(correct.iloc[: idx + 1].to_numpy(), 1) for idx in range(len(user_df))
        ]
        user_df["longest_incorrect_streak"] = [
            longest_streak(correct.iloc[: idx + 1].to_numpy(), 0) for idx in range(len(user_df))
        ]
        user_df["avg_skill_difficulty"] = user_df["skill_difficulty"].expanding().mean().to_numpy()
        user_df["hard_skill_ratio"] = user_df["is_hard_skill"].expanding().mean().to_numpy()
        user_df["easy_skill_ratio"] = user_df["is_easy_skill"].expanding().mean().to_numpy()

        skill_values = user_df["skill_id"].astype(str).to_list()
        item_values = user_df["item_id"].astype(str).to_list()
        user_df["unique_skills"] = prefix_unique_counts(skill_values)
        user_df["unique_items"] = prefix_unique_counts(item_values)
        user_df["skill_entropy"] = prefix_entropy(skill_values)
        user_df["item_entropy"] = prefix_entropy(item_values)
        user_df["repeat_skill_ratio"] = prefix_duplicate_ratio(skill_values)
        user_df["repeat_question_ratio"] = prefix_duplicate_ratio(item_values)

        if "timestamp" in user_df.columns:
            timestamps = pd.to_numeric(user_df["timestamp"], errors="coerce")
            deltas = timestamps.diff() / timestamp_scale
            positive_deltas = deltas.where(deltas > 0)
            user_df["median_time_gap_sec"] = prefix_median(positive_deltas.to_list())
            user_df["p95_time_gap_sec"] = prefix_p95(positive_deltas.to_list())
            user_df["same_timestamp_ratio"] = (deltas == 0).expanding().mean().fillna(0.0).to_numpy()
            if timestamp_unit is None:
                user_df["long_gap_ratio"] = np.nan
                user_df["active_days"] = np.nan
                user_df["session_count_30min"] = np.nan
            else:
                user_df["long_gap_ratio"] = (positive_deltas > 7 * 24 * 3600).astype(float).expanding().mean().fillna(0.0).to_numpy()
                dates = pd.to_datetime(timestamps, unit=timestamp_unit, errors="coerce").dt.date
                user_df["active_days"] = [pd.Series(dates.iloc[: idx + 1]).nunique() for idx in range(len(user_df))]
                user_df["session_count_30min"] = 1 + (positive_deltas > 30 * 60).astype(int).cumsum()
        else:
            user_df["median_time_gap_sec"] = np.nan
            user_df["p95_time_gap_sec"] = np.nan
            user_df["same_timestamp_ratio"] = np.nan
            user_df["long_gap_ratio"] = np.nan
            user_df["active_days"] = np.nan
            user_df["session_count_30min"] = np.nan

        user_df["top_5_skills"] = user_df["skill_id"].astype(str) + ":1"
        user_df["top_5_items"] = user_df["item_id"].astype(str) + ":1"
        rows.append(user_df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_student_features(
    df,
    assignments,
    skill_stats,
    seq_len,
    sequence_option,
    timestamp_scale=1000.0,
    timestamp_unit="ms",
):
    skill_stats = skill_stats.set_index("skill_id")
    difficulty = skill_stats["skill_difficulty"]
    hard_threshold = float(difficulty.quantile(0.75))
    easy_threshold = float(difficulty.quantile(0.25))

    rows = []
    test_df = df[df["split"] == "test"].copy() if "split" in df.columns else df.copy()
    test_df["_row_order"] = np.arange(len(test_df))

    assignment_by_user = assignments.set_index("user_id")
    user_groups = list(test_df.groupby("user_id", sort=True))
    for user_id, user_df in tqdm(user_groups, desc="Building student pedagogy features", unit="user"):
        if user_id not in assignment_by_user.index:
            continue
        seq = choose_sequence(user_df.sort_values("_row_order"), seq_len, sequence_option)
        if seq.empty:
            continue

        correct = seq["correct"].to_numpy(dtype=np.float64)
        split_at = max(1, len(correct) // 2)
        first_half = correct[:split_at]
        second_half = correct[split_at:] if split_at < len(correct) else correct[split_at - 1 :]

        timestamps = seq["timestamp"].to_numpy(dtype=np.float64)
        deltas = (
            np.diff(timestamps) / timestamp_scale
            if len(timestamps) > 1
            else np.array([])
        )
        positive_deltas = deltas[deltas > 0]
        same_timestamp_ratio = float(np.mean(deltas == 0)) if len(deltas) else np.nan
        if timestamp_unit is None:
            long_gap_ratio = np.nan
            session_count = np.nan
            active_days = np.nan
        else:
            long_gap_ratio = float(np.mean(positive_deltas > 7 * 24 * 3600)) if len(positive_deltas) else np.nan
            session_count = int(1 + np.sum(positive_deltas > 30 * 60)) if len(seq) else 0
            active_days = int(
                pd.to_datetime(
                    seq["timestamp"], unit=timestamp_unit, errors="coerce"
                ).dt.date.nunique()
            )

        skill_ids = seq["skill_id"].to_numpy()
        item_ids = seq["item_id"].to_numpy()
        source_skills = seq["source_skill_id"].astype(str).to_numpy() if "source_skill_id" in seq.columns else skill_ids.astype(str)
        source_items = seq["source_item_id"].astype(str).to_numpy() if "source_item_id" in seq.columns else item_ids.astype(str)
        seq_skill_diff = skill_ids
        diff_values = pd.Series(seq_skill_diff).map(difficulty).to_numpy(dtype=np.float64)

        top_skills = pd.Series(source_skills).value_counts().head(5)
        top_items = pd.Series(source_items).value_counts().head(5)
        assign_row = assignment_by_user.loc[user_id]

        rows.append(
            {
                "user_id": user_id,
                "sequence_index": int(assign_row["sequence_index"]),
                "cluster": int(assign_row["cluster"]),
                "distance_to_centroid": float(assign_row.get("distance_to_centroid", np.nan)),
                "num_interactions": int(len(seq)),
                "mean_correct": float(np.mean(correct)),
                "first_half_correct_rate": float(np.mean(first_half)),
                "second_half_correct_rate": float(np.mean(second_half)),
                "correct_rate_delta": float(np.mean(second_half) - np.mean(first_half)),
                "correct_slope": linear_slope(correct),
                "correct_std": float(np.std(correct)),
                "longest_correct_streak": int(longest_streak(correct, 1)),
                "longest_incorrect_streak": int(longest_streak(correct, 0)),
                "unique_skills": int(pd.Series(skill_ids).nunique()),
                "unique_items": int(pd.Series(item_ids).nunique()),
                "skill_entropy": entropy(skill_ids),
                "item_entropy": entropy(item_ids),
                "repeat_skill_ratio": float(pd.Series(skill_ids).duplicated().mean()),
                "repeat_question_ratio": float(pd.Series(item_ids).duplicated().mean()),
                "avg_skill_difficulty": float(np.nanmean(diff_values)),
                "hard_skill_ratio": float(np.nanmean(diff_values >= hard_threshold)),
                "easy_skill_ratio": float(np.nanmean(diff_values <= easy_threshold)),
                "median_time_gap_sec": float(np.median(positive_deltas)) if len(positive_deltas) else np.nan,
                "p95_time_gap_sec": float(np.percentile(positive_deltas, 95)) if len(positive_deltas) else np.nan,
                "same_timestamp_ratio": same_timestamp_ratio,
                "long_gap_ratio": long_gap_ratio,
                "active_days": active_days,
                "session_count_30min": session_count,
                "top_5_skills": ";".join(f"{idx}:{count}" for idx, count in top_skills.items()),
                "top_5_items": ";".join(f"{idx}:{count}" for idx, count in top_items.items()),
            }
        )

    return pd.DataFrame(rows)


def weighted_top_values(df, value_col, cluster_col="cluster", topn=10):
    rows = []
    for cluster, cluster_df in df.groupby(cluster_col):
        counter = {}
        for cell in cluster_df[value_col].dropna():
            for part in str(cell).split(";"):
                if not part or ":" not in part:
                    continue
                key, count = part.rsplit(":", 1)
                counter[key] = counter.get(key, 0) + int(count)
        top = sorted(counter.items(), key=lambda item: item[1], reverse=True)[:topn]
        rows.append({cluster_col: cluster, f"{value_col}_cluster_top{topn}": ";".join(f"{k}:{v}" for k, v in top)})
    return pd.DataFrame(rows)


def suggest_label(row):
    if row["avg_mean_correct"] >= 0.85 and row["avg_correct_rate_delta"] >= -0.03:
        return "High mastery / stable"
    if row["avg_mean_correct"] < 0.65 and row["avg_avg_skill_difficulty"] >= 0.2:
        return "Needs support on challenging skills"
    if row["avg_mean_correct"] < 0.65:
        return "Low performance / needs support"
    if row["avg_correct_rate_delta"] >= 0.08:
        return "Improving learners"
    if row["avg_correct_std"] >= 0.42:
        return "Volatile performance"
    if row["avg_repeat_skill_ratio"] >= 0.45:
        return "Practice-heavy / consolidation"
    return "Moderate performance / developing"


def cluster_summary(student_df):
    agg_map = {
        "n_users": ("user_id", "nunique"),
        "n_steps": ("cluster", "size"),
        "avg_mean_correct": ("mean_correct", "mean"),
        "avg_current_correct": ("correct" if "correct" in student_df.columns else "mean_correct", "mean"),
        "median_mean_correct": ("mean_correct", "median"),
        "avg_first_half_correct": ("first_half_correct_rate", "mean"),
        "avg_second_half_correct": ("second_half_correct_rate", "mean"),
        "avg_correct_rate_delta": ("correct_rate_delta", "mean"),
        "avg_correct_slope": ("correct_slope", "mean"),
        "avg_correct_std": ("correct_std", "mean"),
        "avg_longest_correct_streak": ("longest_correct_streak", "mean"),
        "avg_longest_incorrect_streak": ("longest_incorrect_streak", "mean"),
        "avg_unique_skills": ("unique_skills", "mean"),
        "avg_skill_entropy": ("skill_entropy", "mean"),
        "avg_repeat_skill_ratio": ("repeat_skill_ratio", "mean"),
        "avg_repeat_question_ratio": ("repeat_question_ratio", "mean"),
        "avg_avg_skill_difficulty": ("avg_skill_difficulty", "mean"),
        "avg_hard_skill_ratio": ("hard_skill_ratio", "mean"),
        "avg_easy_skill_ratio": ("easy_skill_ratio", "mean"),
        "median_time_gap_sec": ("median_time_gap_sec", "median"),
        "avg_same_timestamp_ratio": ("same_timestamp_ratio", "mean"),
        "avg_long_gap_ratio": ("long_gap_ratio", "mean"),
        "avg_active_days": ("active_days", "mean"),
        "avg_session_count_30min": ("session_count_30min", "mean"),
        "avg_distance_to_centroid": ("distance_to_centroid", "mean"),
    }
    summary = (
        student_df.groupby("cluster")
        .agg(**agg_map)
        .reset_index()
    )
    summary = summary.merge(weighted_top_values(student_df, "top_5_skills"), on="cluster", how="left")
    summary = summary.merge(weighted_top_values(student_df, "top_5_items"), on="cluster", how="left")
    summary["pedagogical_label_suggestion"] = summary.apply(suggest_label, axis=1)
    return summary


def write_markdown(summary, reps, path):
    lines = ["# Cluster Pedagogy Analysis", ""]
    for _, row in summary.sort_values("cluster").iterrows():
        cluster = int(row["cluster"])
        count_label = "Steps" if "n_steps" in summary.columns else "Students"
        count_value = int(row["n_steps"] if "n_steps" in summary.columns else row.get("n_students", row.get("n_users", 0)))
        lines.extend(
            [
                f"## Cluster {cluster}: {row['pedagogical_label_suggestion']}",
                "",
                f"- {count_label}: {count_value}",
                f"- Users: {int(row.get('n_users', row.get('n_students', 0)))}",
                f"- Prefix mean correct rate: {row['avg_mean_correct']:.3f}",
                f"- Current-step correct rate: {row.get('avg_current_correct', np.nan):.3f}",
                f"- First half -> second half: {row['avg_first_half_correct']:.3f} -> {row['avg_second_half_correct']:.3f} (delta {row['avg_correct_rate_delta']:+.3f})",
                f"- Avg skill difficulty: {row['avg_avg_skill_difficulty']:.3f}; hard-skill ratio: {row['avg_hard_skill_ratio']:.3f}",
                f"- Avg unique skills: {row['avg_unique_skills']:.1f}; skill entropy: {row['avg_skill_entropy']:.2f}",
                f"- Repeat skill/question ratio: {row['avg_repeat_skill_ratio']:.3f} / {row['avg_repeat_question_ratio']:.3f}",
                f"- Median time gap: {row['median_time_gap_sec']:.1f}s; active days: {row['avg_active_days']:.1f}",
                f"- Top skills: {row.get('top_5_skills_cluster_top10', '')}",
                "",
                "Representative students near centroid:",
            ]
        )
        rep_cols = [
            "user_id",
            "mean_correct",
            "first_half_correct_rate",
            "second_half_correct_rate",
            "correct_rate_delta",
            "unique_skills",
            "avg_skill_difficulty",
            "distance_to_centroid",
        ]
        cluster_reps = reps[reps["cluster"] == cluster]
        lines.append(markdown_table(cluster_reps[[col for col in rep_cols if col in cluster_reps.columns]]))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df):
    if df.empty:
        return ""
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
    columns = list(view.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).replace("\n", " ") for col in columns) + " |"
        for _, row in view.iterrows()
    ]
    return "\n".join([header, separator] + rows)


def main():
    parser = argparse.ArgumentParser(description="Compute pedagogical features and cluster-level interpretation.")
    parser.add_argument("--preprocessed_csv", default="dataset/XES3G5M/preprocessed_df.csv")
    parser.add_argument("--clustering_dir", default="outputs/latent_clustering/XES3G5M/official")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--representatives_csv", default=None)
    parser.add_argument(
        "--timestamp_unit",
        choices=["auto", "seconds", "milliseconds", "ordinal"],
        default="auto",
        help=(
            "Timestamp unit used for temporal post-hoc features. In auto mode, "
            "epoch-scale values are treated as milliseconds and smaller relative "
            "values as seconds. Use ordinal when the column only preserves event "
            "order and is not elapsed time."
        ),
    )
    args = parser.parse_args()

    clustering_dir = Path(args.clustering_dir)
    seq_len, sequence_option = load_seq_len(args.config)
    print(f"[cluster_pedagogy] clustering_dir={clustering_dir}")
    print(f"[cluster_pedagogy] seq_len={seq_len}, sequence_option={sequence_option}")

    print(f"[cluster_pedagogy] loading raw data: {args.preprocessed_csv}")
    df = pd.read_csv(
        args.preprocessed_csv,
        sep="\t",
        usecols=[
            "user_id",
            "item_id",
            "timestamp",
            "correct",
            "skill_id",
            "split",
            "source_item_id",
            "source_skill_id",
        ],
    )
    if args.timestamp_unit == "auto":
        timestamp_unit = (
            "milliseconds"
            if pd.to_numeric(df["timestamp"], errors="coerce").abs().max() > 1e11
            else "seconds"
        )
    else:
        timestamp_unit = args.timestamp_unit
    timestamp_scale = 1000.0 if timestamp_unit == "milliseconds" else 1.0
    if timestamp_unit == "ordinal":
        pandas_timestamp_unit = None
    else:
        pandas_timestamp_unit = "ms" if timestamp_unit == "milliseconds" else "s"
    print(f"[cluster_pedagogy] timestamp_unit={timestamp_unit}")
    print(f"[cluster_pedagogy] loading assignments: {clustering_dir / 'cluster_assignments_k3.csv'}")
    assignments = pd.read_csv(clustering_dir / "cluster_assignments_k3.csv")
    print(f"[cluster_pedagogy] assignments rows={len(assignments):,}, timestep_level={is_timestep_assignments(assignments)}")
    print("[cluster_pedagogy] computing distance to centroid")
    assignments = add_distance_to_centroid(assignments, clustering_dir)

    print("[cluster_pedagogy] computing skill difficulty table")
    skill_stats = skill_difficulty_table(df)
    if is_timestep_assignments(assignments):
        student_features = build_timestep_features(
            df,
            assignments,
            skill_stats,
            seq_len,
            sequence_option,
            timestamp_scale,
            pandas_timestamp_unit,
        )
    else:
        student_features = build_student_features(
            df,
            assignments,
            skill_stats,
            seq_len,
            sequence_option,
            timestamp_scale,
            pandas_timestamp_unit,
        )
    print(f"[cluster_pedagogy] feature rows={len(student_features):,}")
    print("[cluster_pedagogy] summarizing clusters")
    summary = cluster_summary(student_features)

    student_path = clustering_dir / ("timestep_pedagogy_features.csv" if is_timestep_assignments(assignments) else "student_pedagogy_features.csv")
    summary_path = clustering_dir / "cluster_pedagogy_summary.csv"
    report_path = clustering_dir / "cluster_pedagogy_report.md"
    student_features.to_csv(student_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"[cluster_pedagogy] saved features: {student_path}")
    print(f"[cluster_pedagogy] saved summary: {summary_path}")

    reps_path = Path(args.representatives_csv) if args.representatives_csv else clustering_dir / "representatives_random_near_centroid.csv"
    print(f"[cluster_pedagogy] enriching representatives: {reps_path if reps_path.exists() else 'fallback nearest rows'}")
    if reps_path.exists():
        reps = pd.read_csv(reps_path)
        merge_keys = [key for key in ["feature_index", "user_id", "step_index", "sequence_index"] if key in reps.columns and key in student_features.columns]
        reps = reps.merge(
            student_features.drop(columns=[col for col in ["cluster", "distance_to_centroid"] if col in student_features.columns]),
            on=merge_keys,
            how="left",
        )
    else:
        reps = (
            student_features.sort_values(["cluster", "distance_to_centroid"])
            .groupby("cluster", as_index=False)
            .head(5)
        )
    reps = reps.sort_values(["cluster", "distance_to_centroid"])
    enriched_reps_path = clustering_dir / "representatives_pedagogy_enriched.csv"
    reps.to_csv(enriched_reps_path, index=False)
    write_markdown(summary, reps, report_path)

    display_cols = [
        "cluster",
        "n_steps",
        "n_users",
        "avg_mean_correct",
        "avg_current_correct",
        "avg_correct_rate_delta",
        "avg_avg_skill_difficulty",
        "avg_unique_skills",
        "avg_repeat_skill_ratio",
        "median_time_gap_sec",
        "pedagogical_label_suggestion",
    ]
    print(summary[display_cols].to_string(index=False))
    print(f"Saved student features to: {student_path}")
    print(f"Saved cluster summary to: {summary_path}")
    print(f"Saved enriched representatives to: {enriched_reps_path}")
    print(f"Saved interpretation report to: {report_path}")


if __name__ == "__main__":
    main()
