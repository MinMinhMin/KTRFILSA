#!/usr/bin/env python3
"""Order-shuffling control used to reproduce Table VI of the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm.auto import tqdm

from extract_and_cluster import (
    build_test_dataframe,
    dataset_class,
    load_config,
    load_model,
    prepare_dataframe,
)


def choose_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def move_dataset_to_cpu(dataset):
    for name in ("padded_q", "padded_s", "padded_r", "attention_mask"):
        setattr(dataset, name, getattr(dataset, name).detach().cpu())
    return dataset


def load_cluster_bundle(path):
    bundle = joblib.load(path)
    if hasattr(bundle, "predict"):
        return {"kmeans": bundle, "scaler": None, "best_k": bundle.n_clusters}
    if "kmeans" not in bundle:
        raise ValueError("Cluster bundle must contain a 'kmeans' object")
    return bundle


def load_saved_latents(path):
    latent = np.load(path, allow_pickle=True)
    required = {"features", "feature_index", "user_ids", "step_indices"}
    missing = required - set(latent.files)
    if missing:
        raise ValueError(f"Latent NPZ is missing keys: {sorted(missing)}")
    feature_indices = latent["feature_index"].astype(int)
    if len(np.unique(feature_indices)) != len(feature_indices):
        raise ValueError("feature_index values in latent NPZ are not unique")
    features = latent["features"].astype(np.float32)
    return {
        int(feature_index): feature
        for feature_index, feature in zip(feature_indices, features)
    }


def cosine_similarity_rows(left, right):
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.clip(denominator, 1e-12, None)


def derangement(length, rng):
    if length <= 1:
        return np.arange(length)
    identity = np.arange(length)
    for _ in range(100):
        permutation = rng.permutation(length)
        if np.all(permutation != identity):
            return permutation
    return np.roll(identity, int(rng.randint(1, length)))


def group_assignments(assignments):
    return {
        int(sequence_index): group.sort_values("step_index").reset_index(drop=True)
        for sequence_index, group in assignments.groupby("sequence_index", sort=True)
    }


def validate_inputs(assignments, saved_features, bundle):
    required = {
        "feature_index",
        "sequence_index",
        "step_index",
        "user_id",
        "skill_id",
        "correct",
        "cluster",
    }
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Assignments CSV is missing columns: {sorted(missing)}")
    if assignments["feature_index"].duplicated().any():
        raise ValueError("Assignments contain duplicate feature_index values")

    missing_features = set(assignments["feature_index"].astype(int)) - set(saved_features)
    if missing_features:
        raise ValueError(
            f"{len(missing_features)} assignments cannot be matched to saved latents"
        )

    matrix = np.stack(
        [saved_features[index] for index in assignments["feature_index"].astype(int)]
    )
    scaler = bundle.get("scaler")
    cluster_input = scaler.transform(matrix) if scaler is not None else matrix
    predicted = bundle["kmeans"].predict(cluster_input).astype(int)
    stored = assignments["cluster"].to_numpy(dtype=int)
    match_rate = float(np.mean(predicted == stored))
    return pd.DataFrame(
        [
            {
                "n_interactions": len(assignments),
                "saved_cluster_match_rate": match_rate,
                "cluster_mismatches": int(np.sum(predicted != stored)),
            }
        ]
    )


def validate_dataset_alignment(dataset, assignments_by_sequence):
    rows = []
    for sequence_index in range(len(dataset)):
        if sequence_index not in assignments_by_sequence:
            raise ValueError(f"Missing assignments for sequence {sequence_index}")
        mask = dataset.attention_mask[sequence_index].numpy().astype(bool)
        skills = dataset.padded_s[sequence_index].numpy()[mask]
        responses = dataset.padded_r[sequence_index].numpy()[mask]
        saved = assignments_by_sequence[sequence_index]
        rows.append(
            {
                "sequence_index": sequence_index,
                "length_match": len(saved) == int(mask.sum()),
                "skill_match": len(saved) == len(skills)
                and np.array_equal(skills, saved["skill_id"].to_numpy()),
                "response_match": len(saved) == len(responses)
                and np.array_equal(responses, saved["correct"].to_numpy()),
            }
        )
    checks = pd.DataFrame(rows)
    if not checks[["length_match", "skill_match", "response_match"]].all().all():
        raise ValueError("Dataset tensors do not align with saved assignments")
    return pd.DataFrame(
        [
            {
                "n_sequences": len(checks),
                "length_match_rate": float(checks["length_match"].mean()),
                "skill_match_rate": float(checks["skill_match"].mean()),
                "response_match_rate": float(checks["response_match"].mean()),
            }
        ]
    )


def make_batch(dataset, indices, device):
    return {
        "questions": torch.stack([dataset.padded_q[i] for i in indices]).to(device),
        "skills": torch.stack([dataset.padded_s[i] for i in indices]).to(device),
        "responses": torch.stack([dataset.padded_r[i] for i in indices]).to(device),
        "attention_mask": torch.stack(
            [dataset.attention_mask[i] for i in indices]
        ).to(device),
    }


def verify_identity_inference(
    model,
    dataset,
    assignments_by_sequence,
    saved_features,
    bundle,
    batch_size,
    device,
    min_match_rate,
):
    kmeans = bundle["kmeans"]
    scaler = bundle.get("scaler")
    n_points = 0
    n_matches = 0
    cosine_sum = 0.0

    with torch.no_grad():
        for start in tqdm(
            range(0, len(dataset), batch_size),
            desc="Identity-control inference",
            unit="batch",
        ):
            indices = list(range(start, min(start + batch_size, len(dataset))))
            output = model.extract_features(
                make_batch(dataset, indices, device), pool=False
            )
            states = output["sequence_features"].detach().cpu().numpy().astype(np.float32)
            masks = output["attention_mask"].detach().cpu().numpy().astype(bool)

            for local_index, sequence_index in enumerate(indices):
                current = states[local_index][masks[local_index]]
                saved = assignments_by_sequence[sequence_index]
                original = np.stack(
                    [
                        saved_features[index]
                        for index in saved["feature_index"].to_numpy(dtype=int)
                    ]
                )
                cluster_input = scaler.transform(current) if scaler is not None else current
                predicted = kmeans.predict(cluster_input).astype(int)
                stored = saved["cluster"].to_numpy(dtype=int)
                n_points += len(stored)
                n_matches += int(np.sum(predicted == stored))
                cosine_sum += float(cosine_similarity_rows(original, current).sum())

    match_rate = n_matches / n_points
    result = pd.DataFrame(
        [
            {
                "n_interactions": n_points,
                "checkpoint_cluster_match_rate": match_rate,
                "checkpoint_latent_mean_cosine": cosine_sum / n_points,
                "cluster_mismatches": n_points - n_matches,
                "required_min_match_rate": min_match_rate,
                "passed_tolerance": match_rate >= min_match_rate,
            }
        ]
    )
    if match_rate < min_match_rate:
        raise ValueError(
            "Identity-control inference is below the required match rate:\n"
            + result.to_string(index=False)
        )
    return result


def switch_rate(labels):
    return float(np.mean(labels[1:] != labels[:-1])) if len(labels) > 1 else 0.0


def transition_counts(labels_by_user, n_clusters):
    counts = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for labels in labels_by_user.values():
        if len(labels) > 1:
            np.add.at(counts, (labels[:-1], labels[1:]), 1)
    return counts


def transition_probabilities(counts):
    row_sums = counts.sum(axis=1, keepdims=True)
    return np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts, dtype=float),
        where=row_sums > 0,
    )


def infer_one_shuffle(
    model,
    dataset,
    user_ids,
    assignments_by_sequence,
    saved_features,
    bundle,
    seed,
    batch_size,
    device,
):
    rng = np.random.RandomState(seed)
    kmeans = bundle["kmeans"]
    scaler = bundle.get("scaler")
    n_clusters = int(kmeans.n_clusters)
    confusion = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    user_rows = []
    original_all = []
    shuffled_all = []
    shuffled_labels_by_user = {}
    total_matches = 0
    total_points = 0
    total_fixed_positions = 0
    cosine_sum = 0.0
    cosine_sq_sum = 0.0

    with torch.no_grad():
        for start in tqdm(
            range(0, len(dataset), batch_size),
            desc=f"Shuffle seed {seed}",
            unit="batch",
            leave=False,
        ):
            indices = list(range(start, min(start + batch_size, len(dataset))))
            q_batch = torch.stack([dataset.padded_q[i] for i in indices]).clone()
            s_batch = torch.stack([dataset.padded_s[i] for i in indices]).clone()
            r_batch = torch.stack([dataset.padded_r[i] for i in indices]).clone()
            m_batch = torch.stack([dataset.attention_mask[i] for i in indices])
            permutations = {}

            for local_index, sequence_index in enumerate(indices):
                positions = torch.where(m_batch[local_index].bool())[0]
                permutation = derangement(len(positions), rng)
                permutations[sequence_index] = permutation
                permutation_tensor = torch.as_tensor(permutation, dtype=torch.long)
                q_values = q_batch[local_index, positions].clone()
                s_values = s_batch[local_index, positions].clone()
                r_values = r_batch[local_index, positions].clone()
                q_batch[local_index, positions] = q_values[permutation_tensor]
                s_batch[local_index, positions] = s_values[permutation_tensor]
                r_batch[local_index, positions] = r_values[permutation_tensor]
                total_fixed_positions += int(
                    np.sum(permutation == np.arange(len(permutation)))
                )

            batch = {
                "questions": q_batch.to(device),
                "skills": s_batch.to(device),
                "responses": r_batch.to(device),
                "attention_mask": m_batch.to(device),
            }
            output = model.extract_features(batch, pool=False)
            states = output["sequence_features"].detach().cpu().numpy().astype(np.float32)
            masks = output["attention_mask"].detach().cpu().numpy().astype(bool)

            for local_index, sequence_index in enumerate(indices):
                user_id = user_ids[sequence_index]
                shuffled_position_features = states[local_index][masks[local_index]]
                permutation = permutations[sequence_index]
                saved = assignments_by_sequence[sequence_index]
                feature_indices = saved["feature_index"].to_numpy(dtype=int)
                original_features = np.stack(
                    [saved_features[index] for index in feature_indices]
                )
                original_clusters = saved["cluster"].to_numpy(dtype=int)

                cluster_input = (
                    scaler.transform(shuffled_position_features)
                    if scaler is not None
                    else shuffled_position_features
                )
                shuffled_position_clusters = kmeans.predict(cluster_input).astype(int)

                shuffled_clusters_by_original = np.empty_like(
                    shuffled_position_clusters
                )
                shuffled_clusters_by_original[permutation] = shuffled_position_clusters
                shuffled_features_by_original = np.empty_like(
                    shuffled_position_features
                )
                shuffled_features_by_original[permutation] = shuffled_position_features

                cosines = cosine_similarity_rows(
                    original_features, shuffled_features_by_original
                )
                matches = shuffled_clusters_by_original == original_clusters
                total_matches += int(matches.sum())
                total_points += len(matches)
                cosine_sum += float(cosines.sum())
                cosine_sq_sum += float(np.square(cosines).sum())
                np.add.at(
                    confusion,
                    (original_clusters, shuffled_clusters_by_original),
                    1,
                )
                original_all.append(original_clusters)
                shuffled_all.append(shuffled_clusters_by_original)
                shuffled_labels_by_user[user_id] = shuffled_position_clusters
                user_rows.append(
                    {
                        "shuffle_seed": seed,
                        "user_id": user_id,
                        "n_interactions": len(matches),
                        "cluster_retention_pct": 100.0 * float(matches.mean()),
                        "cluster_change_pct": 100.0 * float((~matches).mean()),
                        "assignment_change_rate": float((~matches).mean()),
                        "mean_latent_cosine_similarity": float(cosines.mean()),
                        "original_switch_rate": switch_rate(original_clusters),
                        "shuffled_order_switch_rate": switch_rate(
                            shuffled_position_clusters
                        ),
                        "fixed_position_pct": 100.0
                        * float(np.mean(permutation == np.arange(len(permutation)))),
                    }
                )

    original_all = np.concatenate(original_all)
    shuffled_all = np.concatenate(shuffled_all)
    user_metrics = pd.DataFrame(user_rows)
    cosine_mean = cosine_sum / total_points
    cosine_variance = max(cosine_sq_sum / total_points - cosine_mean**2, 0.0)
    global_row = {
        "shuffle_seed": seed,
        "n_students": len(user_metrics),
        "n_interactions": total_points,
        "micro_cluster_retention_pct": 100.0 * total_matches / total_points,
        "micro_cluster_change_pct": 100.0
        * (total_points - total_matches)
        / total_points,
        "assignment_change_rate": (total_points - total_matches) / total_points,
        "macro_student_cluster_retention_pct": float(
            user_metrics["cluster_retention_pct"].mean()
        ),
        "mean_latent_cosine_similarity": cosine_mean,
        "std_latent_cosine_similarity": float(np.sqrt(cosine_variance)),
        "ari": adjusted_rand_score(original_all, shuffled_all),
        "nmi": normalized_mutual_info_score(original_all, shuffled_all),
        "fixed_position_pct": 100.0 * total_fixed_positions / total_points,
    }
    return global_row, user_metrics, confusion, shuffled_labels_by_user


def empirical_summary(values):
    values = pd.Series(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "median": float(values.median()),
        "q025": float(values.quantile(0.025)),
        "q975": float(values.quantile(0.975)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_global(global_metrics):
    metrics = [
        "assignment_change_rate",
        "micro_cluster_change_pct",
        "micro_cluster_retention_pct",
        "macro_student_cluster_retention_pct",
        "mean_latent_cosine_similarity",
        "std_latent_cosine_similarity",
        "ari",
        "nmi",
        "transition_matrix_frobenius_distance",
        "fixed_position_pct",
    ]
    return pd.DataFrame(
        [
            {
                "metric": metric,
                **empirical_summary(global_metrics[metric]),
                "n_shuffles": len(global_metrics),
            }
            for metric in metrics
        ]
    )


def summarize_students(student_metrics):
    rows = []
    for user_id, group in student_metrics.groupby("user_id", sort=True):
        retention = empirical_summary(group["cluster_retention_pct"])
        cosine = empirical_summary(group["mean_latent_cosine_similarity"])
        rows.append(
            {
                "user_id": user_id,
                "n_interactions": int(group["n_interactions"].iloc[0]),
                "retention_mean_pct": retention["mean"],
                "retention_std_pct": retention["std"],
                "change_mean_pct": 100.0 - retention["mean"],
                "assignment_change_rate_mean": 1.0 - retention["mean"] / 100.0,
                "latent_cosine_mean": cosine["mean"],
                "latent_cosine_std": cosine["std"],
                "original_switch_rate": float(group["original_switch_rate"].mean()),
                "shuffled_switch_rate_mean": float(
                    group["shuffled_order_switch_rate"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle within-student interactions and reproduce Table VI."
    )
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--data_name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--assignments_csv", required=True)
    parser.add_argument("--cluster_model", required=True)
    parser.add_argument("--latent_features", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_shuffles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12405)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--identity_min_match_rate", type=float, default=0.9999)
    args = parser.parse_args()

    if args.num_shuffles < 1:
        raise ValueError("--num_shuffles must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")
    if not 0.0 <= args.identity_min_match_rate <= 1.0:
        raise ValueError("--identity_min_match_rate must be between 0 and 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    config = load_config(args.config)
    config.model_name = "cl4kt"
    config.data_name = args.data_name
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    frame, users, num_skills, num_questions = prepare_dataframe(config)
    test_frame, _ = build_test_dataframe(frame, users, args.fold, int(config.seed))
    user_ids = [user_id for user_id, _ in test_frame.groupby("user_id")]
    dataset = dataset_class(config.train_config.sequence_option)(
        test_frame,
        config.train_config.seq_len,
        num_skills,
        num_questions,
    )
    dataset = move_dataset_to_cpu(dataset)
    model = load_model(
        config,
        num_skills,
        num_questions,
        config.train_config.seq_len,
        args.checkpoint,
        device,
    )

    assignments = pd.read_csv(args.assignments_csv)
    saved_features = load_saved_latents(args.latent_features)
    bundle = load_cluster_bundle(args.cluster_model)
    cluster_verification = validate_inputs(assignments, saved_features, bundle)
    assignments_by_sequence = group_assignments(assignments)
    alignment = validate_dataset_alignment(dataset, assignments_by_sequence)
    identity_verification = verify_identity_inference(
        model,
        dataset,
        assignments_by_sequence,
        saved_features,
        bundle,
        args.batch_size,
        device,
        args.identity_min_match_rate,
    )

    n_clusters = int(bundle["kmeans"].n_clusters)
    original_labels_by_user = {
        group["user_id"].iloc[0]: group["cluster"].to_numpy(dtype=int)
        for group in assignments_by_sequence.values()
    }
    original_transition = transition_probabilities(
        transition_counts(original_labels_by_user, n_clusters)
    )

    global_rows = []
    student_frames = []
    aggregate_confusion = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for repetition in range(args.num_shuffles):
        shuffle_seed = args.seed + repetition
        global_row, student_metrics, confusion, shuffled_labels_by_user = (
            infer_one_shuffle(
                model,
                dataset,
                user_ids,
                assignments_by_sequence,
                saved_features,
                bundle,
                shuffle_seed,
                args.batch_size,
                device,
            )
        )
        shuffled_transition = transition_probabilities(
            transition_counts(shuffled_labels_by_user, n_clusters)
        )
        global_row["transition_matrix_frobenius_distance"] = float(
            np.linalg.norm(original_transition - shuffled_transition)
        )
        global_rows.append(global_row)
        student_frames.append(student_metrics)
        aggregate_confusion += confusion

    global_metrics = pd.DataFrame(global_rows)
    student_metrics = pd.concat(student_frames, ignore_index=True)
    global_summary = summarize_global(global_metrics)
    student_summary = summarize_students(student_metrics)
    table_vi = pd.DataFrame(
        [
            {
                "dataset": args.data_name,
                "n_shuffles": args.num_shuffles,
                "change_rate": float(global_metrics["assignment_change_rate"].mean()),
                "change_rate_pct": float(
                    global_metrics["micro_cluster_change_pct"].mean()
                ),
                "mean_latent_cosine": float(
                    global_metrics["mean_latent_cosine_similarity"].mean()
                ),
            }
        ]
    )
    preservation = pd.DataFrame(
        [
            {
                "input_sequence_correctness_preserved_by_design": True,
                "interaction_multiset_preserved_by_design": True,
                "mean_fixed_position_pct": float(
                    global_metrics["fixed_position_pct"].mean()
                ),
                "model_retrained": False,
                "kmeans_refitted": False,
            }
        ]
    )

    global_metrics.to_csv(output_dir / "order_shuffle_runs.csv", index=False)
    student_metrics.to_csv(output_dir / "order_shuffle_per_user.csv", index=False)
    table_vi.to_csv(output_dir / "order_shuffle_summary.csv", index=False)
    global_metrics.to_csv(output_dir / "global_shuffle_metrics.csv", index=False)
    global_summary.to_csv(output_dir / "global_shuffle_summary.csv", index=False)
    student_summary.to_csv(
        output_dir / "per_student_shuffle_summary.csv", index=False
    )
    pd.DataFrame(aggregate_confusion).to_csv(
        output_dir / "aggregate_cluster_migration_counts.csv",
        index_label="original_cluster",
    )
    cluster_verification.to_csv(
        output_dir / "cluster_model_verification.csv", index=False
    )
    alignment.to_csv(output_dir / "dataset_assignment_alignment.csv", index=False)
    identity_verification.to_csv(
        output_dir / "identity_checkpoint_verification.csv", index=False
    )
    preservation.to_csv(output_dir / "shuffle_preservation_checks.csv", index=False)

    print(table_vi.to_string(index=False))
    print(f"Device: {device}")
    print(f"Saved order-shuffle control to: {output_dir}")


if __name__ == "__main__":
    main()
