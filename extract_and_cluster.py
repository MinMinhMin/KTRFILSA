import argparse
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loaders import MostEarlyQuestionSkillDataset, MostRecentQuestionSkillDataset, SimCLRDatasetWrapper
from models.cl4kt import CL4KT
from utils.config import ConfigNode as CN
from utils.file_io import PathManager


def load_config(path):
    with PathManager.open(path, "r") as f:
        cfg = CN(yaml.safe_load(f))
    cfg.set_new_allowed(True)
    return cfg


def dataset_class(sequence_option):
    if sequence_option == "recent":
        return MostRecentQuestionSkillDataset
    if sequence_option == "early":
        return MostEarlyQuestionSkillDataset
    raise ValueError("sequence_option must be 'recent' or 'early'")


def latest_checkpoint(model_name, data_name):
    patterns = [
        os.path.join("saved_model", model_name, data_name, "params_*"),
        os.path.join(".ckpts", model_name, data_name, "params_*"),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint found. Pass --checkpoint, or train first so a params_* file exists."
        )
    return max(candidates, key=os.path.getmtime)


def load_model(config, num_skills, num_questions, seq_len, checkpoint_path, device):
    model = CL4KT(num_skills, num_questions, seq_len, **config.cl4kt_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_dataframe(config):
    df_path = os.path.join(config.dataset_path, config.data_name, "preprocessed_df.csv")
    df = pd.read_csv(df_path, sep="\t")
    df["skill_id"] += 1
    df["item_id"] += 1
    users = df["user_id"].unique()
    num_skills = df["skill_id"].max() + 1
    num_questions = df["item_id"].max() + 1
    return df, users, num_skills, num_questions


def build_test_dataframe(df, users, fold, seed):
    if "split" in df.columns and "test" in set(df["split"].unique()):
        test_df = df[df["split"] == "test"].copy()
        return test_df, test_df["user_id"].unique()

    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(kfold.split(users))
    if fold < 0 or fold >= len(splits):
        raise ValueError("--fold must be between 0 and 4 for the repo's 5-fold split")
    _, test_ids = splits[fold]
    test_users = users[test_ids]
    return df[df["user_id"].isin(test_users)].copy(), test_users


def make_loader(config, dataset):
    wrapped = SimCLRDatasetWrapper(
        dataset,
        config.train_config.seq_len,
        0,
        0,
        0,
        0,
        0,
        eval_mode=True,
    )
    return DataLoader(
        wrapped,
        batch_size=config.train_config.eval_batch_size,
        shuffle=False,
    )


def selected_sequence_stats(dataset, user_ids):
    rows = []
    for idx, user_id in enumerate(user_ids):
        mask = dataset.attention_mask[idx].cpu().numpy().astype(bool)
        responses = dataset.padded_r[idx].cpu().numpy()[mask]
        rows.append(
            {
                "sequence_index": idx,
                "user_id": user_id,
                "num_interactions": int(mask.sum()),
                "mean_correct": float(responses.mean()) if len(responses) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ordered_user_ids(df):
    return [user_id for user_id, _ in df.groupby("user_id")]


def extract_timestep_features(model, loader, dataset, user_ids, device):
    rows = []
    feature_blocks = []
    offset = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting timestep latent features"):
            batch_size = batch["skills"].size(0)
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model.extract_features(batch, pool=False)
            states = out["sequence_features"].detach().cpu().numpy().astype(np.float32)
            masks = out["attention_mask"].detach().cpu().numpy().astype(bool)

            for local_idx in range(batch_size):
                sequence_index = offset + local_idx
                user_id = user_ids[sequence_index]
                valid_mask = masks[local_idx]
                valid_states = states[local_idx][valid_mask]
                valid_positions = np.flatnonzero(valid_mask)

                questions = dataset.padded_q[sequence_index].cpu().numpy()[valid_mask]
                skills = dataset.padded_s[sequence_index].cpu().numpy()[valid_mask]
                responses = dataset.padded_r[sequence_index].cpu().numpy()[valid_mask]

                start = len(feature_blocks)
                for step_index, (position, question, skill, response) in enumerate(
                    zip(valid_positions, questions, skills, responses)
                ):
                    rows.append(
                        {
                            "feature_index": start + step_index,
                            "sequence_index": sequence_index,
                            "user_id": user_id,
                            "step_index": step_index,
                            "token_position": int(position),
                            "item_id": int(question),
                            "skill_id": int(skill),
                            "correct": int(response),
                        }
                    )
                feature_blocks.extend(valid_states)
            offset += batch_size

    if not feature_blocks:
        raise ValueError("No non-padding timestep latent states were extracted")
    return np.stack(feature_blocks).astype(np.float32), pd.DataFrame(rows)


def sample_for_metric(features, labels, seed, sample_size):
    if sample_size > 0 and len(features) > sample_size:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(features), size=sample_size, replace=False)
        return features[indices], labels[indices]
    return features, labels


def evaluate_kmeans(
    features_for_clustering,
    raw_features,
    metadata,
    output_dir,
    k_values,
    seed,
    scaler=None,
    metric_sample=5000,
):
    metrics = []
    models = {}
    for k in k_values:
        if len(features_for_clustering) <= k:
            continue
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(features_for_clustering)
        metric_features, metric_labels = sample_for_metric(features_for_clustering, labels, seed, metric_sample)
        metrics.append(
            {
                "k": k,
                "metric_sample_size": len(metric_features),
                "silhouette_score": silhouette_score(metric_features, metric_labels),
                "davies_bouldin_index": davies_bouldin_score(metric_features, metric_labels),
                "calinski_harabasz_score": calinski_harabasz_score(
                    metric_features, metric_labels
                ),
                "cluster_size_min_max_ratio": float(
                    np.bincount(labels, minlength=k).min()
                    / np.bincount(labels, minlength=k).max()
                ),
            }
        )
        models[k] = (kmeans, labels)

    metrics_df = pd.DataFrame(metrics)
    if metrics_df.empty:
        raise ValueError("Not enough feature rows to evaluate the requested k values")

    metrics_df.to_csv(output_dir / "cluster_metrics.csv", index=False)
    best_k = int(metrics_df.sort_values(
        ["silhouette_score", "davies_bouldin_index"], ascending=[False, True]
    ).iloc[0]["k"])
    best_model, labels = models[best_k]

    labeled = metadata.copy()
    labeled["cluster"] = labels
    labeled.to_csv(output_dir / f"cluster_assignments_k{best_k}.csv", index=False)

    representatives = []
    for cluster_id, center in enumerate(best_model.cluster_centers_):
        cluster_indices = np.where(labels == cluster_id)[0]
        dists = np.linalg.norm(features_for_clustering[cluster_indices] - center, axis=1)
        order = np.argsort(dists)[:5]
        reps = labeled.iloc[cluster_indices[order]].copy()
        reps["distance_to_centroid"] = dists[order]
        representatives.append(reps)
    pd.concat(representatives).to_csv(output_dir / f"representatives_k{best_k}.csv", index=False)

    np.savez_compressed(
        output_dir / "latent_features.npz",
        features=raw_features,
        user_ids=metadata["user_id"].to_numpy(),
        sequence_indices=metadata["sequence_index"].to_numpy(),
        step_indices=metadata["step_index"].to_numpy(),
        feature_index=metadata["feature_index"].to_numpy(),
        labels=labels,
        best_k=np.array(best_k),
    )
    joblib.dump(
        {
            "kmeans": best_model,
            "scaler": scaler,
            "best_k": best_k,
            "cluster_unit": "timestep",
            "feature_space": "standardized" if scaler is not None else "raw",
        },
        output_dir / "kmeans_model.pkl",
    )
    return best_k, labels, metrics_df


def visualize_tsne(features_for_embedding, labels, metadata, output_dir, seed, sample_size):
    if len(features_for_embedding) < 3:
        return None
    if sample_size > 0 and len(features_for_embedding) > sample_size:
        rng = np.random.RandomState(seed)
        indices = np.sort(rng.choice(len(features_for_embedding), size=sample_size, replace=False))
        plot_features = features_for_embedding[indices]
        plot_labels = labels[indices]
        plot_metadata = metadata.iloc[indices].reset_index(drop=True)
    else:
        plot_features = features_for_embedding
        plot_labels = labels
        plot_metadata = metadata.reset_index(drop=True)

    perplexity = min(30, max(2, (len(plot_features) - 1) // 3))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(plot_features)

    tsne_df = plot_metadata.copy()
    tsne_df["tsne_x"] = embedding[:, 0]
    tsne_df["tsne_y"] = embedding[:, 1]
    tsne_df["cluster"] = plot_labels
    tsne_df.to_csv(output_dir / "tsne_embedding.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=plot_labels,
        s=10,
        cmap="tab10",
        alpha=0.75,
        linewidths=0,
    )
    ax.set_title("CL4KT Timestep Latent Space Clusters")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    path = output_dir / "clusters_tsne.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def parse_k_values(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Extract CL4KT timestep latent vectors and cluster behavior states.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--data_name", default="algebra05")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--fold", type=int, default=0, help="Which 5-fold CV test split to extract.")
    parser.add_argument("--output_dir", default="outputs/latent_clustering")
    parser.add_argument("--k_values", default="3,4,5,6")
    parser.add_argument("--metric_sample", type=int, default=5000)
    parser.add_argument("--tsne_sample", type=int, default=5000)
    parser.add_argument("--no_standardize", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    args = parser.parse_args()

    config = load_config(args.config)
    config.model_name = "cl4kt"
    if args.data_name is not None:
        config.data_name = args.data_name
    if args.seed is not None:
        config.seed = args.seed

    split_name = "official" if "split" in pd.read_csv(os.path.join(config.dataset_path, config.data_name, "preprocessed_df.csv"), sep="\t", nrows=1).columns else f"fold_{args.fold}"
    output_dir = Path(args.output_dir) / config.data_name / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    df, users, num_skills, num_questions = prepare_dataframe(config)
    if "split" not in df.columns:
        rng = np.random.RandomState(config.seed)
        users = users.copy()
        rng.shuffle(users)

    test_df, _ = build_test_dataframe(df, users, args.fold, config.seed)
    dataset = dataset_class(config.train_config.sequence_option)(
        test_df,
        config.train_config.seq_len,
        num_skills,
        num_questions,
    )
    loader = make_loader(config, dataset)

    checkpoint = args.checkpoint or latest_checkpoint(config.model_name, config.data_name)
    device = torch.device(args.device)
    model = load_model(config, num_skills, num_questions, config.train_config.seq_len, checkpoint, device)

    user_ids = ordered_user_ids(test_df)
    features, metadata = extract_timestep_features(model, loader, dataset, user_ids, device)
    metadata.to_csv(output_dir / "latent_metadata.csv", index=False)

    if args.no_standardize:
        scaler = None
        cluster_features = features
    else:
        scaler = StandardScaler()
        cluster_features = scaler.fit_transform(features)

    best_k, labels, metrics_df = evaluate_kmeans(
        cluster_features,
        features,
        metadata,
        output_dir,
        parse_k_values(args.k_values),
        config.seed,
        scaler=scaler,
        metric_sample=args.metric_sample,
    )
    plot_path = visualize_tsne(cluster_features, labels, metadata, output_dir, config.seed, args.tsne_sample)

    print(f"Checkpoint: {checkpoint}")
    print(f"Saved outputs to: {output_dir}")
    print(metrics_df.to_string(index=False))
    print(f"Best k: {best_k}")
    if plot_path is not None:
        print(f"t-SNE plot: {plot_path}")


if __name__ == "__main__":
    main()
