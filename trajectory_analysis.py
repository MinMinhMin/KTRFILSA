import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from extract_and_cluster import (
    build_test_dataframe,
    dataset_class,
    latest_checkpoint,
    load_config,
    load_model,
    make_loader,
    prepare_dataframe,
)


def cluster_cmap_norm(n_clusters):
    cmap = plt.get_cmap("tab10", n_clusters)
    norm = BoundaryNorm(np.arange(-0.5, n_clusters + 0.5, 1), n_clusters)
    return cmap, norm


def ordered_user_ids(df):
    return [user_id for user_id, _ in df.groupby("user_id")]


def load_cluster_bundle(path):
    bundle = joblib.load(path)
    if hasattr(bundle, "predict"):
        return {"kmeans": bundle, "scaler": None, "best_k": bundle.n_clusters}
    if "kmeans" not in bundle:
        raise ValueError("Cluster model file must contain a 'kmeans' object")
    return bundle


def extract_sequential_features(model, loader, dataset, user_ids, device):
    rows = []
    feature_blocks = []
    offset = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting sequential latent states"):
            batch_size = batch["skills"].size(0)
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model.extract_features(batch, pool=False)
            states = out["sequence_features"].detach().cpu().numpy()
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
                feature_blocks.extend(valid_states.astype(np.float32))
            offset += batch_size

    if not feature_blocks:
        raise ValueError("No non-padding latent states were extracted")
    features = np.stack(feature_blocks).astype(np.float32)
    return features, pd.DataFrame(rows)


def assign_dynamic_clusters(features, metadata, bundle):
    scaler = bundle.get("scaler")
    kmeans = bundle["kmeans"]
    cluster_features = scaler.transform(features) if scaler is not None else features
    metadata = metadata.copy()
    metadata["cluster_label"] = kmeans.predict(cluster_features)
    return cluster_features, metadata


def compute_transition_matrix(labels_by_user, n_clusters):
    counts = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for labels in labels_by_user.values():
        if len(labels) < 2:
            continue
        for src, dst in zip(labels[:-1], labels[1:]):
            counts[int(src), int(dst)] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    matrix = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_sums > 0,
    )
    return counts, matrix


def labels_grouped_by_user(states_df):
    grouped = {}
    for user_id, user_df in states_df.sort_values(["user_id", "step_index"]).groupby("user_id"):
        grouped[user_id] = user_df["cluster_label"].to_numpy(dtype=int)
    return grouped


def save_transition_outputs(counts, matrix, output_dir):
    index = [f"cluster_{i}" for i in range(matrix.shape[0])]
    counts_df = pd.DataFrame(counts, index=index, columns=index)
    matrix_df = pd.DataFrame(matrix, index=index, columns=index)
    counts_df.to_csv(output_dir / "transition_counts.csv")
    matrix_df.to_csv(output_dir / "transition_matrix.csv")
    return matrix_df


def plot_transition_heatmap(matrix_df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix_df,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Transition probability"},
        ax=ax,
    )
    ax.set_title("Cluster Transition Probability Matrix")
    ax.set_xlabel("Next cluster")
    ax.set_ylabel("Current cluster")
    fig.tight_layout()
    path = output_dir / "transition_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fit_projection(cluster_features):
    pca = PCA(n_components=2, random_state=0)
    return pca, pca.fit_transform(cluster_features)


def plot_student_trajectory(user_id, trajectory_df, output_dir):
    user_df = trajectory_df[trajectory_df["user_id"].astype(str) == str(user_id)].sort_values("step_index")
    if user_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        user_df["x"],
        user_df["y"],
        c=user_df["cluster_label"],
        cmap="tab10",
        s=42,
        edgecolors="black",
        linewidths=0.35,
        zorder=3,
    )
    xs = user_df["x"].to_numpy()
    ys = user_df["y"].to_numpy()
    for idx in range(len(user_df) - 1):
        ax.annotate(
            "",
            xy=(xs[idx + 1], ys[idx + 1]),
            xytext=(xs[idx], ys[idx]),
            arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 1.0, "alpha": 0.75},
        )
    ax.set_title(f"Learning Trajectory: user {user_id}")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    safe_user_id = str(user_id).replace("/", "_")
    path = output_dir / f"trajectory_user_{safe_user_id}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_student_cluster_timeline(user_id, trajectory_df, output_dir, n_clusters):
    user_df = trajectory_df[trajectory_df["user_id"].astype(str) == str(user_id)].sort_values("step_index")
    if user_df.empty:
        return None

    cmap, norm = cluster_cmap_norm(n_clusters)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.step(
        user_df["step_index"],
        user_df["cluster_label"],
        where="post",
        color="0.35",
        linewidth=1.6,
        alpha=0.75,
    )
    scatter = ax.scatter(
        user_df["step_index"],
        user_df["cluster_label"],
        c=user_df["cluster_label"],
        cmap=cmap,
        norm=norm,
        s=32,
        edgecolors="black",
        linewidths=0.25,
        zorder=3,
    )
    ax.set_title(f"Cluster Timeline: user {user_id}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cluster")
    ax.set_yticks(range(n_clusters))
    ax.set_ylim(-0.4, n_clusters - 0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(n_clusters), pad=0.02)
    cbar.set_label("Cluster")
    fig.tight_layout()
    safe_user_id = str(user_id).replace("/", "_")
    path = output_dir / f"cluster_timeline_user_{safe_user_id}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_cluster_timeline_overlay(user_ids, trajectory_df, output_dir, n_clusters):
    selected = trajectory_df[trajectory_df["user_id"].astype(str).isin([str(user_id) for user_id in user_ids])]
    if selected.empty:
        return None

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for user_id, user_df in selected.sort_values(["user_id", "step_index"]).groupby("user_id"):
        ax.step(
            user_df["step_index"],
            user_df["cluster_label"],
            where="post",
            linewidth=1.3,
            alpha=0.72,
            label=str(user_id),
        )
    ax.set_title("Cluster Timelines")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cluster")
    ax.set_yticks(range(n_clusters))
    ax.set_ylim(-0.4, n_clusters - 0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if len(user_ids) <= 12:
        ax.legend(title="user_id", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    path = output_dir / "cluster_timeline_selected_students.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def choose_students(states_df, requested_ids, max_students):
    if requested_ids:
        return requested_ids
    lengths = states_df.groupby("user_id").size().sort_values(ascending=False)
    return [str(user_id) for user_id in lengths.head(max_students).index]


def stage_label(labels, fraction):
    if len(labels) == 0:
        return None
    idx = min(len(labels) - 1, max(0, int(np.ceil(len(labels) * fraction)) - 1))
    return int(labels[idx])


def plot_sankey(labels_by_user, n_clusters, output_dir):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    stages = [("t1", 1 / 3), ("t2", 2 / 3), ("t3", 1.0)]
    node_labels = [f"{stage}: C{cluster}" for stage, _ in stages for cluster in range(n_clusters)]
    node_index = {(stage_idx, cluster): stage_idx * n_clusters + cluster for stage_idx in range(len(stages)) for cluster in range(n_clusters)}
    flows = {}

    for labels in labels_by_user.values():
        stage_clusters = [stage_label(labels, fraction) for _, fraction in stages]
        if any(cluster is None for cluster in stage_clusters):
            continue
        for stage_idx in range(len(stage_clusters) - 1):
            src = node_index[(stage_idx, stage_clusters[stage_idx])]
            dst = node_index[(stage_idx + 1, stage_clusters[stage_idx + 1])]
            flows[(src, dst)] = flows.get((src, dst), 0) + 1

    if not flows:
        return None

    sources, targets, values = zip(*[(src, dst, value) for (src, dst), value in flows.items()])
    fig = go.Figure(
        data=[
            go.Sankey(
                node={"label": node_labels, "pad": 15, "thickness": 16},
                link={"source": sources, "target": targets, "value": values},
            )
        ]
    )
    fig.update_layout(title_text="Cluster Flow Across Learning Stages", font_size=11)
    path = output_dir / "cluster_flow_sankey.html"
    fig.write_html(path)
    return path


def parse_user_ids(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Analyze dynamic CL4KT learning trajectories.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--data_name", default="algebra05")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cluster_model", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs/trajectory_analysis")
    parser.add_argument("--student_ids", default="", help="Comma-separated user ids to plot.")
    parser.add_argument("--max_students_plot", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    args = parser.parse_args()

    config = load_config(args.config)
    config.model_name = "cl4kt"
    config.data_name = args.data_name
    if args.seed is not None:
        config.seed = args.seed

    has_official_split = "split" in pd.read_csv(
        Path(config.dataset_path) / config.data_name / "preprocessed_df.csv",
        sep="\t",
        nrows=1,
    ).columns
    split_name = "official" if has_official_split else f"fold_{args.fold}"
    output_dir = Path(args.output_dir) / config.data_name / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    df, users, num_skills, num_questions = prepare_dataframe(config)
    if "split" not in df.columns:
        rng = np.random.RandomState(config.seed)
        users = users.copy()
        rng.shuffle(users)

    test_df, _ = build_test_dataframe(df, users, args.fold, config.seed)
    user_ids = ordered_user_ids(test_df)
    dataset = dataset_class(config.train_config.sequence_option)(
        test_df,
        config.train_config.seq_len,
        num_skills,
        num_questions,
    )
    loader = make_loader(config, dataset)

    checkpoint = args.checkpoint or latest_checkpoint(config.model_name, config.data_name)
    cluster_model = args.cluster_model or (
        Path("outputs/latent_clustering") / config.data_name / split_name / "kmeans_model.pkl"
    )

    device = torch.device(args.device)
    model = load_model(config, num_skills, num_questions, config.train_config.seq_len, checkpoint, device)
    bundle = load_cluster_bundle(cluster_model)

    raw_features, states_df = extract_sequential_features(model, loader, dataset, user_ids, device)
    cluster_features, states_df = assign_dynamic_clusters(raw_features, states_df, bundle)

    n_clusters = int(bundle.get("best_k") or bundle["kmeans"].n_clusters)
    labels_by_user = labels_grouped_by_user(states_df)
    counts, matrix = compute_transition_matrix(labels_by_user, n_clusters)
    matrix_df = save_transition_outputs(counts, matrix, output_dir)
    heatmap_path = plot_transition_heatmap(matrix_df, output_dir)

    projector, embedding = fit_projection(cluster_features)
    states_df["x"] = embedding[:, 0]
    states_df["y"] = embedding[:, 1]
    states_df.to_csv(output_dir / "sequential_states.csv", index=False)
    states_df.to_pickle(output_dir / "sequential_states.pkl")
    states_with_vectors = states_df.copy()
    states_with_vectors["hidden_state"] = list(raw_features)
    states_with_vectors.to_pickle(output_dir / "sequential_states_with_vectors.pkl")
    np.savez_compressed(
        output_dir / "sequential_hidden_states.npz",
        features=raw_features,
        cluster_features=cluster_features,
        cluster_labels=states_df["cluster_label"].to_numpy(),
        feature_index=states_df["feature_index"].to_numpy(),
        user_ids=states_df["user_id"].to_numpy(),
        step_indices=states_df["step_index"].to_numpy(),
    )
    joblib.dump(projector, output_dir / "pca_projector.pkl")

    requested_ids = parse_user_ids(args.student_ids)
    students = choose_students(states_df, requested_ids, args.max_students_plot)
    trajectory_paths = []
    timeline_paths = []
    for user_id in students:
        path = plot_student_trajectory(user_id, states_df, output_dir)
        if path is not None:
            trajectory_paths.append(path)
        timeline_path = plot_student_cluster_timeline(user_id, states_df, output_dir, n_clusters)
        if timeline_path is not None:
            timeline_paths.append(timeline_path)
    overlay_timeline_path = plot_cluster_timeline_overlay(students, states_df, output_dir, n_clusters)

    sankey_path = plot_sankey(labels_by_user, n_clusters, output_dir)

    print(f"Checkpoint: {checkpoint}")
    print(f"Cluster model: {cluster_model}")
    print(f"Saved outputs to: {output_dir}")
    print(f"Transition heatmap: {heatmap_path}")
    print(f"Student trajectory plots: {len(trajectory_paths)}")
    print(f"Student cluster timeline plots: {len(timeline_paths)}")
    if overlay_timeline_path is not None:
        print(f"Selected-students cluster timeline: {overlay_timeline_path}")
    if sankey_path is not None:
        print(f"Sankey diagram: {sankey_path}")


if __name__ == "__main__":
    main()
