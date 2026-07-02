#!/usr/bin/env python3
"""Generate clear, dependency-light visual summaries for Experiment 3."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("final_experiment_results/experiment_3")
PROFILE_PATH = ROOT / "pedagogical_profile_long.csv"
TEST_PATH = ROOT / "pedagogical_statistical_tests.csv"
FIGURE_DIR = ROOT / "figures"
DATASET = "XES3G5M"
TEMPORAL_MODE = "time"

STATE_ORDER = []
STATE_NAMES = {}
STATE_SHORT = {}
STATE_COLORS = {}

FEATURE_LABELS = {
    "correct": "Current correctness",
    "previous_correct_rate": "Prefix correctness",
    "skill_difficulty": "Skill difficulty",
    "hard_skill_ratio": "Hard-skill ratio",
    "skill_entropy": "Skill entropy",
    "repeat_skill_ratio": "Repeat-skill ratio",
    "repeat_question_ratio": "Repeat-question ratio",
    "longest_correct_streak": "Longest correct streak",
    "longest_incorrect_streak": "Longest incorrect streak",
    "median_time_gap_sec": "Median time gap (min)",
    "active_days": "Active days",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Experiment 3 figures from an experiment result directory."
    )
    parser.add_argument(
        "--root",
        default="final_experiment_results/experiment_3",
        help="Directory containing pedagogical_profile_long.csv and statistical tests.",
    )
    parser.add_argument("--dataset", default="XES3G5M")
    parser.add_argument(
        "--temporal-mode",
        choices=["time", "ordinal"],
        default="time",
        help="Use ordinal when the dataset has event order but no real timestamps.",
    )
    return parser.parse_args()


def configure(args, profile):
    global ROOT, PROFILE_PATH, TEST_PATH, FIGURE_DIR, DATASET, TEMPORAL_MODE
    global FEATURE_LABELS
    global STATE_ORDER, STATE_NAMES, STATE_SHORT, STATE_COLORS

    ROOT = Path(args.root)
    PROFILE_PATH = ROOT / "pedagogical_profile_long.csv"
    TEST_PATH = ROOT / "pedagogical_statistical_tests.csv"
    FIGURE_DIR = ROOT / "figures"
    DATASET = args.dataset
    TEMPORAL_MODE = args.temporal_mode
    if TEMPORAL_MODE == "ordinal":
        FEATURE_LABELS = {
            key: value
            for key, value in FEATURE_LABELS.items()
            if key != "active_days"
        }
        FEATURE_LABELS["median_time_gap_sec"] = "Median order-ID gap"

    correctness = (
        profile.loc[profile["feature"] == "correct", ["cluster", "mean"]]
        .sort_values("mean")
    )
    STATE_ORDER = correctness["cluster"].astype(int).tolist()
    if len(STATE_ORDER) != 3:
        raise ValueError(f"Expected exactly 3 clusters, found {len(STATE_ORDER)}")

    short_names = ["Low", "Middle", "High"]
    long_names = ["Low / Needs support", "Middle / Developing", "High / Stable"]
    colors = ["#D85C5C", "#D9A72E", "#4C9A68"]
    STATE_SHORT = dict(zip(STATE_ORDER, short_names))
    STATE_NAMES = dict(zip(STATE_ORDER, long_names))
    STATE_COLORS = dict(zip(STATE_ORDER, colors))


def cluster_counts(profile):
    counts = profile.loc[profile["feature"] == "correct", ["cluster", "n"]].copy()
    users_by_cluster = {}
    table_path = ROOT / "pedagogical_profile_table.csv"
    if table_path.exists():
        table = pd.read_csv(table_path, usecols=["cluster", "n_users"])
        users_by_cluster = dict(
            zip(table["cluster"].astype(int), table["n_users"].astype(int))
        )
    return {
        int(row.cluster): (int(row.n), users_by_cluster.get(int(row.cluster), 0))
        for row in counts.itertuples(index=False)
    }


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    paths = [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    ]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text(draw, xy, value, size=28, fill="#172033", bold=False, anchor=None):
    draw.text(xy, str(value), font=font(size, bold), fill=fill, anchor=anchor)


def save(image, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    image.save(path, quality=95)
    return path


def profile_lookup(profile):
    return {
        (int(row.cluster), row.feature): row
        for row in profile.itertuples(index=False)
    }


def value(lookup, cluster, feature, statistic="mean"):
    row = lookup[(cluster, feature)]
    result = float(getattr(row, statistic))
    if feature == "median_time_gap_sec" and TEMPORAL_MODE == "time":
        result /= 60.0
    return result


def draw_profile_cards(profile):
    lookup = profile_lookup(profile)
    counts = cluster_counts(profile)
    image = Image.new("RGB", (1800, 760), "white")
    draw = ImageDraw.Draw(image)
    text(draw, (80, 48), f"Experiment 3 — Pedagogical cluster profiles ({DATASET})", 42, bold=True)
    text(
        draw,
        (80, 105),
        (
            "Profiles are computed after clustering; order gaps are not explicit CL4KT input features."
            if TEMPORAL_MODE == "ordinal"
            else "Profiles are computed after clustering; difficulty and time are not explicit CL4KT input features."
        ),
        24,
        fill="#596273",
    )

    card_width = 520
    x_positions = [70, 640, 1210]
    descriptions = {
        "Low": "Lower history accuracy,\nlonger incorrect streaks",
        "Middle": "Intermediate performance,\nmore transitional",
        "High": "High sustained accuracy,\nlong correct streaks",
    }

    for cluster, x in zip(STATE_ORDER, x_positions):
        color = STATE_COLORS[cluster]
        draw.rounded_rectangle((x, 170, x + card_width, 700), radius=28, fill="#F8FAFC", outline=color, width=5)
        draw.rounded_rectangle((x, 170, x + card_width, 255), radius=26, fill=color)
        text(draw, (x + 28, 208), STATE_NAMES[cluster], 30, fill="white", bold=True, anchor="lm")
        text(draw, (x + 30, 290), descriptions[STATE_SHORT[cluster]], 25, fill="#394150")

        metrics = [
            ("Current correctness", value(lookup, cluster, "correct"), ".3f"),
            ("Prefix correctness", value(lookup, cluster, "previous_correct_rate"), ".3f"),
            ("Correct streak", value(lookup, cluster, "longest_correct_streak"), ".1f"),
            ("Incorrect streak", value(lookup, cluster, "longest_incorrect_streak"), ".1f"),
            (
                "Median time gap" if TEMPORAL_MODE == "time" else "Median order-ID gap",
                value(lookup, cluster, "median_time_gap_sec", "median"),
                ".1f",
            ),
        ]
        y = 395
        for label, metric, fmt in metrics:
            text(draw, (x + 30, y), label, 23, fill="#596273")
            suffix = " min" if label == "Median time gap" else ""
            text(draw, (x + card_width - 30, y), format(metric, fmt) + suffix, 25, bold=True, anchor="ra")
            y += 50
        text(
            draw,
            (x + 30, 660),
            f"{counts[cluster][0]:,} timesteps · {counts[cluster][1]:,} students",
            21,
            fill="#596273",
        )
    return save(image, "cluster_profile_cards.png")


def mix_color(low, high, amount):
    def rgb(color):
        color = color.lstrip("#")
        return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))

    a, b = rgb(low), rgb(high)
    values = tuple(round(a[i] + (b[i] - a[i]) * amount) for i in range(3))
    return "#%02x%02x%02x" % values


def draw_heatmap(profile):
    lookup = profile_lookup(profile)
    features = list(FEATURE_LABELS)
    image = Image.new("RGB", (1600, 1080), "white")
    draw = ImageDraw.Draw(image)
    text(draw, (65, 42), "Relative profile heatmap", 42, bold=True)
    text(
        draw,
        (65, 98),
        "Darker cells mean a larger value within that feature; color does not imply better or worse.",
        23,
        fill="#596273",
    )

    left = 440
    top = 185
    cell_w = 330
    cell_h = 72
    for col, cluster in enumerate(STATE_ORDER):
        x = left + col * cell_w
        draw.rectangle((x, top - 60, x + cell_w - 8, top - 8), fill=STATE_COLORS[cluster])
        text(draw, (x + (cell_w - 8) / 2, top - 34), STATE_SHORT[cluster], 27, fill="white", bold=True, anchor="mm")

    for row_idx, feature in enumerate(features):
        y = top + row_idx * cell_h
        text(draw, (65, y + cell_h / 2), FEATURE_LABELS[feature], 24, anchor="lm")
        statistic = "median" if feature == "median_time_gap_sec" else "mean"
        values = np.asarray([value(lookup, c, feature, statistic) for c in STATE_ORDER])
        span = values.max() - values.min()
        normalized = (values - values.min()) / span if span > 0 else np.full(3, 0.5)
        for col, (cluster, metric, norm) in enumerate(zip(STATE_ORDER, values, normalized)):
            x = left + col * cell_w
            fill = mix_color("#EDF3FA", "#315D8A", 0.18 + 0.82 * float(norm))
            draw.rectangle((x, y, x + cell_w - 8, y + cell_h - 8), fill=fill)
            if feature in {"correct", "previous_correct_rate", "skill_difficulty", "hard_skill_ratio", "repeat_skill_ratio", "repeat_question_ratio"}:
                formatted = f"{metric:.3f}"
            elif feature == "median_time_gap_sec":
                formatted = (
                    f"{metric:.1f} min"
                    if TEMPORAL_MODE == "time"
                    else f"{metric:.1f}"
                )
            else:
                formatted = f"{metric:.2f}"
            text(draw, (x + (cell_w - 8) / 2, y + (cell_h - 8) / 2), formatted, 25, fill="white" if norm > 0.55 else "#172033", bold=True, anchor="mm")
    return save(image, "cluster_profile_heatmap.png")


def draw_effect_sizes(tests):
    tests = tests.sort_values("epsilon_squared", ascending=True).reset_index(drop=True)
    image = Image.new("RGB", (1650, 750), "white")
    draw = ImageDraw.Draw(image)

    left = 470
    right = 1550
    top = 30
    row_h = 62
    max_value = max(0.35, float(tests["epsilon_squared"].max()))
    for idx, row in tests.iterrows():
        y = top + idx * row_h
        label = row["feature_label"]
        metric = float(row["epsilon_squared"])
        text(draw, (55, y + 20), label, 23)
        draw.rounded_rectangle((left, y, right, y + 38), radius=12, fill="#EDF1F5")
        width = (right - left) * metric / max_value
        if metric >= 0.10:
            color = "#3C78A8"
        elif metric >= 0.01:
            color = "#79A8C7"
        else:
            color = "#B8CDD9"
        draw.rounded_rectangle((left, y, left + max(3, width), y + 38), radius=12, fill=color)
        text(draw, (left + max(12, width) + 14, y + 19), f"{metric:.4f}", 22, bold=True, anchor="lm")

    return save(image, "feature_effect_size_ranking.png")


def draw_context_comparison(profile):
    lookup = profile_lookup(profile)
    if TEMPORAL_MODE == "ordinal":
        panels = [
            ("Skill difficulty", "skill_difficulty", "mean", ""),
            ("Hard-skill ratio", "hard_skill_ratio", "mean", ""),
            ("Median order-ID gap", "median_time_gap_sec", "median", ""),
            ("Repeat-question ratio", "repeat_question_ratio", "mean", ""),
        ]
        heading = "Difficulty and interaction-order context by cluster"
        subtitle = "Order-ID gaps preserve sequence spacing but are not elapsed time."
    else:
        panels = [
            ("Skill difficulty", "skill_difficulty", "mean", ""),
            ("Hard-skill ratio", "hard_skill_ratio", "mean", ""),
            ("Median time gap", "median_time_gap_sec", "median", " min"),
            ("Active days", "active_days", "mean", " days"),
        ]
        heading = "Difficulty and temporal context by cluster"
        subtitle = "These variables are used for post-hoc interpretation, not as explicit CL4KT inputs."
    image = Image.new("RGB", (1650, 1050), "white")
    draw = ImageDraw.Draw(image)
    text(draw, (65, 42), heading, 42, bold=True)
    text(draw, (65, 98), subtitle, 23, fill="#596273")

    origins = [(70, 175), (850, 175), (70, 590), (850, 590)]
    for (title, feature, statistic, suffix), (ox, oy) in zip(panels, origins):
        draw.rounded_rectangle((ox, oy, ox + 720, oy + 350), radius=22, fill="#FAFBFC", outline="#D7DDE5", width=2)
        text(draw, (ox + 28, oy + 28), title, 29, bold=True)
        values = [value(lookup, c, feature, statistic) for c in STATE_ORDER]
        max_value = max(values) * 1.12
        baseline = oy + 295
        chart_top = oy + 95
        bar_w = 130
        for idx, (cluster, metric) in enumerate(zip(STATE_ORDER, values)):
            x = ox + 70 + idx * 210
            height = 185 * metric / max_value if max_value else 0
            draw.rounded_rectangle((x, baseline - height, x + bar_w, baseline), radius=12, fill=STATE_COLORS[cluster])
            if feature in {"skill_difficulty", "hard_skill_ratio", "repeat_question_ratio"}:
                label = f"{metric:.3f}"
            else:
                label = f"{metric:.1f}{suffix}"
            text(draw, (x + bar_w / 2, baseline - height - 18), label, 22, bold=True, anchor="ms")
            text(draw, (x + bar_w / 2, baseline + 28), STATE_SHORT[cluster], 22, anchor="mm")
        draw.line((ox + 45, baseline, ox + 675, baseline), fill="#AAB3BF", width=2)
    return save(image, "difficulty_time_comparison.png")


def write_summary_table(profile):
    lookup = profile_lookup(profile)
    counts = cluster_counts(profile)
    rows = []
    interpretations = {
        "Low": "Low history accuracy; longest incorrect streak",
        "Middle": "Intermediate performance; transitional profile",
        "High": "High sustained accuracy; longest correct streak",
    }
    for cluster in STATE_ORDER:
        rows.append(
            {
                "state": STATE_SHORT[cluster],
                "cluster": cluster,
                "n_timesteps": counts[cluster][0],
                "current_correctness": value(lookup, cluster, "correct"),
                "prefix_correctness": value(lookup, cluster, "previous_correct_rate"),
                "skill_difficulty": value(lookup, cluster, "skill_difficulty"),
                "hard_skill_ratio": value(lookup, cluster, "hard_skill_ratio"),
                "correct_streak": value(lookup, cluster, "longest_correct_streak"),
                "incorrect_streak": value(lookup, cluster, "longest_incorrect_streak"),
                (
                    "median_time_gap_min"
                    if TEMPORAL_MODE == "time"
                    else "median_order_id_gap"
                ): value(lookup, cluster, "median_time_gap_sec", "median"),
                "active_days": (
                    value(lookup, cluster, "active_days")
                    if TEMPORAL_MODE == "time"
                    else np.nan
                ),
                "interpretation": interpretations[STATE_SHORT[cluster]],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "cluster_interpretation_summary.csv", index=False)


def main():
    args = parse_args()
    root = Path(args.root)
    profile_path = root / "pedagogical_profile_long.csv"
    test_path = root / "pedagogical_statistical_tests.csv"
    if not profile_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing Experiment 3 inputs under {root}: "
            "pedagogical_profile_long.csv and pedagogical_statistical_tests.csv are required"
        )
    profile = pd.read_csv(profile_path)
    tests = pd.read_csv(test_path)
    configure(args, profile)
    if TEMPORAL_MODE == "ordinal":
        tests = tests[tests["feature"] != "active_days"].copy()
        tests.loc[
            tests["feature"] == "median_time_gap_sec", "feature_label"
        ] = "Median order-ID gap"
    paths = [
        draw_profile_cards(profile),
        draw_heatmap(profile),
        draw_effect_sizes(tests),
        draw_context_comparison(profile),
    ]
    write_summary_table(profile)
    for path in paths:
        print(path)
    print(ROOT / "cluster_interpretation_summary.csv")


if __name__ == "__main__":
    main()
