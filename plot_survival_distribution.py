import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# ============================================================
# Config
# ============================================================

RUNS_DIR = "grid_runs"
METHODS = ["ppo", "ppo_cb"]
NUM_TRIALS = 10

# If you trained fewer trials, either set NUM_TRIALS accordingly
# or leave this True to skip missing files.
ALLOW_MISSING = True

# Your environment caps lifespan at episode_horizon=200.
EPISODE_HORIZON = 200

# Optional: only analyze late-training episodes.
# Set to None to use all episodes.
# Example: LAST_N_EPISODES_PER_TRIAL = 100
LAST_N_EPISODES_PER_TRIAL = None

# Optional: restrict to certain worlds.
# Set to None to use all worlds.
# Example: WORLD_IDS = [0, 1, 2, 3]
WORLD_IDS = None

# Paper-ish style
FIG_WIDTH = 3.5
FIG_HEIGHT = 2.6
SAVE_DPI = 300
AXIS_LABEL_SIZE = 9
TITLE_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 8

OUT_DIR = "figures_lifespan"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def trial_run_name(method, trial_idx):
    return f"{method}_trial_{trial_idx:02d}"


def pretty_method_name(method):
    mapping = {
        "ppo": "PPO",
        "ppo_cb": "PPO+CB",
        "ppo_l2": r"PPO+$L_2$",
        "ppo_l2_cb": r"PPO+$L_2$+CB",
    }
    return mapping.get(method, method)


def load_one_run(method, trial_idx, runs_dir=RUNS_DIR):
    run_name = trial_run_name(method, trial_idx)
    path = os.path.join(runs_dir, f"{run_name}_episodes.csv")

    if not os.path.exists(path):
        if ALLOW_MISSING:
            print(f"[WARN] Missing: {path}")
            return None
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df = df.sort_values("env_step").reset_index(drop=True)

    required = ["steps_alive", "death_by_starvation", "world_id"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path} is missing required columns: {missing_cols}")

    if WORLD_IDS is not None:
        df = df[df["world_id"].isin(WORLD_IDS)].copy()

    if LAST_N_EPISODES_PER_TRIAL is not None:
        df = df.tail(LAST_N_EPISODES_PER_TRIAL).copy()

    df["method"] = method
    df["method_pretty"] = pretty_method_name(method)
    df["trial"] = trial_idx
    df["run_name"] = run_name
    df["survived_to_horizon"] = df["steps_alive"] >= EPISODE_HORIZON

    return df


def load_all_runs(methods=METHODS, num_trials=NUM_TRIALS, runs_dir=RUNS_DIR):
    dfs = []

    for method in methods:
        for trial_idx in range(1, num_trials + 1):
            df = load_one_run(method, trial_idx, runs_dir=runs_dir)
            if df is not None and len(df) > 0:
                dfs.append(df)

    if len(dfs) == 0:
        raise FileNotFoundError(
            f"No episode CSV files found in {runs_dir}. "
            f"Expected files like ppo_trial_01_episodes.csv."
        )

    return pd.concat(dfs, ignore_index=True)


def apply_paper_style(ax):
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.0, length=3, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def savefig(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=SAVE_DPI)
    print(f"[saved] {path}")


def print_summary(df):
    summary = (
        df.groupby("method_pretty")
        .agg(
            episodes=("steps_alive", "count"),
            mean_lifespan=("steps_alive", "mean"),
            median_lifespan=("steps_alive", "median"),
            std_lifespan=("steps_alive", "std"),
            starvation_rate=("death_by_starvation", "mean"),
            horizon_rate=("survived_to_horizon", "mean"),
        )
        .reset_index()
    )

    print("\n=== Lifespan summary ===")
    print(summary.to_string(index=False))
    print()


# ============================================================
# Plots
# ============================================================

def plot_violin_lifespan(df):
    labels = [pretty_method_name(m) for m in METHODS if pretty_method_name(m) in set(df["method_pretty"])]
    data = [df.loc[df["method_pretty"] == label, "steps_alive"].to_numpy() for label in labels]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    positions = np.arange(1, len(data) + 1)
    violin_colors = ["C0", "C1", "C2", "C3"]

    parts = ax.violinplot(
        data,
        positions=positions,
        showmeans=True,
        showmedians=True,
        showextrema=False,
    )

    # Color each violin by method.
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(violin_colors[i])
        body.set_edgecolor("black")
        body.set_alpha(0.45)
        body.set_linewidth(0.8)

    # Make mean/median summary lines black and explicit.
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linestyle("--")
    parts["cmeans"].set_linewidth(1.4)

    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linestyle("-")
    parts["cmedians"].set_linewidth(1.4)

    # Add individual trial means as black jittered points.
    rng = np.random.default_rng(0)
    for i, label in enumerate(labels, start=1):
        trial_means = (
            df[df["method_pretty"] == label]
            .groupby("trial")["steps_alive"]
            .mean()
            .to_numpy()
        )
        x = i + rng.normal(0.0, 0.035, size=len(trial_means))
        ax.scatter(
            x,
            trial_means,
            s=12,
            color="black",
            alpha=0.75,
            zorder=3,
        )

    horizon_line = ax.axhline(
        EPISODE_HORIZON,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.55,
        label="Episode horizon",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax.set_ylabel("Steps alive", fontsize=AXIS_LABEL_SIZE, fontweight="bold", labelpad=2)
    ax.set_ylim(0, EPISODE_HORIZON * 1.05)

    legend_elements = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.4, label="Median"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.4, label="Mean"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=4,
            label="Trial mean",
        ),
        Line2D(
            [0],
            [0],
            color=horizon_line.get_color(),
            linestyle=horizon_line.get_linestyle(),
            linewidth=horizon_line.get_linewidth(),
            alpha=horizon_line.get_alpha(),
            label="Episode horizon",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        fontsize=LEGEND_SIZE,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        handlelength=1.4,
        columnspacing=0.7,
    )

    apply_paper_style(ax)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(top=0.82)
    savefig(fig, "lifespan_violin.pdf")
    plt.show()

def plot_box_lifespan(df):
    labels = [pretty_method_name(m) for m in METHODS if pretty_method_name(m) in set(df["method_pretty"])]
    data = [df.loc[df["method_pretty"] == label, "steps_alive"].to_numpy() for label in labels]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    box = ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanline=True,
        widths=0.55,
        patch_artist=False,
    )

    horizon_line = ax.axhline(
        EPISODE_HORIZON,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.55,
        label="Episode horizon",
    )
    ax.set_ylabel("Steps alive", fontsize=AXIS_LABEL_SIZE, fontweight="bold", labelpad=2)
    ax.set_ylim(0, EPISODE_HORIZON * 1.05)


    median_color = box["medians"][0].get_color()
    mean_color = box["means"][0].get_color()

    legend_elements = [
        Line2D([0], [0], color=median_color, linewidth=1.5, label="Median"),
        Line2D([0], [0], color=mean_color, linestyle="--", linewidth=1.5, label="Mean"),
        Line2D(
            [0],
            [0],
            color=horizon_line.get_color(),
            linestyle=horizon_line.get_linestyle(),
            linewidth=horizon_line.get_linewidth(),
            alpha=horizon_line.get_alpha(),
            label="Episode horizon",
        )
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=LEGEND_SIZE,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        handlelength=1.6,
        columnspacing=0.9,
    )

    apply_paper_style(ax)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(top=0.82)
    savefig(fig, "lifespan_boxplot.pdf")
    plt.show()


def plot_hist_lifespan(df):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    bins = np.arange(0, EPISODE_HORIZON + 10, 10)

    for method in METHODS:
        label = pretty_method_name(method)
        vals = df.loc[df["method_pretty"] == label, "steps_alive"].to_numpy()
        if len(vals) == 0:
            continue
        ax.hist(
            vals,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=label,
        )

    ax.axvline(EPISODE_HORIZON, linestyle="--", linewidth=1.0, alpha=0.55, label="Episode horizon")
    ax.set_xlabel("Steps alive", fontsize=AXIS_LABEL_SIZE, fontweight="bold", labelpad=2)
    ax.set_ylabel("Density", fontsize=AXIS_LABEL_SIZE, fontweight="bold", labelpad=2)
    ax.set_xlim(0, EPISODE_HORIZON * 1.03)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False, loc="best")

    apply_paper_style(ax)
    fig.tight_layout(pad=0.4)
    savefig(fig, "lifespan_histogram.pdf")
    plt.show()


def plot_starvation_bar(df):
    """
    Extra companion plot: fraction of episodes ending by starvation.
    This is often clearer than lifespan when lifespan is capped by a horizon.
    """
    labels = [pretty_method_name(m) for m in METHODS if pretty_method_name(m) in set(df["method_pretty"])]
    means = []
    sems = []

    for label in labels:
        trial_rates = (
            df[df["method_pretty"] == label]
            .groupby("trial")["death_by_starvation"]
            .mean()
            .to_numpy()
        )
        means.append(np.mean(trial_rates))
        sems.append(np.std(trial_rates, ddof=1) / np.sqrt(len(trial_rates)) if len(trial_rates) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    x = np.arange(len(labels))

    ax.bar(x, means, yerr=sems, capsize=3, alpha=0.75)

    # Add individual trial dots.
    rng = np.random.default_rng(0)
    for i, label in enumerate(labels):
        trial_rates = (
            df[df["method_pretty"] == label]
            .groupby("trial")["death_by_starvation"]
            .mean()
            .to_numpy()
        )
        jitter = rng.normal(0.0, 0.035, size=len(trial_rates))
        ax.scatter(np.full(len(trial_rates), i) + jitter, trial_rates, s=14, alpha=0.75, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax.set_ylabel("Starvation rate", fontsize=AXIS_LABEL_SIZE, fontweight="bold", labelpad=2)
    ax.set_ylim(0, 1.0)

    apply_paper_style(ax)
    fig.tight_layout(pad=0.4)
    savefig(fig, "starvation_rate_bar.pdf")
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    df = load_all_runs()
    print_summary(df)

    plot_violin_lifespan(df)
    plot_box_lifespan(df)
    plot_hist_lifespan(df)
    plot_starvation_bar(df)
