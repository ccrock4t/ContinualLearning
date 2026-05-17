# plot_activation_diagnostics.py

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy import stats

# ============================================================
# PyCharm-friendly settings
# ============================================================

RUN_DIR = "grid_runs"

SMOOTH_WINDOW = 100

SHOW_WORLD_SWITCHES = True
WORLD_SWITCH_WIDTH = 0.8
WORLD_SWITCH_ALPHA = 0.25

# Set to None to plot all methods.
# Or choose any subset:
METHODS_TO_PLOT = [
    "PPO",
    "PPO + CB",
# "PPO + L2"
# "PPO + L2 + CB"
]



# ============================================================
# Loading
# ============================================================

def parse_method(run_name: str) -> str:
    """
    Important: check longest prefixes first.
    """
    if run_name.startswith("ppo_l2_cb"):
        return "PPO + L2 + CB"
    if run_name.startswith("ppo_cb"):
        return "PPO + CB"
    if run_name.startswith("ppo_l2"):
        return "PPO + L2"
    if run_name.startswith("ppo"):
        return "PPO"
    return run_name

def parse_trial(run_name: str) -> int | None:
    """
    Extracts trial number from names like:
      ppo_trial_01
      ppo_cb_trial_01
      ppo_l2_cb_trial_07
    """
    m = re.search(r"_trial_(\d+)", run_name)
    if m is None:
        return None
    return int(m.group(1))

def filter_methods(df: pd.DataFrame, methods: list[str] | None) -> pd.DataFrame:
    if df.empty or not methods:
        return df

    keep = set(methods)
    filtered = df[df["method"].isin(keep)].copy()

    missing = sorted(keep - set(df["method"].unique()))
    if missing:
        print(f"Warning: requested methods not found: {missing}")
        print(f"Available methods: {sorted(df['method'].unique())}")

    if filtered.empty:
        raise ValueError(
            f"No rows left after filtering methods={methods}. "
            f"Available methods: {sorted(df['method'].unique())}"
        )

    return filtered

def load_activation_logs(run_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(run_dir, "*_activations.csv")))

    if not paths:
        raise FileNotFoundError(
            f"No activation CSV files found in {run_dir}. "
            f"Expected files like ppo_trial_01_activations.csv"
        )

    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)

        if "run_name" not in df.columns:
            name = os.path.basename(path).replace("_activations.csv", "")
            df["run_name"] = name

        df["method"] = df["run_name"].apply(parse_method)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_episode_logs(run_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(run_dir, "*_episodes.csv")))

    if not paths:
        return pd.DataFrame()

    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        run_name = os.path.basename(path).replace("_episodes.csv", "")
        df["run_name"] = run_name
        df["method"] = df["run_name"].apply(parse_method)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def load_world_change_logs(run_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(run_dir, "*_world_changes.csv")))

    if not paths:
        return pd.DataFrame()

    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        run_name = os.path.basename(path).replace("_world_changes.csv", "")
        df["run_name"] = run_name
        df["method"] = df["run_name"].apply(parse_method)
        df["trial"] = df["run_name"].apply(parse_trial)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def add_world_switch_lines(
    ax,
    world_changes: pd.DataFrame,
    methods: list[str] | None = None,
    reference_trial: int = 1,
):
    if not SHOW_WORLD_SWITCHES:
        return

    if world_changes is None or world_changes.empty:
        return

    wc = world_changes.copy()

    if methods:
        wc = wc[wc["method"].isin(methods)].copy()

    # Use one trial as the reference so the plot is not overloaded.
    # Your trials should switch at the same schedule if steps_per_world is fixed.
    if "trial" in wc.columns:
        wc = wc[wc["trial"] == reference_trial].copy()

    if wc.empty:
        return

    wc = wc.sort_values("env_step")

    drawn = False
    for _, row in wc.iterrows():
        ax.axvline(
            row["env_step"],
            color="black",
            linestyle="--",
            linewidth=WORLD_SWITCH_WIDTH,
            alpha=WORLD_SWITCH_ALPHA,
            label="World switch" if not drawn else None,
            zorder=0,
        )
        drawn = True
# ============================================================
# Statistical analysis helpers
# ============================================================

def cohen_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """
    Paired Cohen's dz: mean difference divided by SD of paired differences.
    Positive means x > y.
    """
    diff = x - y
    sd = diff.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return diff.mean() / sd


def paired_summary_tests(
    paired_df: pd.DataFrame,
    value_col_a: str,
    value_col_b: str,
    label: str,
):
    """
    Prints paired t-test, Wilcoxon signed-rank test, and effect size.
    """
    x = paired_df[value_col_a].to_numpy(dtype=float)
    y = paired_df[value_col_b].to_numpy(dtype=float)

    diff = x - y
    n = len(diff)

    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)
    print(f"n paired trials: {n}")
    print(f"{value_col_a} mean ± SD: {x.mean():.6f} ± {x.std(ddof=1):.6f}")
    print(f"{value_col_b} mean ± SD: {y.mean():.6f} ± {y.std(ddof=1):.6f}")
    print(f"paired difference mean ± SD: {diff.mean():.6f} ± {diff.std(ddof=1):.6f}")
    print(f"paired Cohen's dz: {cohen_d_paired(x, y):.6f}")

    if n >= 2:
        t_res = stats.ttest_rel(x, y)
        print(f"paired t-test: t = {t_res.statistic:.6f}, p = {t_res.pvalue:.6g}")
    else:
        print("paired t-test: skipped, need at least 2 paired trials")

    if n >= 2 and np.any(diff != 0):
        try:
            w_res = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            print(f"Wilcoxon signed-rank: W = {w_res.statistic:.6f}, p = {w_res.pvalue:.6g}")
        except ValueError as e:
            print(f"Wilcoxon signed-rank: skipped ({e})")
    else:
        print("Wilcoxon signed-rank: skipped, not enough nonzero paired differences")


def compute_run_level_activation_summaries(
    activations: pd.DataFrame,
    metric: str = "dormant_fraction",
    final_fraction: float = 0.20,
) -> pd.DataFrame:
    """
    Produces one row per run.

    Summaries:
      - final_metric: mean over the last final_fraction of logged env_steps
      - auc_metric: time-normalized area under curve
      - slope_metric_per_million_steps: linear slope per 1M env steps

    Layers are averaged first so each run/env_step contributes once.
    """
    df = activations.copy()

    if "trial" not in df.columns:
        df["trial"] = df["run_name"].apply(parse_trial)

    # Average across hidden layers at each time point.
    if "layer_idx" in df.columns:
        df = (
            df
            .groupby(["method", "run_name", "trial", "env_step"], as_index=False)[metric]
            .mean()
        )

    rows = []

    for (method, run_name, trial), sub in df.groupby(["method", "run_name", "trial"]):
        sub = sub.sort_values("env_step").copy()

        x = sub["env_step"].to_numpy(dtype=float)
        y = sub[metric].to_numpy(dtype=float)

        if len(x) < 2:
            continue

        # Final fraction of training, by env_step threshold.
        max_step = x.max()
        min_step = x.min()
        cutoff = max_step - final_fraction * (max_step - min_step)
        final_y = y[x >= cutoff]

        final_metric = float(np.mean(final_y))

        # Time-normalized AUC. This is the average dormant fraction over training.
        if x.max() > x.min():
            auc_metric = float(np.trapezoid(y, x) / (x.max() - x.min()))
        else:
            auc_metric = np.nan

        # Linear slope per 1M env steps for readability.
        x_million = x / 1_000_000.0
        slope, intercept, r_value, p_value, stderr = stats.linregress(x_million, y)

        rows.append({
            "method": method,
            "run_name": run_name,
            "trial": trial,
            f"final_{metric}": final_metric,
            f"auc_{metric}": auc_metric,
            f"slope_{metric}_per_million_steps": float(slope),
            f"slope_{metric}_p": float(p_value),
            f"slope_{metric}_r2": float(r_value ** 2),
            "n_timepoints": int(len(x)),
            "min_env_step": float(x.min()),
            "max_env_step": float(x.max()),
        })

    return pd.DataFrame(rows)


def make_paired_method_table(
    summaries: pd.DataFrame,
    method_a: str,
    method_b: str,
    value_col: str,
) -> pd.DataFrame:
    """
    Returns one paired row per trial, with columns named after methods.
    """
    sub = summaries[summaries["method"].isin([method_a, method_b])].copy()

    paired = (
        sub
        .pivot_table(
            index="trial",
            columns="method",
            values=value_col,
            aggfunc="mean",
        )
        .dropna(subset=[method_a, method_b])
        .reset_index()
    )

    paired = paired.rename(columns={
        method_a: f"{method_a}_{value_col}",
        method_b: f"{method_b}_{value_col}",
    })

    return paired


def analyze_dormant_fraction_statistics(
    activations: pd.DataFrame,
    run_dir: str,
    method_a: str = "PPO",
    method_b: str = "PPO + CB",
    final_fraction: float = 0.20,
):
    """
    Main statistical analysis for the activation dormant fraction.

    Prints:
      - paired final dormant-fraction test
      - paired AUC dormant-fraction test
      - paired slope test
      - one-sample slope tests within each method

    Saves:
      - activation_dormant_fraction_run_level_stats.csv
    """
    metric = "dormant_fraction"

    if metric not in activations.columns:
        print(f"Skipping dormant-fraction stats: missing column {metric}")
        return

    df = activations.copy()
    df["trial"] = df["run_name"].apply(parse_trial)

    summaries = compute_run_level_activation_summaries(
        df,
        metric=metric,
        final_fraction=final_fraction,
    )

    plot_dir = ensure_plot_dir(run_dir)
    out_path = os.path.join(plot_dir, "activation_dormant_fraction_run_level_stats.csv")
    summaries.to_csv(out_path, index=False)

    print("\n" + "#" * 80)
    print("Dormant-fraction statistical analyses")
    print("#" * 80)
    print(f"Saved run-level summaries to: {out_path}")
    print(f"Methods available: {sorted(summaries['method'].unique())}")
    print(f"Using final_fraction={final_fraction:.2f}; final metric = mean over last {100 * final_fraction:.1f}% of logged training.")

    if method_a not in set(summaries["method"]) or method_b not in set(summaries["method"]):
        print(f"Skipping paired tests: need both {method_a!r} and {method_b!r}.")
        return

    tests = [
        (
            f"final_{metric}",
            f"Final dormant fraction: {method_a} vs {method_b}",
        ),
        (
            f"auc_{metric}",
            f"Time-averaged dormant fraction / AUC: {method_a} vs {method_b}",
        ),
        (
            f"slope_{metric}_per_million_steps",
            f"Dormant-fraction slope per 1M env steps: {method_a} vs {method_b}",
        ),
    ]

    for value_col, label in tests:
        paired = make_paired_method_table(
            summaries=summaries,
            method_a=method_a,
            method_b=method_b,
            value_col=value_col,
        )

        a_col = f"{method_a}_{value_col}"
        b_col = f"{method_b}_{value_col}"

        paired_summary_tests(
            paired_df=paired,
            value_col_a=a_col,
            value_col_b=b_col,
            label=label,
        )

    # One-sample tests: does each method's dormant fraction increase over time?
    slope_col = f"slope_{metric}_per_million_steps"

    print("\n" + "=" * 80)
    print("Within-method trend tests: is dormant fraction slope different from zero?")
    print("=" * 80)

    for method, sub in summaries.groupby("method"):
        slopes = sub[slope_col].dropna().to_numpy(dtype=float)

        if len(slopes) < 2:
            print(f"{method}: skipped, need at least 2 runs")
            continue

        t_res = stats.ttest_1samp(slopes, popmean=0.0)

        print(
            f"{method}: "
            f"mean slope = {slopes.mean():.6f} dormant-fraction units / 1M steps, "
            f"SD = {slopes.std(ddof=1):.6f}, "
            f"t = {t_res.statistic:.6f}, "
            f"p = {t_res.pvalue:.6g}, "
            f"n = {len(slopes)}"
        )

def analyze_activation_metric_statistics(
    activations: pd.DataFrame,
    run_dir: str,
    metric: str,
    metric_label: str,
    method_a: str = "PPO",
    method_b: str = "PPO + CB",
    final_fraction: float = 0.20,
):
    """
    Generic statistical analysis for an activation diagnostic.

    Prints:
      - paired final metric test
      - paired AUC/time-averaged metric test
      - paired slope test
      - one-sample slope tests within each method

    Saves:
      - activation_<metric>_run_level_stats.csv
    """

    if metric not in activations.columns:
        print(f"Skipping {metric_label} stats: missing column {metric}")
        return

    df = activations.copy()
    df["trial"] = df["run_name"].apply(parse_trial)

    summaries = compute_run_level_activation_summaries(
        df,
        metric=metric,
        final_fraction=final_fraction,
    )

    plot_dir = ensure_plot_dir(run_dir)
    out_path = os.path.join(plot_dir, f"activation_{metric}_run_level_stats.csv")
    summaries.to_csv(out_path, index=False)

    print("\n" + "#" * 80)
    print(f"{metric_label} statistical analyses")
    print("#" * 80)
    print(f"Saved run-level summaries to: {out_path}")
    print(f"Methods available: {sorted(summaries['method'].unique())}")
    print(
        f"Using final_fraction={final_fraction:.2f}; "
        f"final metric = mean over last {100 * final_fraction:.1f}% of logged training."
    )

    if method_a not in set(summaries["method"]) or method_b not in set(summaries["method"]):
        print(f"Skipping paired tests: need both {method_a!r} and {method_b!r}.")
        return

    tests = [
        (
            f"final_{metric}",
            f"Final {metric_label}: {method_a} vs {method_b}",
        ),
        (
            f"auc_{metric}",
            f"Time-averaged {metric_label} / AUC: {method_a} vs {method_b}",
        ),
        (
            f"slope_{metric}_per_million_steps",
            f"{metric_label} slope per 1M env steps: {method_a} vs {method_b}",
        ),
    ]

    for value_col, label in tests:
        paired = make_paired_method_table(
            summaries=summaries,
            method_a=method_a,
            method_b=method_b,
            value_col=value_col,
        )

        a_col = f"{method_a}_{value_col}"
        b_col = f"{method_b}_{value_col}"

        paired_summary_tests(
            paired_df=paired,
            value_col_a=a_col,
            value_col_b=b_col,
            label=label,
        )

    slope_col = f"slope_{metric}_per_million_steps"

    print("\n" + "=" * 80)
    print(f"Within-method trend tests: is {metric_label} slope different from zero?")
    print("=" * 80)

    for method, sub in summaries.groupby("method"):
        slopes = sub[slope_col].dropna().to_numpy(dtype=float)

        if len(slopes) < 2:
            print(f"{method}: skipped, need at least 2 runs")
            continue

        t_res = stats.ttest_1samp(slopes, popmean=0.0)

        print(
            f"{method}: "
            f"mean slope = {slopes.mean():.6f} {metric_label} units / 1M steps, "
            f"SD = {slopes.std(ddof=1):.6f}, "
            f"t = {t_res.statistic:.6f}, "
            f"p = {t_res.pvalue:.6g}, "
            f"n = {len(slopes)}"
        )
# ============================================================
# Plot helpers
# ============================================================

def ensure_plot_dir(run_dir: str) -> str:
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def summarize_for_lineplot(
    df: pd.DataFrame,
    metric: str,
    x_col: str = "env_step",
    aggregate_layers: bool = True,
) -> pd.DataFrame:
    plot_df = df.copy()

    if aggregate_layers and "layer_idx" in plot_df.columns:
        plot_df = (
            plot_df
            .groupby(["method", "run_name", x_col], as_index=False)[metric]
            .mean()
        )

    summary = (
        plot_df
        .groupby(["method", x_col], as_index=False)[metric]
        .agg(["mean", "sem"])
        .reset_index()
    )

    summary["sem"] = summary["sem"].fillna(0.0)
    return summary


def plot_metric_by_method(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: str,
    x_col: str = "env_step",
    aggregate_layers: bool = True,
    world_changes: pd.DataFrame | None = None,
):
    if metric not in df.columns:
        print(f"Skipping {metric}: column not found.")
        return

    summary = summarize_for_lineplot(
        df=df,
        metric=metric,
        x_col=x_col,
        aggregate_layers=aggregate_layers,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for method, sub in summary.groupby("method"):
        sub = sub.sort_values(x_col)

        x = sub[x_col].to_numpy()
        y = sub["mean"].to_numpy()
        sem = sub["sem"].to_numpy()

        ax.plot(x, y, label=method)
        ax.fill_between(x, y - sem, y + sem, alpha=0.2)

    add_world_switch_lines(
        ax=ax,
        world_changes=world_changes,
        methods=list(df["method"].dropna().unique()),
        reference_trial=1,
    )

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.show()
    plt.close(fig)

    print(f"Saved {output_path}")



def plot_cb_reinitializations(
    df: pd.DataFrame,
    output_path: str,
    world_changes: pd.DataFrame | None = None,
):
    metric = "num_reinitialized_total"

    if metric not in df.columns:
        print("Skipping CB reinitializations: column not found.")
        return

    plot_df = (
        df
        .groupby(["method", "run_name", "env_step"], as_index=False)[metric]
        .max()
    )

    summary = (
        plot_df
        .groupby(["method", "env_step"], as_index=False)[metric]
        .agg(["mean", "sem"])
        .reset_index()
    )

    summary["sem"] = summary["sem"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    for method, sub in summary.groupby("method"):
        sub = sub.sort_values("env_step")

        x = sub["env_step"].to_numpy()
        y = sub["mean"].to_numpy()
        sem = sub["sem"].to_numpy()

        ax.plot(x, y, label=method)
        ax.fill_between(x, y - sem, y + sem, alpha=0.2)

    add_world_switch_lines(
        ax=ax,
        world_changes=world_changes,
        methods=list(df["method"].dropna().unique()),
        reference_trial=1,
    )

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Total neurons reinitialized")
    ax.set_title("Continual Backprop neuron replacements")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved {output_path}")


def plot_episode_metric(
    episodes_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: str,
    smooth_window: int = 100,
    world_changes: pd.DataFrame | None = None,
):
    if episodes_df.empty:
        print(f"Skipping {metric}: no episode logs found.")
        return

    if metric not in episodes_df.columns:
        print(f"Skipping {metric}: column not found.")
        return

    df = episodes_df.copy()
    df = df.sort_values(["run_name", "env_step"])

    df[f"{metric}_smooth"] = (
        df
        .groupby("run_name")[metric]
        .transform(lambda s: s.rolling(smooth_window, min_periods=1).mean())
    )

    summary = (
        df
        .groupby(["method", "env_step"], as_index=False)[f"{metric}_smooth"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    summary["sem"] = summary["sem"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    for method, sub in summary.groupby("method"):
        sub = sub.sort_values("env_step")

        x = sub["env_step"].to_numpy()
        y = sub["mean"].to_numpy()
        sem = sub["sem"].to_numpy()

        ax.plot(x, y, label=method)
        ax.fill_between(x, y - sem, y + sem, alpha=0.2)

    add_world_switch_lines(
        ax=ax,
        world_changes=world_changes,
        methods=list(episodes_df["method"].dropna().unique()),
        reference_trial=1,
    )

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}, rolling mean window={smooth_window}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved {output_path}")


# ============================================================
# Main plotting
# ============================================================

def make_plots(
    run_dir: str,
    smooth_window: int = 100,
    methods: list[str] | None = None,
):
    plot_dir = ensure_plot_dir(run_dir)

    activations = load_activation_logs(run_dir)
    episodes = load_episode_logs(run_dir)
    world_changes = load_world_change_logs(run_dir)

    activations = filter_methods(activations, methods)
    episodes = filter_methods(episodes, methods)

    if not world_changes.empty:
        world_changes = filter_methods(world_changes, methods)
        print(f"Loaded world switches: {len(world_changes)} rows")
        print(world_changes[["run_name", "method", "trial", "env_step"]].head())
    else:
        print("No world switch files found.")

    print(f"Plotting methods: {sorted(activations['method'].unique())}")

    # Activation diagnostic statistical analyses.
    analysis_metrics = [
        ("dormant_fraction", "Dormant fraction"),
        ("effective_rank", "Effective rank"),
        ("stable_rank_99", "Stable rank 99%"),
        ("mean_abs_activation", "Mean absolute activation"),
        ("std_activation", "Activation standard deviation"),
    ]

    for metric, metric_label in analysis_metrics:
        analyze_activation_metric_statistics(
            activations=activations,
            run_dir=run_dir,
            metric=metric,
            metric_label=metric_label,
            method_a="PPO",
            method_b="PPO + CB",
            final_fraction=0.20,
        )

    # Main paper-style diagnostics:
    # dormant units and representation diversity/rank.
    plot_metric_by_method(
        activations,
        metric="dormant_fraction",
        ylabel="Dormant fraction",
        output_path=os.path.join(plot_dir, "activation_dormant_fraction.pdf"),
        world_changes=world_changes,
    )

    plot_metric_by_method(
        activations,
        metric="effective_rank",
        ylabel="Effective rank",
        output_path=os.path.join(plot_dir, "activation_effective_rank.pdf"),
        world_changes=world_changes,
    )

    plot_metric_by_method(
        activations,
        metric="stable_rank_99",
        ylabel="Stable rank 99%",
        output_path=os.path.join(plot_dir, "activation_stable_rank_99.pdf"),
        world_changes=world_changes,
    )

    plot_metric_by_method(
        activations,
        metric="mean_abs_activation",
        ylabel="Mean absolute activation",
        output_path=os.path.join(plot_dir, "activation_mean_abs.pdf"),
        world_changes=world_changes,
    )

    plot_metric_by_method(
        activations,
        metric="std_activation",
        ylabel="Activation standard deviation",
        output_path=os.path.join(plot_dir, "activation_std.pdf"),
        world_changes=world_changes,
    )


    # Confirms CB is actually replacing neurons.
    # This requires plot_cb_reinitializations to accept world_changes.
    plot_cb_reinitializations(
        activations,
        output_path=os.path.join(plot_dir, "cb_reinitializations.pdf"),
        world_changes=world_changes,
    )

    # Behavioral performance plots from episode logs.
    # This requires plot_episode_metric to accept world_changes.
    plot_episode_metric(
        episodes,
        metric="avg_energy",
        ylabel="Average episode energy",
        output_path=os.path.join(plot_dir, "episode_avg_energy.pdf"),
        smooth_window=smooth_window,
        world_changes=world_changes,
    )

    plot_episode_metric(
        episodes,
        metric="reward",
        ylabel="Episode reward",
        output_path=os.path.join(plot_dir, "episode_reward.pdf"),
        smooth_window=smooth_window,
        world_changes=world_changes,
    )

    plot_episode_metric(
        episodes,
        metric="steps_alive",
        ylabel="Steps alive",
        output_path=os.path.join(plot_dir, "episode_steps_alive.pdf"),
        smooth_window=smooth_window,
        world_changes=world_changes,
    )

    print("\nDone.")
    print(f"Plots saved in: {plot_dir}")


def main():
    make_plots(
        run_dir=RUN_DIR,
        smooth_window=SMOOTH_WINDOW,
        methods=METHODS_TO_PLOT,
    )

if __name__ == "__main__":
    main()