import os
import numpy as np
import pandas as pd
from scipy import stats

RUNS_DIR = "grid_runs"
METHOD_A = "ppo"
METHOD_B = "ppo_cb"
NUM_TRIALS = 10

SMOOTH_WINDOW = 20
FINAL_FRACTION = 0.10   # use last 10% of env steps for final performance
GRID_POINTS = 500
EARLY_FRACTION = 0.10
LATE_FRACTION = 0.10


def trial_run_name(method, trial_idx):
    return f"{method}_trial_{trial_idx:02d}"


def load_episode_data(run_name, runs_dir=RUNS_DIR):
    path = os.path.join(runs_dir, f"{run_name}_episodes.csv")
    df = pd.read_csv(path)
    df = df.sort_values("env_step").reset_index(drop=True)

    df["reward_smooth"] = df["reward"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
    df["final_energy_smooth"] = df["final_energy"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
    df["avg_energy_smooth"] = df["avg_energy"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
    return df


def load_world_change_data(run_name, runs_dir=RUNS_DIR):
    path = os.path.join(runs_dir, f"{run_name}_world_changes.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.sort_values("env_step").reset_index(drop=True)
    return df


def compute_auc(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan
    return float(np.trapezoid(y, x))


def compute_final_window_mean(df, value_col="avg_energy_smooth", final_fraction=FINAL_FRACTION):
    max_step = df["env_step"].max()
    threshold = max_step * (1.0 - final_fraction)
    tail = df[df["env_step"] >= threshold]
    if len(tail) == 0:
        return np.nan
    return float(tail[value_col].mean())


def compute_auc_metric(df, value_col="avg_energy_smooth"):
    return float(compute_auc(df["env_step"].to_numpy(), df[value_col].to_numpy()))


def compute_per_world_final_window_mean(df, wc=None, value_col="avg_energy_smooth", final_fraction=0.10):
    """
    For each world block, take the mean over the last `final_fraction` of that block,
    then average across world blocks.
    """
    if wc is None or len(wc) == 0:
        return compute_final_window_mean(df, value_col=value_col, final_fraction=final_fraction)

    switches = wc["env_step"].to_numpy()
    boundaries = [df["env_step"].min()] + list(switches) + [df["env_step"].max() + 1]

    per_block_values = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        block = df[(df["env_step"] >= start) & (df["env_step"] < end)]
        if len(block) == 0:
            continue

        block_start = block["env_step"].min()
        block_end = block["env_step"].max()
        threshold = block_start + (1.0 - final_fraction) * (block_end - block_start)

        tail = block[block["env_step"] >= threshold]
        if len(tail) > 0:
            per_block_values.append(float(tail[value_col].mean()))

    if len(per_block_values) == 0:
        return np.nan

    return float(np.mean(per_block_values))


def cohens_d_independent(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return np.nan
    return (np.mean(y) - np.mean(x)) / pooled_sd


def cohens_d_paired(x, y):
    diff = np.asarray(y, dtype=float) - np.asarray(x, dtype=float)
    if len(diff) < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return np.nan
    return np.mean(diff) / sd


def collect_trial_metrics(method, num_trials=NUM_TRIALS, runs_dir=RUNS_DIR):
    rows = []

    for trial_idx in range(1, num_trials + 1):
        run_name = trial_run_name(method, trial_idx)
        ep_path = os.path.join(runs_dir, f"{run_name}_episodes.csv")

        if not os.path.exists(ep_path):
            print(f"[WARN] Missing {ep_path}")
            continue

        df = load_episode_data(run_name, runs_dir=runs_dir)
        wc = load_world_change_data(run_name, runs_dir=runs_dir)

        row = {
            "trial_idx": trial_idx,
            "run_name": run_name,
            "final_window_mean": compute_final_window_mean(df, value_col="avg_energy_smooth"),
            "auc": compute_auc_metric(df, value_col="avg_energy_smooth"),
            "per_world_final_mean": compute_per_world_final_window_mean(
                df, wc=wc, value_col="avg_energy_smooth"
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("trial_idx").reset_index(drop=True)


def run_significance_tests(df_a, df_b, metric_name, method_a=METHOD_A, method_b=METHOD_B):
    merged = pd.merge(
        df_a[["trial_idx", metric_name]],
        df_b[["trial_idx", metric_name]],
        on="trial_idx",
        suffixes=(f"_{method_a}", f"_{method_b}")
    ).dropna()

    a = merged[f"{metric_name}_{method_a}"].to_numpy(dtype=float)
    b = merged[f"{metric_name}_{method_b}"].to_numpy(dtype=float)

    print("=" * 80)
    print(f"Metric: {metric_name}")
    print(f"{method_a}: mean={np.mean(a):.6f}, std={np.std(a, ddof=1):.6f}, n={len(a)}")
    print(f"{method_b}: mean={np.mean(b):.6f}, std={np.std(b, ddof=1):.6f}, n={len(b)}")
    print(f"Mean difference ({method_b} - {method_a}) = {np.mean(b - a):.6f}")
    print()

    # Paired tests: appropriate here because trial_i shares same world set across methods
    t_paired = stats.ttest_rel(b, a, nan_policy="omit")
    try:
        w_paired = stats.wilcoxon(b, a, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        w_paired = None

    # Unpaired tests: also reported for completeness
    t_welch = stats.ttest_ind(b, a, equal_var=False, nan_policy="omit")
    try:
        u_test = stats.mannwhitneyu(b, a, alternative="two-sided")
    except ValueError:
        u_test = None

    print("Paired t-test:")
    print(f"  statistic = {t_paired.statistic:.6f}, p = {t_paired.pvalue:.6g}")

    print("Wilcoxon signed-rank:")
    if w_paired is None:
        print("  could not compute")
    else:
        print(f"  statistic = {w_paired.statistic:.6f}, p = {w_paired.pvalue:.6g}")

    print("Welch t-test:")
    print(f"  statistic = {t_welch.statistic:.6f}, p = {t_welch.pvalue:.6g}")

    print("Mann-Whitney U:")
    if u_test is None:
        print("  could not compute")
    else:
        print(f"  statistic = {u_test.statistic:.6f}, p = {u_test.pvalue:.6g}")

    print()
    print(f"Cohen's d (paired)      = {cohens_d_paired(a, b):.6f}")
    print(f"Cohen's d (independent) = {cohens_d_independent(a, b):.6f}")
    print()

    print("Per-trial values:")
    print(merged.to_string(index=False))
    print("=" * 80)
    print()


# ============================================================
# Added: over-time gap growth analysis
# ============================================================

def interpolate_single_trial(df, value_col="avg_energy_smooth", grid_points=GRID_POINTS):
    x = df["env_step"].to_numpy(dtype=float)
    y = df[value_col].to_numpy(dtype=float)

    uniq_mask = np.concatenate([[True], x[1:] != x[:-1]])
    x = x[uniq_mask]
    y = y[uniq_mask]

    x_min = x.min()
    x_max = x.max()
    denom = max(x_max - x_min, 1.0)

    # Normalize training progress to [0, 1]
    x_norm = (x - x_min) / denom
    common_x = np.linspace(0.0, 1.0, grid_points)
    y_interp = np.interp(common_x, x_norm, y)

    return common_x, y_interp


def collect_paired_difference_curves(
    method_a=METHOD_A,
    method_b=METHOD_B,
    num_trials=NUM_TRIALS,
    runs_dir=RUNS_DIR,
    value_col="avg_energy_smooth",
    grid_points=GRID_POINTS,
):
    rows = []

    for trial_idx in range(1, num_trials + 1):
        run_a = trial_run_name(method_a, trial_idx)
        run_b = trial_run_name(method_b, trial_idx)

        path_a = os.path.join(runs_dir, f"{run_a}_episodes.csv")
        path_b = os.path.join(runs_dir, f"{run_b}_episodes.csv")

        if not (os.path.exists(path_a) and os.path.exists(path_b)):
            print(f"[WARN] Missing matched pair for trial {trial_idx}")
            continue

        df_a = load_episode_data(run_a, runs_dir=runs_dir)
        df_b = load_episode_data(run_b, runs_dir=runs_dir)

        x_a, y_a = interpolate_single_trial(df_a, value_col=value_col, grid_points=grid_points)
        x_b, y_b = interpolate_single_trial(df_b, value_col=value_col, grid_points=grid_points)

        diff = y_b - y_a

        early_mask = x_a <= EARLY_FRACTION
        late_mask = x_a >= (1.0 - LATE_FRACTION)

        rows.append({
            "trial_idx": trial_idx,
            "x": x_a,
            "diff": diff,
            "early_mean_diff": float(np.mean(diff[early_mask])),
            "late_mean_diff": float(np.mean(diff[late_mask])),
        })

    return rows


def fit_slope(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r_value),
        "p": float(p_value),
        "stderr": float(std_err),
    }


def run_gap_growth_tests(
    method_a=METHOD_A,
    method_b=METHOD_B,
    num_trials=NUM_TRIALS,
    runs_dir=RUNS_DIR,
    value_col="avg_energy_smooth",
    grid_points=GRID_POINTS,
):
    paired_curves = collect_paired_difference_curves(
        method_a=method_a,
        method_b=method_b,
        num_trials=num_trials,
        runs_dir=runs_dir,
        value_col=value_col,
        grid_points=grid_points,
    )

    if len(paired_curves) == 0:
        print("[WARN] No matched trial pairs found for gap growth tests.")
        return

    slope_rows = []
    slopes = []
    early_diffs = []
    late_diffs = []
    late_minus_early = []

    for row in paired_curves:
        fit = fit_slope(row["x"], row["diff"])
        lme = row["late_mean_diff"] - row["early_mean_diff"]

        slope_rows.append({
            "trial_idx": row["trial_idx"],
            "slope": fit["slope"],
            "intercept": fit["intercept"],
            "r": fit["r"],
            "per_trial_regression_p": fit["p"],
            "early_mean_diff": row["early_mean_diff"],
            "late_mean_diff": row["late_mean_diff"],
            "late_minus_early": lme,
        })

        slopes.append(fit["slope"])
        early_diffs.append(row["early_mean_diff"])
        late_diffs.append(row["late_mean_diff"])
        late_minus_early.append(lme)

    slope_df = pd.DataFrame(slope_rows).sort_values("trial_idx").reset_index(drop=True)

    slopes = np.asarray(slopes, dtype=float)
    early_diffs = np.asarray(early_diffs, dtype=float)
    late_diffs = np.asarray(late_diffs, dtype=float)
    late_minus_early = np.asarray(late_minus_early, dtype=float)

    print("=" * 80)
    print(f"Gap growth analysis: {method_b} - {method_a}")
    print()

    print("Per-trial slope of paired difference over normalized training time:")
    print(slope_df.to_string(index=False))
    print()

    print(f"Mean slope = {np.mean(slopes):.6f}, std={np.std(slopes, ddof=1):.6f}, n={len(slopes)}")

    slope_t = stats.ttest_1samp(slopes, popmean=0.0, alternative="greater")
    print("One-sample t-test for slope > 0:")
    print(f"  statistic = {slope_t.statistic:.6f}, p = {slope_t.pvalue:.6g}")

    try:
        slope_w = stats.wilcoxon(slopes, zero_method="wilcox", alternative="greater")
        print("Wilcoxon signed-rank for slope > 0:")
        print(f"  statistic = {slope_w.statistic:.6f}, p = {slope_w.pvalue:.6g}")
    except ValueError:
        print("Wilcoxon signed-rank for slope > 0:")
        print("  could not compute")

    print()
    print(f"Early mean paired difference = {np.mean(early_diffs):.6f} ± {np.std(early_diffs, ddof=1):.6f}")
    print(f"Late mean paired difference  = {np.mean(late_diffs):.6f} ± {np.std(late_diffs, ddof=1):.6f}")
    print(f"Late - Early                = {np.mean(late_minus_early):.6f} ± {np.std(late_minus_early, ddof=1):.6f}")
    print()

    early_late_t = stats.ttest_rel(late_diffs, early_diffs, alternative="greater")
    print("Paired t-test for late difference > early difference:")
    print(f"  statistic = {early_late_t.statistic:.6f}, p = {early_late_t.pvalue:.6g}")

    try:
        early_late_w = stats.wilcoxon(late_diffs, early_diffs, zero_method="wilcox", alternative="greater")
        print("Wilcoxon signed-rank for late difference > early difference:")
        print(f"  statistic = {early_late_w.statistic:.6f}, p = {early_late_w.pvalue:.6g}")
    except ValueError:
        print("Wilcoxon signed-rank for late difference > early difference:")
        print("  could not compute")

    print("=" * 80)
    print()


if __name__ == "__main__":
    df_ppo = collect_trial_metrics(METHOD_A)
    df_cb = collect_trial_metrics(METHOD_B)

    print("\nTrial-level summary table:\n")
    summary = pd.merge(
        df_ppo,
        df_cb,
        on="trial_idx",
        suffixes=(f"_{METHOD_A}", f"_{METHOD_B}")
    )
    print(summary.to_string(index=False))
    print()

    # Original overall tests
    for metric in ["final_window_mean", "auc", "per_world_final_mean"]:
        run_significance_tests(df_ppo, df_cb, metric_name=metric, method_a=METHOD_A, method_b=METHOD_B)

    # Added time-dependent gap-growth tests
    run_gap_growth_tests(
        method_a=METHOD_A,
        method_b=METHOD_B,
        value_col="avg_energy_smooth",
    )