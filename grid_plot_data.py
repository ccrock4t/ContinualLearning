import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = "grid_runs"
METHODS = ["ppo", "ppo_cb"]   # or ["ppo", "ppo_l2", "ppo_cb", "ppo_l2_cb"]

NUM_TRIALS = 10
SMOOTH_WINDOW = 20
SHOW_WORLD_SWITCHES = True
SHOW_ERROR_BAND = True   # mean ± SEM


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


def get_method_trial_dfs(method, num_trials=NUM_TRIALS, runs_dir=RUNS_DIR):
    dfs = []
    missing = []

    for trial_idx in range(1, num_trials + 1):
        run_name = trial_run_name(method, trial_idx)
        path = os.path.join(runs_dir, f"{run_name}_episodes.csv")
        if os.path.exists(path):
            dfs.append(load_episode_data(run_name, runs_dir=runs_dir))
        else:
            missing.append(run_name)

    if missing:
        print(f"[WARN] Missing runs for {method}: {missing}")

    if len(dfs) == 0:
        raise FileNotFoundError(f"No trial files found for method '{method}' in {runs_dir}")

    return dfs


def interpolate_trials(dfs, value_col, grid_points=1000):
    min_x = min(df["env_step"].min() for df in dfs)
    max_x = max(df["env_step"].max() for df in dfs)

    common_x = np.linspace(min_x, max_x, grid_points)
    Y = []

    for df in dfs:
        x = df["env_step"].to_numpy()
        y = df[value_col].to_numpy()

        uniq_mask = np.concatenate([[True], x[1:] != x[:-1]])
        x = x[uniq_mask]
        y = y[uniq_mask]

        y_interp = np.interp(common_x, x, y)
        y_interp[common_x < x[0]] = np.nan
        y_interp[common_x > x[-1]] = np.nan

        Y.append(y_interp)

    Y = np.vstack(Y)
    return common_x, Y


def plot_metric_across_trials(
    methods,
    metric_col,
    smooth_col=None,
    ylabel="",
    title="",
    runs_dir=RUNS_DIR,
    num_trials=NUM_TRIALS,
    grid_points=1000,
):
    plt.figure(figsize=(12, 6))

    world_switch_drawn = False

    for method in methods:
        dfs = get_method_trial_dfs(method, num_trials=num_trials, runs_dir=runs_dir)
        value_col = smooth_col if smooth_col is not None else metric_col

        common_x, Y = interpolate_trials(dfs, value_col=value_col, grid_points=grid_points)

        mean_y = np.nanmean(Y, axis=0)
        std_y = np.nanstd(Y, axis=0)
        n_y = np.sum(~np.isnan(Y), axis=0)
        sem_y = std_y / np.sqrt(np.maximum(n_y, 1))

        label = f"{method} (moving average, window={SMOOTH_WINDOW})" if smooth_col is not None else method
        plt.plot(common_x, mean_y, linewidth=2.5, label=label)

        if SHOW_ERROR_BAND:
            plt.fill_between(
                common_x,
                mean_y - sem_y,
                mean_y + sem_y,
                alpha=0.2,
            )

        if SHOW_WORLD_SWITCHES and not world_switch_drawn:
            wc = load_world_change_data(trial_run_name(method, 1), runs_dir=runs_dir)
            if wc is not None and len(wc) > 0:
                for i, (_, row) in enumerate(wc.iterrows()):
                    plt.axvline(
                        row["env_step"],
                        color="black",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.5,
                        label="World switch" if i == 0 else None,
                    )
                world_switch_drawn = True

    plot_title = title
    if smooth_col is not None:
        plot_title += f" (moving average, window={SMOOTH_WINDOW})"

    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.legend()
    plt.tight_layout()

    safe_name = ylabel.lower().replace(" ", "_")
    plt.savefig(f"{safe_name}_avg_10_trials.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_single_method_average_with_world_switches(
    method,
    metric_col="avg_energy",
    smooth_col="avg_energy_smooth",
    runs_dir=RUNS_DIR,
    num_trials=NUM_TRIALS,
    grid_points=1000,
):
    dfs = get_method_trial_dfs(method, num_trials=num_trials, runs_dir=runs_dir)
    value_col = smooth_col if smooth_col is not None else metric_col

    common_x, Y = interpolate_trials(dfs, value_col=value_col, grid_points=grid_points)
    mean_y = np.nanmean(Y, axis=0)
    std_y = np.nanstd(Y, axis=0)
    n_y = np.sum(~np.isnan(Y), axis=0)
    sem_y = std_y / np.sqrt(np.maximum(n_y, 1))

    wc = load_world_change_data(trial_run_name(method, 1), runs_dir=runs_dir)

    plt.figure(figsize=(12, 6))

    label = f"{method} mean ({num_trials} trials, window={SMOOTH_WINDOW})" if smooth_col is not None else f"{method} mean ({num_trials} trials)"
    plt.plot(common_x, mean_y, linewidth=2.5, label=label)

    if SHOW_ERROR_BAND:
        plt.fill_between(common_x, mean_y - sem_y, mean_y + sem_y, alpha=0.2, label="± SEM")

    if SHOW_WORLD_SWITCHES and wc is not None and len(wc) > 0:
        for i, (_, row) in enumerate(wc.iterrows()):
            plt.axvline(
                row["env_step"],
                linestyle="--",
                color="black",
                alpha=0.5,
                label="World switch" if i == 0 else None,
            )

    plot_title = f"{method}: average over {num_trials} trials"
    if smooth_col is not None:
        plot_title += f" (moving average, window={SMOOTH_WINDOW})"

    plt.xlabel("Environment steps")
    plt.ylabel("Average energy")
    plt.title(plot_title)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_metric_across_trials(
        METHODS,
        metric_col="avg_energy",
        smooth_col="avg_energy_smooth",
        ylabel="Average energy per episode",
        title="Average energy vs environment steps",
    )

    for method in METHODS:
        plot_single_method_average_with_world_switches(method)