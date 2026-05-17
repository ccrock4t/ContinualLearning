import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = "grid_runs"
METHODS = ["ppo", "ppo_cb"]

NUM_TRIALS = 10
SMOOTH_WINDOW = 1000
SHOW_WORLD_SWITCHES = True
SHOW_ERROR_BAND = True

# One-column paper style
FIG_WIDTH = 3.5
FIG_HEIGHT = 2.4
LINEWIDTH = 2.0
AXIS_LABEL_SIZE = 9
Y_LABEL_SIZE = 8
TITLE_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 7
WORLD_SWITCH_WIDTH = 0.8
SAVE_DPI = 300
SHOW_TITLE = False   # better for paper figures; describe in caption instead


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


def load_episode_data(run_name, runs_dir=RUNS_DIR):
    path = os.path.join(runs_dir, f"{run_name}_episodes.csv")
    df = pd.read_csv(path)
    df = df.sort_values("env_step").reset_index(drop=True)
    df["steps_alive_smooth"] = df["steps_alive"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
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

    return common_x, np.vstack(Y)


def apply_paper_style(ax):
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.0, length=3, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def plot_metric_across_trials(
    methods,
    metric_col,
    smooth_col=None,
    ylabel="Avg. energy",
    title="",
    runs_dir=RUNS_DIR,
    num_trials=NUM_TRIALS,
    grid_points=1000,
):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    world_switch_drawn = False

    for method in methods:
        dfs = get_method_trial_dfs(method, num_trials=num_trials, runs_dir=runs_dir)
        value_col = smooth_col if smooth_col is not None else metric_col

        common_x, Y = interpolate_trials(dfs, value_col=value_col, grid_points=grid_points)

        mean_y = np.nanmean(Y, axis=0)
        std_y = np.nanstd(Y, axis=0)
        n_y = np.sum(~np.isnan(Y), axis=0)
        sem_y = std_y / np.sqrt(np.maximum(n_y, 1))

        ax.plot(common_x, mean_y, linewidth=LINEWIDTH, label=pretty_method_name(method))

        if SHOW_ERROR_BAND:
            ax.fill_between(common_x, mean_y - sem_y, mean_y + sem_y, alpha=0.18)

        if SHOW_WORLD_SWITCHES and not world_switch_drawn:
            wc = load_world_change_data(trial_run_name(method, 1), runs_dir=runs_dir)
            if wc is not None and len(wc) > 0:
                for i, (_, row) in enumerate(wc.iterrows()):
                    ax.axvline(
                        row["env_step"],
                        color="black",
                        linestyle="--",
                        linewidth=WORLD_SWITCH_WIDTH,
                        alpha=0.35,
                        label="World switch" if i == 0 else None,
                    )
                world_switch_drawn = True

    ax.set_xlabel("Env. steps", fontsize=AXIS_LABEL_SIZE, fontweight="bold", labelpad=2)
    ax.set_ylabel(ylabel, fontsize=Y_LABEL_SIZE, fontweight="bold", labelpad=2)

    if SHOW_TITLE:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=4)

    leg = ax.legend(
        fontsize=LEGEND_SIZE,
        frameon=False,
        loc="best",
        handlelength=1.8,
        borderaxespad=0.2,
    )

    apply_paper_style(ax)

    fig.tight_layout(pad=0.4)

    safe_name = ylabel.lower().replace(" ", "_").replace(".", "")
    fig.savefig(f"{safe_name}_avg_10_trials.pdf", format="pdf", bbox_inches="tight", dpi=SAVE_DPI)
    plt.show()


if __name__ == "__main__":
    plot_metric_across_trials(
        METHODS,
        metric_col="avg_energy",
        smooth_col="avg_energy_smooth",
        ylabel="Avg. energy",
        title="Average energy vs environment steps",
    )
