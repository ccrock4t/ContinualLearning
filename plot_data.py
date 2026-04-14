import os
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = "runs"
METHODS = ["ppo", "ppo_l2", "ppo_l2_cb"]

SMOOTH_WINDOW = 20
SHOW_WORLD_SWITCHES = True


def load_episode_data(run_name, runs_dir=RUNS_DIR):
    path = os.path.join(runs_dir, f"{run_name}_episodes.csv")
    df = pd.read_csv(path)
    df = df.sort_values("env_step").reset_index(drop=True)

    df["reward_smooth"] = df["reward"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
    df["food_smooth"] = df["food_eaten"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
    return df


def load_world_change_data(run_name, runs_dir=RUNS_DIR):
    path = os.path.join(runs_dir, f"{run_name}_world_changes.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.sort_values("env_step").reset_index(drop=True)
    return df


def plot_metric(methods, metric_col, smooth_col=None, ylabel="", title="", runs_dir=RUNS_DIR):
    plt.figure(figsize=(12, 6))

    for run_name in methods:
        df = load_episode_data(run_name, runs_dir=runs_dir)

        if metric_col is not None:
            plt.plot(
                df["env_step"],
                df[metric_col],
                alpha=0.2,
                linewidth=1,
                label=f"{run_name} raw" if smooth_col is None else None,
            )

        if smooth_col is not None:
            plt.plot(
                df["env_step"],
                df[smooth_col],
                linewidth=2,
                label=run_name,
            )
        elif metric_col is not None:
            plt.plot(
                df["env_step"],
                df[metric_col],
                linewidth=2,
                label=run_name,
            )

    if SHOW_WORLD_SWITCHES:
        wc = load_world_change_data(methods[0], runs_dir=runs_dir)
        if wc is not None and len(wc) > 0:
            for i, (_, row) in enumerate(wc.iterrows()):
                plt.axvline(
                    row["env_step"],
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.6,
                    label="World switch" if i == 0 else None,
                )

    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_single_run_with_world_switches(run_name, runs_dir=RUNS_DIR):
    df = load_episode_data(run_name, runs_dir=runs_dir)
    wc = load_world_change_data(run_name, runs_dir=runs_dir)

    plt.figure(figsize=(12, 6))
    plt.plot(df["env_step"], df["reward"], alpha=0.2, linewidth=1, label="Episode return")
    plt.plot(df["env_step"], df["reward_smooth"], linewidth=2, label=f"Smoothed return ({SMOOTH_WINDOW})")

    if SHOW_WORLD_SWITCHES and wc is not None and len(wc) > 0:
        for _, row in wc.iterrows():
            plt.axvline(row["env_step"], linestyle="--", alpha=1.0)

    plt.xlabel("Environment steps")
    plt.ylabel("Episode return")
    plt.title(f"{run_name}: return over training")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_metric(
        METHODS,
        metric_col="reward",
        smooth_col="reward_smooth",
        ylabel="Episode return",
        title="Episode return vs environment steps",
    )

    plot_metric(
        METHODS,
        metric_col="food_eaten",
        smooth_col="food_smooth",
        ylabel="Food eaten per episode",
        title="Food eaten vs environment steps",
    )

    for run_name in METHODS:
        plot_single_run_with_world_switches(run_name)