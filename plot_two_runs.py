import csv
import os
import matplotlib.pyplot as plt


# ============================================================
# Hardcoded filenames
# ============================================================
FILE1 = "runs/20260308_225902_episodes.csv"
FILE2 = "runs/20260308_230542_episodes.csv"

LABEL1 = "Run 1"
LABEL2 = "Run 2"

MOVING_AVG_WINDOW = 20


# ============================================================
# Helpers
# ============================================================
def load_episode_rewards(csv_path):
    episodes = []
    rewards = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))

    return episodes, rewards


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return values[:]

    averaged = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        averaged.append(sum(chunk) / len(chunk))
    return averaged


# ============================================================
# Main
# ============================================================
def main():
    if not os.path.exists(FILE1):
        raise FileNotFoundError(f"Could not find file: {FILE1}")

    if not os.path.exists(FILE2):
        raise FileNotFoundError(f"Could not find file: {FILE2}")

    ep1, rew1 = load_episode_rewards(FILE1)
    ep2, rew2 = load_episode_rewards(FILE2)

    ma1 = moving_average(rew1, MOVING_AVG_WINDOW)
    ma2 = moving_average(rew2, MOVING_AVG_WINDOW)

    plt.figure(figsize=(10, 6))

    # Raw rewards
    plt.plot(ep1, rew1, alpha=0.3, label=f"{LABEL1} raw")
    plt.plot(ep2, rew2, alpha=0.3, label=f"{LABEL2} raw")

    # Smoothed rewards
    plt.plot(ep1, ma1, linewidth=2, label=f"{LABEL1} MA({MOVING_AVG_WINDOW})")
    plt.plot(ep2, ma2, linewidth=2, label=f"{LABEL2} MA({MOVING_AVG_WINDOW})")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()