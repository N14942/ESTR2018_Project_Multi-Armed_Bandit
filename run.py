from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from Bandits_and_algorithms import (
    Multi_Armed_Bandits,
    AlgorithmType,
    create_algorithm
)

SEED = 42
REWARDS = [0.8, 0.6, 0.7, 0.4]
EPSILON = 0.2
WINDOW = 100

CHANGE_TIME_1 = 350
CHANGE_TIME_2 = 650
CHANGE_ARM_1 = {
    0: 0.55,
    2: 0.58
}

CHANGE_ARM_2 = {
    1: 0.56,
    3: 0.62
}

#unfinished
def single_run(seed: int, 
               rewards: list, 
               epsilon: float, 
               n: int, 
               algorithm_type,
               window_size):
    """set seed"""
    rng = np.random.default_rng(seed)

    B = Multi_Armed_Bandits(rng, rewards)
    E = create_algorithm(algorithm_type, B, epsilon, window_size)
    E.regret_history = []
    E.reward_history = []
    E.best_arm_history = []
    E.selected_arm_history = []
    total_reward = 0

    for i in range(n):
        B.change_reward(
        CHANGE_TIME,
        CHANGE_ARM
        )
        arm = E.choose_arm()
        E.selected_arm_history.append(arm)
        reward = E.update(arm)
        total_reward += reward
        E.regret_history.append(E.regret)
        E.reward_history.append(total_reward)
        E.best_arm_history.append(E.current)
        print(f"{algorithm_type.value} | Seed: {seed} | Round {i+1:>3} | "
          f"Estimated best arm: {E.current} | "
          f"Arm: {arm} | "
          f"Reward: {reward} | "
          f"Counts: {E.counts} | "
          f"Means: {np.round(E.means, 3)} | "
          f"Regret: {E.regret:.3f}")
    return E


def multiple_run(seeed, algorithm_type, rewards, epsilon, window_size, n_rounds=1000, n_runs=50):
    all_regrets = []
    all_rewards = []
    all_counts = []
    regret_history = np.zeros(n_rounds)
    reward_history = np.zeros(n_rounds)
    for i in range(n_runs):
        E = single_run(seeed + i,
            rewards,
            epsilon,
            n_rounds,
            algorithm_type,
            window_size)
        if i == 0:
            best_arm_history = E.best_arm_history
            selected_arm_history = E.selected_arm_history
        all_regrets.append(E.regret)
        all_rewards.append(np.sum(E.rewards))
        all_counts.append(E.counts.copy())
        regret_history += E.regret_history
        reward_history += E.reward_history
    return {
        "best_arm_history": best_arm_history,
        "selected_arm_history": selected_arm_history,
        "seed": seeed,
        "mean_regret_history": regret_history / n_runs,
        "mean_reward_history": reward_history / n_runs,
        "mean_regret": np.mean(all_regrets),
        "mean_reward": np.mean(all_rewards),
        "mean_counts": np.mean(all_counts, axis=0) #dictionary
    }

    
def plot_results(results):
    results_dir = Path(__file__).resolve().parent
    regret_fig, regret_ax = plt.subplots()
    reward_fig, reward_ax = plt.subplots()
    counts_fig, counts_ax = plt.subplots()
    best_fig, best_axes = plt.subplots(len(results), 1, sharex=True, squeeze=False,
                                      figsize=(10, 2.6 * len(results)))
    width = 0.8 / len(results)
    for i, (name, result) in enumerate(results.items()):
        first_round = len(result["mean_counts"])
        start = first_round - 1
        rounds = np.arange(first_round, len(result["mean_regret_history"]) + 1)
        regret_ax.plot(rounds, result["mean_regret_history"][start:], label=name)
        reward_ax.plot(rounds, result["mean_reward_history"][start:], label=name)
        arms = np.arange(len(result["mean_counts"]))
        best_ax = best_axes[i, 0]
        best = np.array(result["best_arm_history"][start:])
        best_ax.scatter(rounds, result["selected_arm_history"][start:],
                        color="teal", s=6, zorder=3, label="Selected arm")
        best_ax.step(rounds, best, where="post", color=f"C{i % 10}", label="Estimated best arm")
        changes = np.flatnonzero(np.diff(best)) + 1
        best_ax.scatter(rounds[changes], best[changes], s=12, color=f"C{i % 10}")
        best_ax.set(title=f"{name} (seed {result['seed']})", ylabel="Arm", yticks=arms)
        if CHANGE_ARM and CHANGE_TIME in rounds:
            best_ax.axvline(CHANGE_TIME, color="gray", linestyle="--", label=f"Rewards change at {CHANGE_TIME}")
            after_change = changes[rounds[changes] >= CHANGE_TIME]
            if len(after_change):
                j = after_change[0]
                best_ax.annotate(f"Round {rounds[j]}: {best[j-1]} to {best[j]}",
                                 (rounds[j], best[j]), xytext=(8, 8), textcoords="offset points")
        best_ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        if len(rounds) > 1:
            for ax in (best_ax, regret_ax, reward_ax):
                ax.set_xlim(rounds[0], rounds[-1])
        offset = (i - (len(results) - 1) / 2) * width
        counts_ax.bar(arms + offset, result["mean_counts"], width, label=name)

    best_axes[-1, 0].set_xlabel("Round")
    regret_ax.set(xlabel="Round", ylabel="Mean cumulative regret")
    reward_ax.set(xlabel="Round", ylabel="Mean cumulative reward")
    counts_ax.set(xlabel="Arm", ylabel="Mean selection count", xticks=arms)
    for fig, ax, filename in (
        (regret_fig, regret_ax, "cumulative_regret.png"),
        (reward_fig, reward_ax, "cumulative_reward.png"),
        (counts_fig, counts_ax, "arm_selection_counts.png")
    ):
        ax.legend()
        fig.tight_layout()
        path = results_dir / filename
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    best_fig.tight_layout()
    path = results_dir / "estimated_best_arm.png"
    best_fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.show()


#two different choosing ways~
epsilon_results = multiple_run(
    SEED,
    AlgorithmType.EPSILON_GREEDY,
    REWARDS,
    EPSILON,
    1000,
    20,
    WINDOW
)

ucb_results = multiple_run(
    SEED,
    AlgorithmType.UCB1,
    REWARDS,
    EPSILON,
    1000,
    20,
    WINDOW
)

print("Epsilon-Greedy Regret:", epsilon_results["mean_regret"])
print("UCB1 Regret:", ucb_results["mean_regret"])
plot_results({"Epsilon-Greedy": epsilon_results, "UCB1": ucb_results})
