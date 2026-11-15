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
TYPE = AlgorithmType.UCB1

CHANGE_TIME = 500
CHANGE_ARM = {
    0: 0.3,
    2: 0.9
}

#unfinished
def single_run(seed: int, rewards: list, epsilon: float, n: int, algorithm_type):
    """set seed"""
    rng = np.random.default_rng(seed)

    B = Multi_Armed_Bandits(rng, rewards)
    E = create_algorithm(algorithm_type, B, epsilon)
    E.regret_history = []
    E.reward_history = []
    total_reward = 0

    for i in range(n):
        B.change_reward(
        CHANGE_TIME,
        i+1,
        CHANGE_ARM
        )
        arm = E.choose_arm()
        reward = E.update(arm)
        total_reward += reward
        E.regret_history.append(E.regret)
        E.reward_history.append(total_reward)
        print(f"Round {i+1:>3} | "
          f"Arm: {arm} | "
          f"Reward: {reward} | "
          f"Counts: {E.counts} | "
          f"Means: {np.round(E.means, 3)} | "
          f"Regret: {E.regret:.3f}")
    return E


def multiple_run(seeed, algorithm_type, rewards, epsilon,n_rounds=1000, n_runs=50):
    all_regrets = []
    all_rewards = []
    all_counts = []
    regret_history = np.zeros(n_rounds)
    reward_history = np.zeros(n_rounds)
    for i in range(n_runs):
        E = single_run(seed=seeed + i,
            rewards=rewards,
            epsilon=epsilon,
            n=n_rounds,
            algorithm_type=algorithm_type
        )
        all_regrets.append(E.regret)
        all_rewards.append(np.sum(E.rewards))
        all_counts.append(E.counts.copy())
        regret_history += E.regret_history
        reward_history += E.reward_history
    return {
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
    width = 0.8 / len(results)
    for i, (name, result) in enumerate(results.items()):
        rounds = np.arange(1, len(result["mean_regret_history"]) + 1)
        regret_ax.plot(rounds, result["mean_regret_history"], label=name)
        reward_ax.plot(rounds, result["mean_reward_history"], label=name)
        arms = np.arange(len(result["mean_counts"]))
        offset = (i - (len(results) - 1) / 2) * width
        counts_ax.bar(arms + offset, result["mean_counts"], width, label=name)

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
    plt.show()


#two different choosing ways~
epsilon_results = multiple_run(
    SEED,
    AlgorithmType.EPSILON_GREEDY,
    REWARDS,
    EPSILON,
    1000,
    20
)

ucb_results = multiple_run(
    SEED,
    AlgorithmType.UCB1,
    REWARDS,
    EPSILON,
    1000,
    20
)

print("Epsilon-Greedy Regret:", epsilon_results["mean_regret"])
print("UCB1 Regret:", ucb_results["mean_regret"])
plot_results({"Epsilon-Greedy": epsilon_results, "UCB1": ucb_results})
