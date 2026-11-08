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

    for i in range(n):
        B.change_reward(
        CHANGE_TIME,
        CHANGE_ARM,
        i+1
        )
        arm = E.choose_arm()
        reward = E.update(arm)
        print(f"Round {i+1:>3} | "
          f"Arm: {arm} | "
          f"Reward: {reward} | "
          f"Counts: {E.counts} | "
          f"Means: {np.round(E.means, 3)} | "
          f"Regret: {E.regret:.3f}")
    return E


def multiple_run(algorithm_type, rewards, epsilon,n_rounds=1000, n_runs=50):
    all_regrets = []
    all_rewards = []
    all_counts = []
    for i in range(n_runs):
        E = single_run(seed=42 + i,
            rewards=rewards,
            epsilon=epsilon,
            n=n_rounds,
            algorithm_type=algorithm_type
        )
        all_regrets.append(E.regret)
        all_rewards.append(np.sum(E.rewards))
        all_counts.append(E.counts.copy())
    return {
        "mean_regret": np.mean(all_regrets),
        "mean_reward": np.mean(all_rewards),
        "mean_counts": np.mean(all_counts, axis=0) #dictionary
    }

    
#two different choosing ways~
epsilon_results = multiple_run(
    AlgorithmType.EPSILON_GREEDY,
    REWARDS,
    EPSILON,
    1000,
    20
)

ucb_results = multiple_run(
    AlgorithmType.UCB1,
    REWARDS,
    EPSILON,
    1000,
    20
)

print("Epsilon-Greedy Regret:", epsilon_results["mean_regret"])
print("UCB1 Regret:", ucb_results["mean_regret"])