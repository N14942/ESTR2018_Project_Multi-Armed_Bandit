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
TYPE = AlgorithmType.EPSILON_GREEDY

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

single_run(SEED, REWARDS, EPSILON, TYPE, 20)