import numpy as np
import matplotlib.pyplot as plt
from Bandits_and_algorithms import Multi_Armed_Bandits, EpsilonGreedy

SEED = 42
REWARDS = [0.8, 0.6, 0.7, 0.4]
EPSILON = 0.2

#unfinished
def single_run(seed: int, rewards: list, epsilon: float, n: int):
    """set seed"""
    seed = SEED
    rng = np.random.default_rng(seed)
    rewards = REWARDS
    epsilon = EPSILON

    B = Multi_Armed_Bandits(rng, rewards)
    E = EpsilonGreedy(epsilon, B)

