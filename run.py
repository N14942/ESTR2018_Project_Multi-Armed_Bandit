import numpy as np
import matplotlib.pyplot as plt
from Bandits_and_algorithms import Multi_Armed_Bandits, EpsilonGreedy

#unfinished
def single_run(seed: int, rewards: list, epsilon: float, n: int):
    """set seed"""
    seed = 42
    rng = np.random.default_rng(seed)
    rewards = [0.8, 0.6, 0.7, 0.4]
    epsilon = 0.2

    B = Multi_Armed_Bandits(rng, rewards)
    E = EpsilonGreedy(epsilon, B)

