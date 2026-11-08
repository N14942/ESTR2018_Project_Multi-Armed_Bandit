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
def single_run(seed: int, rewards: list, epsilon: float, algo, n: int):
    """set seed"""
    rng = np.random.default_rng(seed)

    B = Multi_Armed_Bandits(rng, rewards)
    E = create_algorithm(algo, B, epsilon)

    for i in range(n):
        arm = E.choose_arm()

        print({
            "round":  i + 1,
            "counts": E.counts.copy(),
            "means":  E.means.copy(),
            "regret": E.regret,
        })

single_run(SEED, REWARDS, EPSILON, TYPE, 10)