import numpy as np
from enum import Enum
from collections import deque
from abc import ABC, abstractmethod

class Multi_Armed_Bandits:
    def __init__(self, rng, rewards:list):
        self.arms = np.array(rewards, dtype=float) #reward of each arm
        self.n = len(rewards) #number of arms
        self.best = int(np.argmax(self.arms)) #current arm with best reward
        self.rng = rng #seed

    def get_reward(self, arm: int):
        return int(self.rng.random() < self.arms[arm])

    def change_reward(self, change_time_1,change_time_2, current_time, change_arm_1 = None,change_arm_2 = None):
        #change arm in dict {0: 0.8, 3: 1, etc}
        if change_arm_1 is not None and current_time == change_time_1:
            for arm, new_reward in change_arm_1.items():
                self.arms[arm] = new_reward
            self.best = int(np.argmax(self.arms))
        if change_arm_2 is not None and current_time == change_time_2:
            for arm, new_reward in change_arm_2.items():
                self.arms[arm] = new_reward
            self.best = int(np.argmax(self.arms))
            
class AlgorithmType(Enum):
    EPSILON_GREEDY = "epsilon_greedy"
    UCB1 = "ucb1"
    SW_UCB = "sw_ucb"

def create_algorithm(algorithm_type, bandits, epsilon, window_size):

    if algorithm_type == AlgorithmType.EPSILON_GREEDY:
        return EpsilonGreedy(epsilon, bandits)

    if algorithm_type == AlgorithmType.UCB1:
        return UCB1(bandits)

    if algorithm_type == AlgorithmType.SW_UCB:
        return SlidingWindowUCB(bandits, window_size)


class Algorithm(ABC):
    def __init__(self, bandits: Multi_Armed_Bandits):
        self.bandits = bandits
        self.t = 0 #total times of decision is made
        self.counts = np.zeros(self.bandits.n, dtype=int) #number of times choosing an arm
        self.rewards = np.zeros(self.bandits.n, dtype=int) #rewards get from each arm
        self.means = np.zeros(self.bandits.n, dtype=float) #detected average reward

        self.regret = 0 #regret(addition of (expected best mean - choosed expected mean))
        self.current = 0 #best arm detected

        self.rng = self.bandits.rng

    @abstractmethod
    def choose_arm(self) -> int:
        pass

    @abstractmethod
    def update(self, arm: int) -> int:
        pass

class EpsilonGreedy(Algorithm):
    def __init__(self, epsilon: float, bandits: Multi_Armed_Bandits):
        super().__init__(bandits)
        self.epsilon = epsilon

    def choose_arm(self) -> int:
        if self.t < self.bandits.n:
            untried = np.flatnonzero(self.counts == 0)
            return int(self.rng.choice(untried))

        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.bandits.n))
        
        return self.current

    def update(self, arm) -> int:
        reward = self.bandits.get_reward(arm)
        self.t += 1
        self.counts[arm] += 1
        self.rewards[arm] += reward

        self.regret += self.bandits.arms[self.bandits.best] - self.bandits.arms[arm]
        self.means[arm] = self.rewards[arm] / self.counts[arm]

        best = np.flatnonzero(np.isclose(self.means, self.means.max()))
        self.current = int(self.rng.choice(best))

        return reward

class UCB1(Algorithm):
    def _get_statistics(self):
        return self.counts, self.rewards

    def _record_observation(self, arm, reward):
        pass

    def choose_arm(self) -> int:
        counts, rewards = self._get_statistics()
        untried = np.flatnonzero(counts == 0)

        if len(untried) > 0:
            return int(self.rng.choice(untried))
        
        means = rewards / counts
        ucb_scores = np.sqrt(2 * np.log(self.t) / counts) + means
        return int(np.argmax(ucb_scores))

    def update(self, arm: int) -> int:
        reward = self.bandits.get_reward(arm)

        self.t += 1
        self.counts[arm] += 1
        self.rewards[arm] += reward
        self.regret += self.bandits.arms[self.bandits.best] - self.bandits.arms[arm]
        
        self._record_observation(arm, reward)

        counts, rewards = self._get_statistics()
        valid = counts > 0

        self.means = np.zeros(self.bandits.n, dtype=float)
        self.means[valid] = rewards[valid] / counts[valid]
        estimated_means = np.full(self.bandits.n, -np.inf)
        estimated_means[valid] = self.means[valid]
        self.current = int(np.argmax(estimated_means))
        return reward

class SlidingWindowUCB(UCB1):
    def __init__(self, bandits, window_size=100):
        super().__init__(bandits)
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def _get_statistics(self):
        window_counts = np.zeros(self.bandits.n, dtype=int)
        window_rewards = np.zeros(self.bandits.n, dtype=float)

        for arm, reward in self.history:
            window_counts[arm] += 1
            window_rewards[arm] += reward

        return window_counts, window_rewards

    def _record_observation(self, arm, reward):
        self.history.append((arm, reward))

    def choose_arm(self) -> int:
        counts, rewards = self._get_statistics()
        untried = np.flatnonzero(counts == 0)
        if len(untried) > 0:
            return int(self.rng.choice(untried))

        means = rewards / counts
        confidence = np.sqrt(2 * np.log(self.window_size) / counts)
        ucb_scores = means + confidence
        return int(np.argmax(ucb_scores))