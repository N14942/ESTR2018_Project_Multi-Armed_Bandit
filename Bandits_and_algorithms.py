import numpy as np
from abc import ABC, abstractmethod

class Multi_Armed_Bandits:
    def __init__(self, rng, rewards:list):
        self.arms = np.array(rewards, dtype=float) #reward of each arm
        self.n = len(rewards) #number of arms
        self.best = int(np.argmax(self.arms)) #current arm with best reward
        self.rng = rng #seed

    def get_reward(self, arm: int):
        return int(self.rng.random() < self.arms[arm])

class AlgorithmType(Enum):
    EPSILON_GREEDY = "epsilon_greedy"
    UCB1 = "ucb1"

def create_algorithm(algorithm_type, bandits, epsilon=0.1):

    if algorithm_type == AlgorithmType.EPSILON_GREEDY:
        return EpsilonGreedy(epsilon, bandits)

    if algorithm_type == AlgorithmType.UCB1:
        return UCB1(bandits)


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

class EpsilonGreedy:
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
    def _init_(self, bandits: Multi_Armed_Bandits):
        super().__init__(bandits)
        #already set self.bandits self.t self.counts self.rewards self.means self.regret self.current self.rng

    def choose_arm(self) -> int:    #UCB1 = mean + square root(2lnt/Ni) and then choose the biggest one
        if self.t < self.bandits.n:
            untried = np.flatnonzero(self.counts == 0)
            return int(self.rng.choice(untried)) #when there are still some bandits haven't been tried
        else:
            self.means = self.rewards / self.counts
            used_t = 2* np.log(self.t)
            ucb_part2 = np.sqrt(used_t / self.counts)
            UCB_score = self.means + ucb_part2
            return int(np.argmax(UCB_score))

    def update(self, arm: int) -> int:
        reward = self.bandits.get_reward(arm)
        self.t += 1
        self.counts[arm] += 1
        self.rewards[arm] += reward
        
        self.regret += self.bandits.arms[self.bandits.best] - self.bandits.arms[arm]
        self.means[arm] = self.rewards[arm] / self.counts[arm]

        return reward
