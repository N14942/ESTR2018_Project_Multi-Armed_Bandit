import numpy as np

class Multi_Armed_Bandits:
    def __init__(self, rng, rewards:list):
        self.arms = np.array(rewards, dtype=float) #reward of each arm
        self.n = len(rewards) #number of arms
        self.best = int(np.argmax(self.arms)) #current arm with best reward
        self.rng = rng #seed

    def get_reward(self, arm: int):
        return int(self.rng.random() < self.arms[arm])
    

class EpsilonGreedy:
    def __init__(self, epsilon: float, bandits: Multi_Armed_Bandits):
        self.epsilon = epsilon
        self.bandits = bandits

        self.t = 0 #total times decision is made
        self.counts = np.zeros(self.bandits.n, dtype=int) #number of times choosing an arm
        self.rewards = np.zeros(self.bandits.n, dtype=int) #rewards get from each arm
        self.means = np.zeros(self.bandits.n, dtype=float) #detected average reward

        self.regret = 0 #regret(addition of (expected best mean - choosed expected mean))
        self.current = 0 #best arm detected

        self.rng = self.bandits.rng

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