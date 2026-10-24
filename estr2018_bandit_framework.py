import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Environment: Bernoulli multi-armed bandit
# ============================================================

def get_probabilities(t, scenario="stationary"):
    """
    Return the reward probability of each arm at round t.
    t starts from 0.

    stationary: reward probabilities never change
    changing: probabilities change halfway through the experiment
    """
    if scenario == "stationary":
        return np.array([0.8, 0.5, 0.3])

    if scenario == "changing":
        if t < 1000:
            return np.array([0.8, 0.5, 0.3])
        else:
            return np.array([0.2, 0.9, 0.4])

    raise ValueError(f"Unknown scenario: {scenario}")


def pull_arm(probability, rng):
    """Draw a Bernoulli reward: 1 with probability p, otherwise 0."""
    return int(rng.random() < probability)


# ============================================================
# 2. Policies: choose which arm to pull
# ============================================================

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms, dtype=int)
        self.reward_sums = np.zeros(n_arms, dtype=float)

    def select_arm(self, t, rng):
        # Explore: choose a random arm
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_arms))

        # Exploit: choose the arm with the highest empirical mean.
        # Untried arms are explored first to avoid undefined means.
        untried = np.flatnonzero(self.counts == 0)
        if len(untried) > 0:
            return int(rng.choice(untried))

        means = self.reward_sums / self.counts
        best = np.flatnonzero(np.isclose(means, means.max()))
        return int(rng.choice(best))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.reward_sums[arm] += reward


class UCB1:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms, dtype=int)
        self.reward_sums = np.zeros(n_arms, dtype=float)

    def select_arm(self, t, rng):
        # Try each arm once before using the UCB score.
        untried = np.flatnonzero(self.counts == 0)
        if len(untried) > 0:
            return int(untried[0])

        means = self.reward_sums / self.counts
        total_pulls = self.counts.sum()

        # UCB1 score: empirical mean + confidence bonus
        bonus = np.sqrt(2.0 * np.log(total_pulls) / self.counts)
        scores = means + bonus

        best = np.flatnonzero(np.isclose(scores, scores.max()))
        return int(rng.choice(best))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.reward_sums[arm] += reward


# ============================================================
# 3. Run one simulation
# ============================================================

def run_one(policy_name, scenario, horizon=2000, seed=0, epsilon=0.1):
    rng = np.random.default_rng(seed)

    # Number of arms is 3 in this example.
    n_arms = 3

    if policy_name == "epsilon-greedy":
        policy = EpsilonGreedy(n_arms, epsilon=epsilon)
    elif policy_name == "UCB1":
        policy = UCB1(n_arms)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    rewards = np.zeros(horizon)
    regrets = np.zeros(horizon)
    chosen_arms = np.zeros(horizon, dtype=int)

    for t in range(horizon):
        probabilities = get_probabilities(t, scenario)

        arm = policy.select_arm(t, rng)
        reward = pull_arm(probabilities[arm], rng)
        policy.update(arm, reward)

        # Expected regret for this round:
        # best true mean in this round minus the chosen arm's true mean.
        best_probability = probabilities.max()
        regrets[t] = best_probability - probabilities[arm]

        rewards[t] = reward
        chosen_arms[t] = arm

    return {
        "rewards": rewards,
        "regrets": regrets,
        "chosen_arms": chosen_arms,
    }


# ============================================================
# 4. Repeat experiments and average the results
# ============================================================

def run_experiment(scenario, horizon=2000, n_runs=100, epsilon=0.1):
    policies = ["epsilon-greedy", "UCB1"]
    results = {}

    for policy_name in policies:
        all_rewards = []
        all_regrets = []

        for run in range(n_runs):
            result = run_one(
                policy_name=policy_name,
                scenario=scenario,
                horizon=horizon,
                seed=1000 + run,
                epsilon=epsilon,
            )
            all_rewards.append(result["rewards"])
            all_regrets.append(result["regrets"])

        all_rewards = np.asarray(all_rewards)
        all_regrets = np.asarray(all_regrets)

        results[policy_name] = {
            "mean_cumulative_reward": np.cumsum(all_rewards, axis=1).mean(axis=0),
            "mean_cumulative_regret": np.cumsum(all_regrets, axis=1).mean(axis=0),
            "mean_reward_per_round": all_rewards.mean(),
        }

    return results


# ============================================================
# 5. Plot results
# ============================================================

def plot_results(results, scenario, horizon=2000):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, result in results.items():
        axes[0].plot(
            result["mean_cumulative_reward"],
            label=name
        )
        axes[1].plot(
            result["mean_cumulative_regret"],
            label=name
        )

    axes[0].set_title(f"Cumulative Reward ({scenario})")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Mean cumulative reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(f"Cumulative Regret ({scenario})")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Mean cumulative expected regret")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if scenario == "changing":
        for ax in axes:
            ax.axvline(1000, linestyle="--", label="Environment changes")
            ax.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Main: run both experiments
# ============================================================

if __name__ == "__main__":
    HORIZON = 2000
    N_RUNS = 100
    EPSILON = 0.1

    for scenario in ["stationary", "changing"]:
        print(f"\nScenario: {scenario}")
        results = run_experiment(
            scenario=scenario,
            horizon=HORIZON,
            n_runs=N_RUNS,
            epsilon=EPSILON,
        )

        for policy_name, result in results.items():
            print(
                f"{policy_name:16s} | "
                f"final cumulative reward = "
                f"{result['mean_cumulative_reward'][-1]:.2f} | "
                f"final cumulative regret = "
                f"{result['mean_cumulative_regret'][-1]:.2f} | "
                f"average reward per round = "
                f"{result['mean_reward_per_round']:.3f}"
            )

        plot_results(results, scenario, horizon=HORIZON)
