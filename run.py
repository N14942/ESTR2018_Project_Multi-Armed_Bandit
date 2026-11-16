from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
import matplotlib.pyplot as plt
from Bandits_and_algorithms import (
    Multi_Armed_Bandits,
    AlgorithmType,
    create_algorithm
)

SEED = 42
REWARDS = [0.85, 0.20, 0.45, 0.35]
EPSILON = 0.2
WINDOW = 150

CHANGE_NUM = 1
CHANGE_TIME_1 = 350
CHANGE_TIME_2 = 350
CHANGE_ARM_1 = {0: 0.20, 1: 0.85}
CHANGE_ARM_2 = {0: 0.20, 1: 0.85}

def active_changes():
    if type(CHANGE_NUM) is not int or not 0 <= CHANGE_NUM <= 2:
        raise ValueError("CHANGE_NUM must be 0, 1 or 2")
    return (CHANGE_ARM_1 if CHANGE_NUM >= 1 else None,
            CHANGE_ARM_2 if CHANGE_NUM == 2 else None)


#unfinished
def single_run(seed: int, 
               rewards: list, 
               epsilon: float, 
               n: int, 
               algorithm_type,
               window_size):
    """set seed"""
    rng = np.random.default_rng(seed)

    B = Multi_Armed_Bandits(rng, rewards)
    E = create_algorithm(
        algorithm_type=algorithm_type,
        bandits=B,
        window_size=window_size,
        epsilon=epsilon
    )
    E.regret_history = []
    E.reward_history = []
    E.best_arm_history = []
    E.selected_arm_history = []
    E.true_best_history = []
    E.correct_history = []
    total_reward = 0
    active_arm_1, active_arm_2 = active_changes()

    for i in range(n):
        B.change_reward(
            CHANGE_TIME_1, CHANGE_TIME_2, i+1, active_arm_1, active_arm_2
        )
        arm = E.choose_arm()
        E.selected_arm_history.append(arm)
        reward = E.update(arm)
        total_reward += reward
        E.regret_history.append(E.regret)
        E.reward_history.append(total_reward)
        E.best_arm_history.append(E.current)
        E.true_best_history.append(B.best)
        E.correct_history.append(B.arms[E.current] == B.arms[B.best])
        print(f"{algorithm_type.value} | Seed: {seed} | Round {i+1:>3} | "
          f"Estimated best arm: {E.current} | "
          f"Arm: {arm} | "
          f"Reward: {reward} | "
          f"Counts: {E.counts} | "
          f"Means: {np.round(E.means, 3)} | "
          f"Regret: {E.regret:.3f}")
        
    return E


def multiple_run(seeed, algorithm_type, rewards, epsilon, window_size, n_rounds=1000, n_runs=25):
    all_regrets = []
    all_rewards = []
    all_counts = []
    change_times = sorted({t for t, updates in zip((CHANGE_TIME_1, CHANGE_TIME_2), active_changes())
                           if updates and 1 <= t <= n_rounds})
    boundaries = sorted({1, *change_times, n_rounds + 1})
    phase_counts = {(first, end - 1): np.zeros(len(rewards), dtype=int)
                    for first, end in zip(boundaries, boundaries[1:])}
    regret_history = np.zeros(n_rounds)
    reward_history = np.zeros(n_rounds)
    for i in range(n_runs):
        E = single_run(seeed + i,
            rewards,
            epsilon,
            n_rounds,
            algorithm_type,
            window_size)
        if i == 0:
            best_arm_history = E.best_arm_history
            selected_arm_history = E.selected_arm_history
            true_best_history = E.true_best_history
            correct_history = E.correct_history
        all_regrets.append(E.regret)
        all_rewards.append(E.reward_history[-1])
        all_counts.append(E.counts.copy())
        for (first, last), counts in phase_counts.items():
            counts += np.bincount(E.selected_arm_history[first - 1:last], minlength=len(rewards))
        regret_history += E.regret_history
        reward_history += E.reward_history
    return {
        "best_arm_history": best_arm_history,
        "selected_arm_history": selected_arm_history,
        "true_best_history": true_best_history,
        "correct_history": correct_history,
        "seed": seeed,
        "change_times": change_times,
        "mean_phase_counts": {phase: counts / n_runs for phase, counts in phase_counts.items()},
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
    phases = sorted({phase for result in results.values() for phase in result["mean_phase_counts"]})
    counts_fig, counts_axes = plt.subplots(1, len(phases), sharey=True, squeeze=False,
                                          figsize=(4.5 * len(phases), 4))
    phase_axes = dict(zip(phases, counts_axes[0]))
    best_fig, best_axes = plt.subplots(len(results), 1, sharex=True, squeeze=False,
                                      figsize=(10, 2.6 * len(results)))
    width = 0.8 / len(results)
    for i, (name, result) in enumerate(results.items()):
        start = 0
        rounds = np.arange(1, len(result["mean_regret_history"]) + 1)
        regret_ax.plot(rounds, result["mean_regret_history"][start:], label=name)
        reward_ax.plot(rounds, result["mean_reward_history"][start:], label=name)
        arms = np.arange(len(result["mean_counts"]))
        best_ax = best_axes[i, 0]
        best = np.array(result["best_arm_history"][start:])
        best_ax.scatter(rounds, result["selected_arm_history"][start:],
                        color="teal", s=6, zorder=3, label="Selected arm")
        best_ax.step(rounds, best, where="post", color=f"C{i % 10}", label="Estimated best arm")
        best_ax.step(rounds, result["true_best_history"][start:], where="post",
                     color="black", linestyle="--", zorder=4, label="True best arm")
        changes = np.flatnonzero(np.diff(best)) + 1
        best_ax.scatter(rounds[changes], best[changes], s=12, color=f"C{i % 10}")
        best_ax.set(title=f"{name} (seed {result['seed']})", ylabel="Arm", yticks=arms)
        change_times = result["change_times"]
        for change, end in zip(change_times, change_times[1:] + [rounds[-1] + 1]):
            hits = np.flatnonzero((rounds >= change) & (rounds < end) & result["correct_history"][start:])
            midpoint = (change + end - 1) / 2
            text_at = (midpoint, 0.97)
            if len(hits):
                j = hits[0]
                best_ax.scatter(rounds[j], best[j], marker="*", s=100, color="black", zorder=5)
                best_ax.annotate(f"Arm {best[j]}: round {rounds[j]} (+{rounds[j] - change})",
                                 xy=(rounds[j], best[j]), xytext=text_at,
                                 textcoords=best_ax.get_xaxis_transform(), ha="center", va="top",
                                 fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))
            else:
                best_ax.text(*text_at, f"No correct estimate by {end - 1}",
                             transform=best_ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
            best_ax.axvline(change, color="gray", linestyle="--", label=f"Rewards change at {change}")
        best_ax.set_ylim(-0.2, arms[-1] + 0.85)
        best_ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        if len(rounds) > 1:
            for ax in (best_ax, regret_ax, reward_ax):
                ax.set_xlim(rounds[0], rounds[-1])
        offset = (i - (len(results) - 1) / 2) * width
        for phase, counts in result["mean_phase_counts"].items():
            phase_axes[phase].bar(arms + offset, counts, width, label=name)

    best_axes[-1, 0].set_xlabel("Round")
    regret_ax.set(xlabel="Round", ylabel="Mean cumulative regret")
    reward_ax.set(xlabel="Round", ylabel="Mean cumulative reward")
    for (first, last), ax in phase_axes.items():
        ax.set(title=f"Rounds {first}-{last}", xlabel="Arm", xticks=arms)
    counts_axes[0, 0].set_ylabel("Mean selection count")
    for fig, ax, filename in (
        (regret_fig, regret_ax, "cumulative_regret.png"),
        (reward_fig, reward_ax, "cumulative_reward.png"),
        (counts_fig, counts_axes[0, 0], "arm_selection_counts.png")
    ):
        ax.legend()
        fig.tight_layout()
        path = results_dir / filename
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    best_fig.tight_layout()
    path = results_dir / "estimated_best_arm.png"
    with NamedTemporaryFile(dir=results_dir, suffix=".png", delete=False) as image:
        temp_path = Path(image.name)
    try:
        best_fig.savefig(temp_path, dpi=150)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"Saved {path}")
    plt.show()


#two different choosing ways~
epsilon_results = multiple_run(
    SEED,
    AlgorithmType.EPSILON_GREEDY,
    REWARDS,
    EPSILON,
    WINDOW
)

ucb_results = multiple_run(
    SEED,
    AlgorithmType.UCB1,
    REWARDS,
    EPSILON,
    WINDOW
)

sw_ucb_results = multiple_run(
    SEED,
    AlgorithmType.SW_UCB,
    REWARDS,
    EPSILON,
    WINDOW
)

print("Epsilon-Greedy Regret:", epsilon_results["mean_regret"])
print("UCB1 Regret:", ucb_results["mean_regret"])
print("sw_ucb Regret:", sw_ucb_results["mean_regret"])
plot_results({
    "Epsilon-Greedy": epsilon_results,
    "UCB1": ucb_results,
    "SW-UCB": sw_ucb_results
})
