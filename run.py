
from pathlib import Path
import csv
import ast

import numpy as np
import matplotlib.pyplot as plt

from Bandits_and_algorithms import (
    Multi_Armed_Bandits,
    AlgorithmType,
    create_algorithm
)


# ============================================================
# 1. Experiment configuration
# ============================================================

EPSILON = 0.2
WINDOW = 150

N_ROUNDS = 500
ANOMALY_ROUNDS = 100

# Use the SECOND anomalous seed in CSV
CSV_ROW_NUMBER = 2

# Require 10 consecutive correct predictions
STABLE_CORRECT_ROUNDS = 10

CSV_PATH = (
    Path(__file__).resolve().parent
    / "anomalous_seeds.csv"
)


# ============================================================
# 2. Read the SECOND anomalous seed from CSV
# ============================================================

def read_second_anomalous_seed():

    with open(
        CSV_PATH,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        # Skip the first data row
        first_row = next(reader, None)

        if first_row is None:
            raise ValueError("CSV file contains no data.")

        # Read the second data row
        row = next(reader, None)

        if row is None:
            raise ValueError(
                "CSV file contains fewer than two anomalous seeds."
            )

    seed = int(row["seed"])

    rewards = np.array(
        ast.literal_eval(row["expected_means"]),
        dtype=float
    )

    observed_means = np.array(
        ast.literal_eval(row["observed_means"]),
        dtype=float
    )

    true_best_arm = int(row["best_arm"])

    biased_arm = int(row["worst_competitor"])

    misrank_gap = float(
        observed_means[biased_arm]
        - observed_means[true_best_arm]
    )

    if len(rewards) != len(observed_means):
        raise ValueError(
            "Rewards and observed means have different lengths."
        )

    if not 0 <= true_best_arm < len(rewards):
        raise ValueError("Invalid true_best_arm in CSV.")

    if not 0 <= biased_arm < len(rewards):
        raise ValueError("Invalid biased_arm in CSV.")

    if true_best_arm != int(np.argmax(rewards)):
        raise ValueError(
            "CSV best_arm does not match true reward probabilities."
        )

    return (
        seed,
        rewards,
        observed_means,
        true_best_arm,
        biased_arm,
        misrank_gap
    )


# ============================================================
# 3. Generate shared reward tape
# ============================================================

def generate_reward_tape(seed, rewards, observed_means):

    rng = np.random.default_rng(seed)

    n_arms = len(rewards)

    tape = np.zeros(
        (N_ROUNDS, n_arms),
        dtype=int
    )

    # --------------------------------------------------------
    # First 100 rounds:
    # Reproduce the anomalous observations from CSV.
    # --------------------------------------------------------

    for arm in range(n_arms):

        success_count = int(round(
            observed_means[arm] * ANOMALY_ROUNDS
        ))

        if not 0 <= success_count <= ANOMALY_ROUNDS:
            raise ValueError(
                f"Invalid observed mean for arm {arm}."
            )

        outcomes = np.array(
            [1] * success_count
            + [0] * (ANOMALY_ROUNDS - success_count)
        )

        rng.shuffle(outcomes)

        tape[:ANOMALY_ROUNDS, arm] = outcomes

    # --------------------------------------------------------
    # Remaining 400 rounds:
    # Generate rewards using true probabilities.
    # --------------------------------------------------------

    random_values = rng.random(
        (N_ROUNDS - ANOMALY_ROUNDS, n_arms)
    )

    tape[ANOMALY_ROUNDS:] = (
        random_values < rewards
    ).astype(int)

    # --------------------------------------------------------
    # Verify the first 100 rounds.
    # --------------------------------------------------------

    reproduced_means = tape[:ANOMALY_ROUNDS].mean(axis=0)

    if not np.allclose(
        reproduced_means,
        observed_means,
        atol=1e-9
    ):

        raise AssertionError(
            "Generated anomalous means do not match CSV!"
        )

    return tape


# ============================================================
# 4. Run one algorithm using shared reward tape
# ============================================================

def anomalous_single_run(
    seed,
    rewards,
    algorithm_type,
    epsilon,
    window_size,
    reward_tape,
    algorithm_seed
):

    decision_rng = np.random.default_rng(
        algorithm_seed
    )

    B = Multi_Armed_Bandits(
        decision_rng,
        rewards.tolist()
    )

    assert B.best == int(np.argmax(rewards))

    # --------------------------------------------------------
    # Replace get_reward with shared reward tape.
    # --------------------------------------------------------

    reward_time = 0

    def get_reward(arm):

        nonlocal reward_time

        if reward_time >= N_ROUNDS:
            raise IndexError("Reward tape exhausted.")

        reward = int(
            reward_tape[reward_time, arm]
        )

        reward_time += 1

        return reward

    B.get_reward = get_reward

    # --------------------------------------------------------
    # Create algorithm
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Run 500 rounds
    # --------------------------------------------------------

    for i in range(N_ROUNDS):

        arm = E.choose_arm()

        E.selected_arm_history.append(arm)

        reward = E.update(arm)

        total_reward += reward

        E.regret_history.append(E.regret)

        E.reward_history.append(total_reward)

        # Algorithm's estimated best arm
        E.best_arm_history.append(E.current)

        # Actual best arm
        E.true_best_history.append(B.best)

        # Whether the estimated best arm is correct
        E.correct_history.append(
            E.current == B.best
        )

    # --------------------------------------------------------
    # Verify history lengths
    # --------------------------------------------------------

    assert len(E.selected_arm_history) == N_ROUNDS
    assert len(E.best_arm_history) == N_ROUNDS
    assert len(E.true_best_history) == N_ROUNDS
    assert len(E.correct_history) == N_ROUNDS
    assert len(E.regret_history) == N_ROUNDS

    assert reward_time == N_ROUNDS

    return E


# ============================================================
# 5. Find first STABLE recovery
# ============================================================

def find_first_stable_correct_round(
    correct_history,
    consecutive=STABLE_CORRECT_ROUNDS
):

    correct = np.asarray(
        correct_history,
        dtype=bool
    )

    if len(correct) != N_ROUNDS:
        raise ValueError(
            "correct_history length is incorrect."
        )

    # Start searching from round 101.
    start_index = ANOMALY_ROUNDS

    # Search for consecutive correct predictions.
    for end_index in range(
        start_index + consecutive - 1,
        len(correct)
    ):

        start = end_index - consecutive + 1

        window = correct[start:end_index + 1]

        if np.all(window):

            # Return the round when the 10th
            # consecutive correct prediction occurs.
            return end_index + 1

    # No stable recovery within 500 rounds.
    return None


# ============================================================
# 6. Run three algorithms on the SECOND seed
# ============================================================

def run_second_seed_experiment():

    (
        seed,
        rewards,
        observed_means,
        true_best_arm,
        biased_arm,
        misrank_gap
    ) = read_second_anomalous_seed()

    print("\n" + "=" * 70)

    print("SECOND ANOMALOUS SEED EXPERIMENT")

    print("=" * 70)

    print(f"CSV row: {CSV_ROW_NUMBER}")
    print(f"Seed: {seed}")
    print(f"True probabilities: {rewards}")

    print(
        f"Observed means (first 100 rounds): "
        f"{observed_means}"
    )

    print(f"True best arm: Arm {true_best_arm}")
    print(f"Biased arm: Arm {biased_arm}")
    print(f"Misranking gap: {misrank_gap:.3f}")

    print(
        f"Stable recovery requirement: "
        f"{STABLE_CORRECT_ROUNDS} consecutive correct rounds"
    )

    print("=" * 70)

    # Generate shared reward tape.
    reward_tape = generate_reward_tape(
        seed,
        rewards,
        observed_means
    )

    # Independent decision RNGs.
    seed_sequence = np.random.SeedSequence(seed)

    child_seeds = seed_sequence.spawn(3)

    algorithm_types = {
        "Epsilon-Greedy": AlgorithmType.EPSILON_GREEDY,
        "UCB1": AlgorithmType.UCB1,
        "SW-UCB": AlgorithmType.SW_UCB
    }

    results = {}

    for (name, algorithm_type), child_seed in zip(
        algorithm_types.items(),
        child_seeds
    ):

        algorithm_seed = int(
            child_seed.generate_state(1)[0]
        )

        E = anomalous_single_run(
            seed=seed,
            rewards=rewards,
            algorithm_type=algorithm_type,
            epsilon=EPSILON,
            window_size=WINDOW,
            reward_tape=reward_tape,
            algorithm_seed=algorithm_seed
        )

        # Find stable recovery round.
        recovery_round = find_first_stable_correct_round(
            E.correct_history,
            STABLE_CORRECT_ROUNDS
        )

        results[name] = {

            "seed": seed,

            "best_arm_history": E.best_arm_history,

            "selected_arm_history": E.selected_arm_history,

            "true_best_history": E.true_best_history,

            "correct_history": E.correct_history,

            "regret_history": np.array(
                E.regret_history
            ),

            "reward_history": np.array(
                E.reward_history
            ),

            "mean_regret": E.regret,

            "recovery_round": recovery_round
        }

        print(f"\n{name}")

        if recovery_round is not None:

            first_round = (
                recovery_round
                - STABLE_CORRECT_ROUNDS
                + 1
            )

            print(
                f"Stable recovery: Round {first_round} "
                f"to Round {recovery_round}"
            )

            print(
                f"Confirmation round: {recovery_round}"
            )

        else:

            print(
                "Stable recovery: Not achieved "
                "within 500 rounds."
            )

        print(
            "Final cumulative regret:",
            round(E.regret, 3)
        )

    return (
        results,
        seed,
        rewards,
        observed_means,
        true_best_arm,
        biased_arm,
        misrank_gap
    )


# ============================================================
# 7. Plot arm selection and stable recovery
# ============================================================

def plot_anomalous_behaviour(
    results,
    rewards,
    observed_means,
    best_arm,
    biased_arm,
    misrank_gap,
    epsilon,
    window_size
):

    algorithms = list(results.keys())

    fig, axes = plt.subplots(
        len(algorithms),
        1,
        figsize=(13, 11),
        sharex=True
    )

    if len(algorithms) == 1:
        axes = [axes]

    rounds = np.arange(
        1,
        N_ROUNDS + 1
    )

    # --------------------------------------------------------
    # Algorithm titles
    # --------------------------------------------------------

    def get_title(name):

        name_lower = str(name).lower()

        if "epsilon" in name_lower or "greedy" in name_lower:

            return f"ε-Greedy (ε = {epsilon})"

        elif "sw" in name_lower or "sliding" in name_lower:

            return f"SW-UCB (Window Size = {window_size})"

        elif "ucb" in name_lower:

            return "UCB1"

        return str(name)

    # --------------------------------------------------------
    # Draw each algorithm
    # --------------------------------------------------------

    for ax, name in zip(axes, algorithms):

        result = results[name]

        selected = np.asarray(
            result["selected_arm_history"]
        )

        predicted = np.asarray(
            result["best_arm_history"]
        )

        true_best = np.asarray(
            result["true_best_history"]
        )

        # Selected arm: scatter points.

        ax.scatter(
            rounds,
            selected,
            s=13,
            alpha=0.55,
            label="Selected arm",
            zorder=3
        )

        # Predicted best arm.

        ax.step(
            rounds,
            predicted,
            where="post",
            linewidth=1.8,
            label="Predicted best arm",
            zorder=2
        )

        # True best arm.

        ax.step(
            rounds,
            true_best,
            where="post",
            color="black",
            linestyle=":",
            linewidth=1.5,
            label="True best arm",
            zorder=1
        )

        # Boundary at round 100.

        ax.axvline(
            x=ANOMALY_ROUNDS,
            color="red",
            linestyle="--",
            linewidth=1.4,
            label="End of anomalous period"
        )

        # ----------------------------------------------------
        # Stable recovery star
        # ----------------------------------------------------

        recovery_round = result["recovery_round"]

        if recovery_round is not None:

            # Star is placed at the 10th consecutive
            # correct identification.

            ax.scatter(
                recovery_round,
                predicted[recovery_round - 1],
                marker="*",
                s=220,
                color="red",
                edgecolors="black",
                linewidths=0.7,
                zorder=5,
                label="Stable recovery (10 consecutive)"
            )

            # Mark the beginning of the 10-round streak.

            first_stable_round = (
                recovery_round
                - STABLE_CORRECT_ROUNDS
                + 1
            )

            ax.axvspan(
                first_stable_round,
                recovery_round,
                color="green",
                alpha=0.10,
                zorder=0
            )

            ax.annotate(
                f"10 consecutive correct\n"
                f"Rounds {first_stable_round}-{recovery_round}",
                xy=(
                    recovery_round,
                    predicted[recovery_round - 1]
                ),
                xytext=(-10, 25),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                arrowprops=dict(
                    arrowstyle="->",
                    color="red"
                )
            )

        else:

            ax.text(
                0.98,
                0.88,
                "No stable recovery",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color="red",
                fontsize=9
            )

        ax.set_title(
            get_title(name),
            fontsize=11
        )

        ax.set_ylabel("Arm")

        ax.set_yticks(
            range(len(rewards))
        )

        ax.set_ylim(
            -0.4,
            len(rewards) - 0.6
        )

        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Round")

    # --------------------------------------------------------
    # Explanation box
    # --------------------------------------------------------

    comparison_text = (

        f"True reward probabilities: "
        f"{np.round(rewards, 2).tolist()}\n"

        f"Observed means (initial 100 rounds): "
        f"{np.round(observed_means, 2).tolist()}\n"

        f"True best arm: Arm {best_arm} | "

        f"Initially overestimated arm: Arm {biased_arm} | "

        f"Misranking gap: {misrank_gap:.2f}\n\n"

        f"Stable recovery = "
        f"{STABLE_CORRECT_ROUNDS} consecutive correct predictions\n"

        "Symbols: ● Selected arm    "
        "━ Predicted best arm    "
        "⋯ True best arm    "
        "┆ End of anomalous period    "
        "★ Stable recovery"
    )

    fig.text(
        0.5,
        0.965,
        comparison_text,
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9
        )
    )

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------

    handles = []
    labels = []

    for ax in axes:

        current_handles, current_labels = (
            ax.get_legend_handles_labels()
        )

        for handle, label in zip(
            current_handles,
            current_labels
        ):

            if label not in labels:

                handles.append(handle)
                labels.append(label)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015)
    )

    fig.suptitle(
        "Algorithm Behaviour Under Anomalous Initial Observations",
        fontsize=14,
        y=0.995
    )

    fig.tight_layout(
        rect=[0.03, 0.07, 0.97, 0.83]
    )

    output_path = (
        Path(__file__).resolve().parent
        / "estimated_best_arm_anomalous_seed_2.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"\nFigure saved to: {output_path}")

    plt.show()


# ============================================================
# 8. Plot cumulative regret comparison
# ============================================================

def plot_anomalous_regret(results, seed):

    results_dir = Path(__file__).resolve().parent

    regret_fig, regret_ax = plt.subplots(
        figsize=(10, 6)
    )

    rounds = np.arange(
        1,
        N_ROUNDS + 1
    )

    for name, result in results.items():

        regret_ax.plot(
            rounds,
            result["regret_history"],
            linewidth=2,
            label=name
        )

    regret_ax.axvline(
        ANOMALY_ROUNDS,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="End of anomalous period (100)"
    )

    regret_ax.set(
        title=f"Cumulative Regret Comparison (Seed {seed})",
        xlabel="Round",
        ylabel="Cumulative regret"
    )

    regret_ax.grid(alpha=0.25)

    regret_ax.legend()

    regret_fig.tight_layout()

    path = (
        results_dir
        / "cumulative_regret_anomalous_seed_2.png"
    )

    regret_fig.savefig(
        path,
        dpi=150
    )

    print(f"Saved {path}")

    plt.show()


# ============================================================
# 9. Main
# ============================================================

if __name__ == "__main__":

    (
        results,
        seed,
        rewards,
        observed_means,
        best_arm,
        biased_arm,
        misrank_gap
    ) = run_second_seed_experiment()

    print("\n" + "=" * 65)

    print("FINAL EXPERIMENT CONFIGURATION")

    print("=" * 65)

    print(f"Seed: {seed}")
    print(f"True reward probabilities: {rewards}")
    print(f"Observed means: {observed_means}")
    print(f"True best arm: {best_arm}")
    print(f"Biased arm: {biased_arm}")
    print(f"Misranking gap: {misrank_gap:.3f}")

    print(
        f"Stable recovery requirement: "
        f"{STABLE_CORRECT_ROUNDS} consecutive rounds"
    )

    print("\nStable recovery comparison:")

    for name, result in results.items():

        recovery = result["recovery_round"]

        if recovery is None:

            print(f"{name}: No stable recovery")

        else:

            start = (
                recovery
                - STABLE_CORRECT_ROUNDS
                + 1
            )

            print(
                f"{name}: Rounds {start}-{recovery} "
                f"(confirmed at round {recovery})"
            )

    print("=" * 65)

    # Figure 1: Arm selection and stable recovery.

    plot_anomalous_behaviour(
        results=results,
        rewards=rewards,
        observed_means=observed_means,
        best_arm=best_arm,
        biased_arm=biased_arm,
        misrank_gap=misrank_gap,
        epsilon=EPSILON,
        window_size=WINDOW
    )

    # Figure 2: Cumulative regret.

    plot_anomalous_regret(
        results,
        seed
    )
