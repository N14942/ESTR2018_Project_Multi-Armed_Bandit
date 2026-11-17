
# seed_screening.py

import csv
import numpy as np


# ==========================================
# Experiment settings
# ==========================================

SAMPLES_PER_ARM = 100
DEVIATION_THRESHOLD = 0.15

START_SEED = 0
NUMBER_OF_SEEDS = 1000000

OUTPUT_FILE = "anomalous_seeds.csv"


# ==========================================
# Random number generation
# ==========================================

def make_random_generators(seed: int, n_rounds: int, n_arms: int):
    """
    Create independent random streams for:
    1. Algorithm decisions
    2. Environment rewards

    The reward random stream is shared by screening and algorithm runs.
    """

    seed_sequence = np.random.SeedSequence(seed)
    algorithm_seed, reward_seed = seed_sequence.spawn(2)

    algorithm_rng = np.random.default_rng(algorithm_seed)
    reward_rng = np.random.default_rng(reward_seed)

    # Uniform random numbers, shape: (rounds, arms)
    reward_uniforms = reward_rng.random((n_rounds, n_arms))

    return algorithm_rng, reward_uniforms


# ==========================================
# Seed screening
# ==========================================

def inspect_seed(
    seed: int,
    rewards: list,
    samples_per_arm: int = SAMPLES_PER_ARM,
    threshold: float = DEVIATION_THRESHOLD
):
    """
    Check whether a seed causes a severe empirical misranking.

    Misranking condition:
        max(other arm empirical means)
        - best arm empirical mean >= threshold
    """

    rewards = np.asarray(rewards, dtype=float)
    n_arms = len(rewards)

    if n_arms < 2:
        raise ValueError("At least two arms are required.")

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    best_arm = int(np.argmax(rewards))

    # Generate the same reward random sequence that will be used
    # by the algorithm during its first samples_per_arm rounds.
    _, reward_uniforms = make_random_generators(
        seed,
        samples_per_arm,
        n_arms
    )

    # Potential Bernoulli rewards for every arm at every round.
    reward_tape = (
        reward_uniforms < rewards[np.newaxis, :]
    ).astype(int)

    # Each column represents one arm's 100 potential observations.
    observed_means = reward_tape.mean(axis=0)

    other_arms = [
        arm for arm in range(n_arms)
        if arm != best_arm
    ]

    worst_competitor = max(
        other_arms,
        key=lambda arm: observed_means[arm]
    )

    misrank_gap = (
        observed_means[worst_competitor]
        - observed_means[best_arm]
    )

    is_misranked = misrank_gap >= threshold

    return {
        "seed": seed,
        "is_anomalous": bool(is_misranked),
        "best_arm": best_arm,
        "worst_competitor": worst_competitor,
        "expected_means": rewards.tolist(),
        "observed_means": observed_means.tolist(),
        "best_arm_observed_mean": float(observed_means[best_arm]),
        "competitor_observed_mean": float(
            observed_means[worst_competitor]
        ),
        "misrank_gap": float(misrank_gap)
    }


def screen_seeds(
    rewards: list,
    start_seed: int = START_SEED,
    number_of_seeds: int = NUMBER_OF_SEEDS,
    samples_per_arm: int = SAMPLES_PER_ARM,
    threshold: float = DEVIATION_THRESHOLD
):
    """
    Search a range of seeds and return all misranked seeds.
    """

    anomalous_seeds = []
    all_results = []

    for seed in range(start_seed, start_seed + number_of_seeds):

        result = inspect_seed(
            seed=seed,
            rewards=rewards,
            samples_per_arm=samples_per_arm,
            threshold=threshold
        )

        all_results.append(result)

        if result["is_anomalous"]:
            anomalous_seeds.append(seed)

    return anomalous_seeds, all_results


# ==========================================
# Save results
# ==========================================

def save_anomalies(results, filename=OUTPUT_FILE):
    """
    Save anomalous seeds and their empirical means to CSV.
    """

    anomalous_results = [
        result for result in results
        if result["is_anomalous"]
    ]

    with open(filename, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "seed",
            "best_arm",
            "worst_competitor",
            "best_arm_observed_mean",
            "competitor_observed_mean",
            "misrank_gap",
            "expected_means",
            "observed_means"
        ])

        for result in anomalous_results:

            writer.writerow([
                result["seed"],
                result["best_arm"],
                result["worst_competitor"],
                result["best_arm_observed_mean"],
                result["competitor_observed_mean"],
                result["misrank_gap"],
                result["expected_means"],
                result["observed_means"]
            ])

    print(f"Saved {len(anomalous_results)} anomalous seeds to {filename}")


# ==========================================
# Run screening independently
# ==========================================

if __name__ == "__main__":

    REWARDS = [0.75, 0.20, 0.65, 0.45]

    anomalous_seeds, results = screen_seeds(
        rewards=REWARDS,
        start_seed=START_SEED,
        number_of_seeds=NUMBER_OF_SEEDS,
        samples_per_arm=SAMPLES_PER_ARM,
        threshold=DEVIATION_THRESHOLD
    )

    print("\n========== Seed Screening ==========")
    print(f"Rewards: {REWARDS}")
    print(f"Samples per arm: {SAMPLES_PER_ARM}")
    print(f"Misrank threshold: {DEVIATION_THRESHOLD}")
    print(f"Seeds checked: {NUMBER_OF_SEEDS}")
    print(f"Anomalous seeds found: {len(anomalous_seeds)}")

    for seed in anomalous_seeds:
        result = next(
            item for item in results
            if item["seed"] == seed
        )

        print(
            f"\nSeed: {seed}"
            f"\nBest arm: {result['best_arm']}"
            f"\nWorst competitor: {result['worst_competitor']}"
            f"\nExpected means: {result['expected_means']}"
            f"\nObserved means: {np.round(result['observed_means'], 3)}"
            f"\nMisrank gap: {result['misrank_gap']:.3f}"
        )

    save_anomalies(results)
