import numpy as np


def explain_random_choice():
    # Basic example
    options = ["A", "B", "C"]
    probs = [0.5, 0.3, 0.2]  # Must sum to 1.0

    print("1. Basic probability sampling:")
    print(f"Options: {options}")
    print(f"Probabilities: {probs}")

    # Internal working simulation
    np.random.seed(42)  # For reproducibility

    # How random.choice works internally:
    def simulate_random_choice(options, probs, n_samples=1):
        cumsum = np.cumsum(probs)  # Cumulative sum: [0.5, 0.8, 1.0]
        results = []

        for _ in range(n_samples):
            # Generate random number between 0 and 1
            r = np.random.random()

            # Find where this random number falls
            for idx, cum_prob in enumerate(cumsum):
                if r <= cum_prob:
                    results.append(options[idx])
                    break

        return results[0] if n_samples == 1 else results

    # Show the cumulative probabilities
    print("\n2. Cumulative probabilities (internal):")
    cumsum = np.cumsum(probs)
    for i, (opt, cum) in enumerate(zip(options, cumsum)):
        prev = cumsum[i - 1] if i > 0 else 0
        print(f"Option {opt}: region {prev:.2f} to {cum:.2f}")

    # Sample multiple times to show distribution
    n_trials = 100
    # debug print for option, size, p
    print(f"option: {options}, size: {n_trials}, p: {probs}")
    results = np.random.choice(options, size=n_trials, p=probs)
    # debug print for results
    print(f"results: {results}")

    print(f"\n3. Results from {n_trials} samples:")
    unique, counts = np.unique(results, return_counts=True)
    for opt, count in zip(unique, counts):
        expected = probs[options.index(opt)] * n_trials
        print(f"Option {opt}:")
        print(f"  Expected: {expected:.0f} times ({probs[options.index(opt)]:.1%})")
        print(f"  Actual:   {count} times ({count/n_trials:.1%})")

    # Visual representation of probability regions
    print("\n4. Visual representation of probability regions:")
    print("0.0    0.5    0.8    1.0")
    print("│      │      │      │")
    print("V      V      V      V")
    print("├──────┼──────┼──────┤")
    print("│   A  │  B   │  C   │")
    print("└──────┴──────┴──────┘")

    # Show some random samples with their random numbers
    print("\n5. Sample random number examples:")
    for _ in range(5):
        r = np.random.random()
        choice = simulate_random_choice(options, probs)
        print(f"Random number: {r:.3f} -> Chose: {choice}")


explain_random_choice()
