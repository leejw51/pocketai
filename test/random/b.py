import numpy as np


def sample_tokens(logits, method="argmax", temperature=1.0, k=None):
    """Compare different sampling methods"""
    # Ensure logits are numpy array
    logits = np.array(logits)

    if method == "argmax":
        # Simply take highest probability token
        return np.argmax(logits)

    # Apply temperature scaling for other methods
    logits = logits / temperature

    if method == "sample":
        # Convert to probabilities
        probs = np.exp(logits) / np.sum(np.exp(logits))
        # Sample from full distribution
        return np.random.choice(len(logits), p=probs)

    elif method == "top_k":
        if k is None:
            k = len(logits) // 2  # Default to half of vocabulary
        # Get top k indices
        top_k_indices = np.argsort(logits)[-k:]
        # Get probabilities for top k
        top_k_logits = logits[top_k_indices]
        probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
        # Sample from restricted distribution
        selected_idx = np.random.choice(k, p=probs)
        return top_k_indices[selected_idx]


# Example with repeated sampling
np.random.seed(42)  # For reproducibility

# Simulate logits for a small vocabulary
example_logits = [2.0, 1.8, 1.5, 1.0, 0.5]
n_samples = 10

print("Logits:", example_logits)
print("\nSampling 10 tokens with different methods:")

print("\nArgmax sampling:")
for _ in range(n_samples):
    token = sample_tokens(example_logits, method="argmax")
    print(f"Selected token: {token}", end=" ")

print("\n\nRegular sampling (temperature=1.0):")
for _ in range(n_samples):
    token = sample_tokens(example_logits, method="sample", temperature=1.0)
    print(f"Selected token: {token}", end=" ")

print("\n\nTop-k sampling (k=3, temperature=1.0):")
for _ in range(n_samples):
    token = sample_tokens(example_logits, method="top_k", k=3, temperature=1.0)
    print(f"Selected token: {token}", end=" ")

# Show effect of temperature
print("\n\nEffect of temperature on sampling:")
temperatures = [0.1, 0.5, 1.0, 2.0]
for temp in temperatures:
    print(f"\nTemperature = {temp}:")
    for _ in range(5):
        token = sample_tokens(example_logits, method="sample", temperature=temp)
        print(f"Selected token: {token}", end=" ")
