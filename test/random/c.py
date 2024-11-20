import numpy as np


def apply_temperature(logits, temperature=1.0):
    """
    Apply temperature scaling to logits.

    Args:
        logits (np.array): Raw logits from model
        temperature (float): Temperature parameter (default=1.0)

    Returns:
        np.array: Temperature-scaled logits
    """
    if temperature == 0:  # Handle division by zero - equivalent to argmax
        return np.where(logits == np.max(logits), np.inf, -np.inf)
    return logits / temperature


def softmax(logits):
    """
    Convert logits to probabilities using softmax function.

    Args:
        logits (np.array): Input logits

    Returns:
        np.array: Probability distribution
    """
    exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exp_logits / np.sum(exp_logits)


def top_k_sampling(logits, k, temperature=1.0, return_probs=False):
    """
    Perform top-k sampling on logits.

    Args:
        logits (np.array): Raw logits from model
        k (int): Number of top tokens to consider
        temperature (float): Temperature parameter for sampling
        return_probs (bool): Whether to return probabilities for top-k tokens

    Returns:
        int: Sampled token index
        (optional) np.array: Probabilities of top-k tokens if return_probs=True
    """
    # Apply temperature scaling
    scaled_logits = apply_temperature(logits, temperature)

    # Get indices of top-k logits
    top_k_indices = np.argsort(scaled_logits)[-k:]

    # Get the corresponding logits
    top_k_logits = scaled_logits[top_k_indices]

    # Convert to probabilities (softmax)
    top_k_probs = softmax(top_k_logits)

    # Sample from the top-k distribution
    sampled_idx = np.random.choice(k, p=top_k_probs)

    # Convert back to original token index
    selected_token = top_k_indices[sampled_idx]

    if return_probs:
        return selected_token, top_k_probs
    return selected_token


# Example usage and tests
if __name__ == "__main__":
    # Example 1: Basic sampling
    print("\nExample 1: Basic sampling")
    # Simulate logits for a vocabulary of size 10
    example_logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 0.5])

    print("Original logits:", example_logits)
    sampled_token = top_k_sampling(example_logits, k=5, temperature=1.0)
    print(f"Sampled token index (k=5, temp=1.0): {sampled_token}")

    # Example 2: Effect of temperature
    print("\nExample 2: Effect of temperature")
    # Sample with different temperatures
    for temp in [0.5, 1.0, 2.0]:
        token, probs = top_k_sampling(
            example_logits, k=5, temperature=temp, return_probs=True
        )
        print(f"\nTemperature = {temp}")
        print(f"Top-5 probabilities: {probs.round(3)}")
        print(f"Sampled token: {token}")

    # Example 3: Extreme case - very low temperature
    print("\nExample 3: Very low temperature (close to argmax)")
    token = top_k_sampling(example_logits, k=5, temperature=0.1)
    print(f"Sampled token with temp=0.1: {token}")

    # Example 4: Different k values
    print("\nExample 4: Different k values")
    for k in [2, 5, 8]:
        token, probs = top_k_sampling(
            example_logits, k=k, temperature=1.0, return_probs=True
        )
        print(f"\nk = {k}")
        print(f"Probabilities: {probs.round(3)}")
        print(f"Sampled token: {token}")
