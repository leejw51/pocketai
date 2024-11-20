import numpy as np

# Simple random sampling from array
arr = np.array([1, 2, 3, 4, 5])
random_element = np.random.choice(arr)
print("Single random element:", random_element)

# Multiple samples
three_samples = np.random.choice(arr, size=3)
print("Three random samples:", three_samples)

# Sampling without replacement (no duplicates)
unique_samples = np.random.choice(arr, size=3, replace=False)
print("Three unique samples:", unique_samples)

# Sampling with custom probabilities
probabilities = [0.1, 0.1, 0.2, 0.3, 0.3]
weighted_sample = np.random.choice(arr, size=2, p=probabilities)
print("Two weighted samples:", weighted_sample)

# Sampling from range using integer
dice_roll = np.random.choice(6, size=1) + 1
print("Dice roll:", dice_roll[0])
