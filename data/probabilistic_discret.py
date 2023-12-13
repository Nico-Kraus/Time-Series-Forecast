import numpy as np
import random
import warnings

def probabilistic_discret(rng, size = 1000, n = 10, m = 5, min_p = 0.1, max_p = 0.9, first_p = False, **params):
    # Check if m is less than n
    if m >= n:
        m = n - 1
        warnings.warn("m should be less than n, m is n-1 now", EncodingWarning)

    # Generate probabilities for each integer n_
    transitions = {}
    for n_ in range(n):
        followers = rng.choice(range(n), m, replace=False)
        if first_p:
            probabilities = np.append(rng.uniform(min_p, 1 - first_p, m-1), first_p)
        else:
            probabilities = rng.uniform(min_p, max_p, m)
        probabilities /= probabilities.sum()  # Normalize to sum up to 1
        transitions[n_] = dict(zip(followers, probabilities))

    # Generate the time series
    time_series = [0]
    for _ in range(size - 1):
        current = time_series[-1]
        choices, probs = zip(*transitions[current].items())
        next_int = rng.choice(choices, p=probs)
        time_series.append(next_int)

    return np.array(time_series)