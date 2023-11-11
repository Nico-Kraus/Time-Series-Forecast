import numpy as np
import pandas as pd


def repeating(size=1000, period=10, min_value=0, max_value=1):
    """
    Create a DataFrame with repeating values.

    Parameters:
    - size: Size of the dataframe
    - period: sequence_length is the lenght of the repeating series
    - min_value: Minimum value
    - max_value: Maximum value

    Returns:
    - numpy time series
    """
    sequence = np.random.uniform(min_value, max_value, period)
    return np.tile(sequence, size // len(sequence) + 1)[:size]
