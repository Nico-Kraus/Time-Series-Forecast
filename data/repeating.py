import numpy as np
import pandas as pd


def repeating(difficulty, size=1000):
    """
    Create a DataFrame with repeating values.

    Parameters:
    - period: sequence_length is the lenght of the repeating series
    - min_value: Minimum value
    - max_value: Maximum value
    - size: Size of the dataframe

    Returns:
    - DataFrame
    """
    sequence = np.random.uniform(0, 1, difficulty)
    repeated_sequence = np.tile(sequence, size // len(sequence) + 1)[:size]
    return pd.DataFrame({"values": repeated_sequence}, index=range(size))
