import numpy as np
import pandas as pd


def sinusoidal(difficulty, size=1000):
    """
    Create a DataFrame with sinusoidal values.

    Parameters:
    - period: Period in terms of index size
    - min_value: Minimum amplitude value
    - max_value: Maximum amplitude value
    - size: Size of the dataframe

    Returns:
    - DataFrame
    """
    max_value, min_value, phase_offset, period = 1, 0, 0, difficulty
    x = np.linspace(0, 2 * np.pi * (size / period), size) + phase_offset
    y = 0.5 * (max_value - min_value) * np.sin(x) + 0.5 * (max_value + min_value)

    return pd.DataFrame({"values": y}, index=range(size))
