import numpy as np
import pandas as pd


def sinusoidal(
    difficulty, min_value=0, max_value=1, phase_offset=0, size=1000, **params
):
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
    period = difficulty
    x = np.linspace(0, 2 * np.pi * (size / period), size) + phase_offset
    y = 0.5 * (max_value - min_value) * np.sin(x) + 0.5 * (max_value + min_value)

    return pd.DataFrame({"values": y}, index=range(size))
