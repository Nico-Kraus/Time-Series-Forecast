import numpy as np
import pandas as pd


def sinusoidal(size=1000, period=100, phase_offset=0, min_value=0, max_value=1):
    """
    Create a DataFrame with sinusoidal values.

    Parameters:
    - size: Size of the dataframe
    - period: Period in terms of index size
    - phase_offset: offset to change wheree the period starts at 0
    - min_value: Minimum amplitude value
    - max_value: Maximum amplitude value

    Returns:
    - DataFrame
    """
    x = np.linspace(0, 2 * np.pi * (size / period), size) + phase_offset
    return 0.5 * (max_value - min_value) * np.sin(x) + 0.5 * (max_value + min_value)
