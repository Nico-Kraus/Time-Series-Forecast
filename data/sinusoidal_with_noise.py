import numpy as np
import pandas as pd


def sinusoidal_with_noise(difficulty, size=1000):
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
    x = np.linspace(0, 2 * 10 * np.pi, size)
    y = 0.5 * np.sin(x) + 0.5

    noise = np.random.randn(size) * (difficulty / 100)
    y += noise

    return pd.DataFrame({"values": y}, index=range(size))
