import numpy as np
import pandas as pd


def noise(size=1000, mean=0, std_dev=1):
    """
    Generate a random time series with the given mean and standard deviation.

    Parameters:
    - size (int): The size of the time series. Default is 1000.
    - mean (float): The desired mean of the time series.
    - std_dev (float): The desired standard deviation of the time series.

    Returns:
    - numpy time series
    """
    random_series = np.random.randn(size)
    return random_series * std_dev + mean
