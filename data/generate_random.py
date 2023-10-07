import numpy as np
import pandas as pd


def generate_random(mean, std_dev, size=1000, **params):
    """
    Generate a random time series with the given mean and standard deviation.

    Parameters:
    mean (float): The desired mean of the time series.
    std_dev (float): The desired standard deviation of the time series.
    size (int): The size of the time series. Default is 1000.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the random time series.
    """
    random_series = np.random.randn(size)
    scaled_series = random_series * std_dev + mean
    return pd.DataFrame({"Value": scaled_series})
