import numpy as np
import pandas as pd


def sinus_linear_noise(difficulty, size=1000, **params):
    """
    Generate a time series data of given size.

    :param difficulty: Controls the level of noise and complexity of seasonality.
    :param size: Number of data points.
    :return: A Pandas Series representing the time series.
    """

    # Generate trend component
    trend = np.linspace(start=0, stop=size // 10, num=size)

    # Generate seasonal component
    frequency = difficulty  # Higher difficulty, higher frequency of oscillations
    seasonal = 10 * np.sin(2 * np.pi * frequency * np.arange(size) / size)

    # Generate noise component
    noise_level = difficulty  # Higher difficulty, higher noise
    noise = noise_level * np.random.normal(loc=0, scale=1, size=size)

    # Combine components to create time series
    time_series_data = trend + seasonal + noise

    # Create a Pandas Series
    time_series = pd.Series(data=time_series_data)
    time_series = time_series.to_frame(name="values")

    return time_series
