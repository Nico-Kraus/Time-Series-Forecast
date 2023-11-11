import numpy as np
import pandas as pd


def autoregressive(difficulty, size=1000):
    # Define coefficients for autoregressive model
    coefficients = np.random.uniform(-1, 1, difficulty)

    time_series_data = np.random.normal(loc=0, scale=1, size=difficulty)
    time_series_data = np.concatenate([time_series_data, np.zeros(size - difficulty)])

    for i in range(difficulty, size):
        time_series_data[i] = np.dot(
            time_series_data[i - difficulty : i], coefficients
        ) + np.random.normal(loc=0, scale=1)

    return pd.DataFrame({"values": time_series_data}, index=range(size))
