import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def autoregressive_time_series(difficulty, size):
    # Define coefficients for autoregressive model
    coefficients = np.random.uniform(-1, 1, difficulty)

    # Initialize time series
    time_series_data = np.random.normal(loc=0, scale=1, size=difficulty)
    time_series_data = np.concatenate([time_series_data, np.zeros(size - difficulty)])

    for i in range(difficulty, size):
        time_series_data[i] = np.dot(
            time_series_data[i - difficulty : i], coefficients
        ) + np.random.normal(loc=0, scale=1)

    return pd.Series(data=time_series_data)


ts = autoregressive_time_series(difficulty=1, size=365)
plt.figure(figsize=(10, 6))
ts.plot()
plt.title("Autoregressive Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
