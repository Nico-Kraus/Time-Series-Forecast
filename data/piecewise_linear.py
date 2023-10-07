import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def piecewise_linear(difficulty, size=1000, **params):
    # Initialize time series
    time_series_data = np.zeros(size)

    # Number of segments is difficulty + 1
    num_segments = difficulty + 1
    segment_size = size // num_segments

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (
            (i + 1) * segment_size if i != num_segments - 1 else size
        )  # Ensure the last segment goes up to the end
        slope = np.random.uniform(-1, 1)
        segment = slope * np.arange(end_idx - start_idx) + (
            0 if i == 0 else time_series_data[start_idx - 1]
        )
        time_series_data[start_idx:end_idx] = segment

    # Add noise
    noise = np.random.normal(loc=0, scale=1, size=size)
    time_series_data += noise

    time_series = pd.Series(data=time_series_data)
    time_series = time_series.to_frame(name="values")

    return time_series
