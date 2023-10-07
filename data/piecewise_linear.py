import numpy as np
import pandas as pd


def piecewise_linear(difficulty, size=1000):
    # Generate random breakpoints within the series, ensuring start and end points are included
    breakpoints = [0, size]
    breakpoints.extend(
        np.random.choice(np.arange(1, size), difficulty - 1, replace=False)
    )
    breakpoints.sort()

    # Initialize time series
    time_series_data = np.zeros(size)

    for i in range(difficulty):
        start_idx = breakpoints[i]
        end_idx = breakpoints[i + 1]
        slope = np.random.uniform(-1, 1)
        segment = slope * np.arange(end_idx - start_idx) + (
            0 if i == 0 else time_series_data[start_idx - 1]
        )
        time_series_data[start_idx:end_idx] = segment

    return pd.DataFrame({"values": time_series_data}, index=range(size))


def uniform_piecewise_linear(difficulty, size=1000):
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

    return pd.DataFrame({"values": time_series_data}, index=range(size))
