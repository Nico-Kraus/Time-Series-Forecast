import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def piecewise_linear(size=1000, num_seg=5, min_value=0, max_value=1):
    """
    Create a DataFrame with a piecewise linear function.
    The size of the segments is random

    Parameters:
    - size: Size of the dataframe
    - num_seg: number of segments
    - min_value: Minimum value
    - max_value: Maximum value

    Returns:
    - numpy time series

    """
    # Generate random breakpoints within the series, ensuring start and end points are included
    breakpoints = [0, size]
    breakpoints.extend(np.random.choice(np.arange(1, size), num_seg - 1, replace=False))
    breakpoints.sort()

    ts = np.zeros(size)

    for i in range(num_seg):
        start_idx = breakpoints[i]
        end_idx = breakpoints[i + 1]
        slope = np.random.uniform(-1, 1)
        segment = slope * np.arange(end_idx - start_idx) + (
            0 if i == 0 else ts[start_idx - 1]
        )
        ts[start_idx:end_idx] = segment
    return (
        MinMaxScaler().fit_transform(ts.reshape(-1, 1)) * (max_value - min_value)
        + min_value
    )
