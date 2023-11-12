import numpy as np
from sklearn.preprocessing import MinMaxScaler


def uniform_piecewise_linear(rng, size=1000, num_seg=5, min_value=0, max_value=1):
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
    breakpoints = [i * (size // num_seg) for i in range(num_seg)] + [size]

    ts = np.zeros(size)

    for i in range(num_seg):
        start_idx = breakpoints[i]
        end_idx = breakpoints[i + 1]
        slope = rng.uniform(-1, 1)
        segment = slope * np.arange(end_idx - start_idx) + (
            0 if i == 0 else ts[start_idx - 1]
        )
        ts[start_idx:end_idx] = segment
    ts = MinMaxScaler().fit_transform(ts.reshape(-1, 1))
    return (ts * (max_value - min_value) + min_value).flatten()
