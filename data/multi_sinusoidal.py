import numpy as np
from sklearn.preprocessing import MinMaxScaler

from data.sinusoidal import sinusoidal


def multi_sinusoidal(rng, size=1000, num_sin=5, min_value=0, max_value=1):
    """
    Create a DataFrame with sinusoidal values.

    Parameters:
    - difficulty: number of periods of sinusoidal functions
    - min_value: Minimum amplitude value
    - max_value: Maximum amplitude value
    - size: Size of the dataframe

    Returns:
    - DataFrame
    """
    ts = np.zeros(size)
    amplitude_range = max_value - min_value

    for _ in range(num_sin):
        random_period = rng.uniform(min(10, size / 10), size / 2)
        random_amplitude = rng.uniform(0.1 * amplitude_range, amplitude_range)

        puffer = (amplitude_range - random_amplitude) / 2
        random_y_offset = rng.uniform(-puffer, puffer)
        random_min = (
            min_value + random_y_offset + (amplitude_range - random_amplitude) / 2
        )
        random_max = random_min + random_amplitude

        rdm_phase_offset = rng.uniform(0, 2 * np.pi)
        single_df = sinusoidal(
            rng=rng,
            size=size,
            period=random_period,
            phase_offset=rdm_phase_offset,
            min_value=random_min,
            max_value=random_max,
        )
        ts += single_df

    # Normalize the final sum to ensure it's within the specified min_value and max_value
    ts = MinMaxScaler().fit_transform(ts.reshape(-1, 1))
    return (ts * (max_value - min_value) + min_value).flatten()
