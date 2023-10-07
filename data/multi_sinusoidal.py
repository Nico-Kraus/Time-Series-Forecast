import numpy as np
import pandas as pd

from data.sinusoidal import sinusoidal


def multi_sinusoidal(difficulty, min_value=0, max_value=1, size=1000, **params):
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
    multi_df = pd.DataFrame({"values": np.zeros(size)}, index=range(size))
    amplitude_range = max_value - min_value

    for _ in range(difficulty):
        random_period = np.random.uniform(min(10, size / 10), size / 2)
        random_amplitude = np.random.uniform(0.1 * amplitude_range, amplitude_range)

        puffer = (amplitude_range - random_amplitude) / 2
        random_y_offset = np.random.uniform(-puffer, puffer)
        random_min = (
            min_value + random_y_offset + (amplitude_range - random_amplitude) / 2
        )
        random_max = random_min + random_amplitude

        rdm_phase_offset = np.random.uniform(0, 2 * np.pi)
        single_df = sinusoidal(
            random_period, random_min, random_max, rdm_phase_offset, size
        )
        multi_df["values"] += single_df["values"]

    # Normalize the final sum to ensure it's within the specified min_value and max_value
    multi_df["values"] = (max_value - min_value) * (
        (multi_df["values"] - multi_df["values"].min())
        / (multi_df["values"].max() - multi_df["values"].min())
    ) + min_value

    return multi_df
