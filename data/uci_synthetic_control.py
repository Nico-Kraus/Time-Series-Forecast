import warnings
import numpy as np
from pathlib import Path

def uci_synthetic_control(data_type, number, size, **params):
    PATH = Path(__file__).parent.resolve()
    file_path = Path(PATH, "uci_data/synthetic_control.data")

    data_types = {
        'normal': 1,
        'cyclic': 101,
        'increasing_trend': 201,
        'decreasing_trend': 301,
        'upward_shift': 401,
        'downward_shift': 501
    }
    if not 1 <= number <= 100:
        raise ValueError("Number must be between 1 and 100.")

    if data_type not in data_types:
        raise ValueError("Invalid data type. Must be one of: " + ", ".join(data_types.keys()))

    row_number = data_types[data_type] + number - 1
    with open(file_path, 'r') as file:
        for i, line in enumerate(file, start=1):
            if i == row_number:
                row_data = np.array([float(value) for value in line.split()])
                if size > len(row_data):
                    warnings.warn("Size is larger than the length of the time series. Returning full data.")
                    return row_data
                return row_data[:size]

    raise ValueError(f"Row {row_number} not found in the file.")

