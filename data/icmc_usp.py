import warnings
import numpy as np
from pathlib import Path

def icmc_usp(name, size, start= 0, **params):
    PATH = Path(__file__).parent.resolve()
    file_path = Path(PATH, f"icmc_usp/{name}.data")
    if Path.exists(file_path):
        ts = np.loadtxt(file_path)
        if size > len(ts):
            # warnings.warn("Size is larger than the length of the time series. Returning full data and copy beginning.")
            diff = size - len(ts)
            ts = np.concatenate((ts[:diff], ts))
            return ts
        elif size + start > len(ts):
            # warnings.warn("Size + start is larger than the length of the time series. Returning last size elements.")
            return ts[-size:]
        return ts[start:start+size]
    else:
        raise KeyError(f"Dataset {name} does not exist")


