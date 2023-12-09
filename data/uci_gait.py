import pandas as pd
from pathlib import Path
import warnings

def uci_gait(size, start, **params):
    PATH = Path(__file__).parent.resolve()
    path = Path(PATH, "uci_data/gait.csv")
    df = pd.read_csv(path)
    ts = df['angle'].to_numpy()
    if size > len(ts):
        warnings.warn("Size is larger than the length of the time series. Returning full data.")
        return ts
    elif size + start > len(ts):
        warnings.warn("Size + start is larger than the length of the time series. Returning last size elements.")
        return ts[-size:]
    return ts[start: start + size]