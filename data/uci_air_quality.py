import pandas as pd
from pathlib import Path
import warnings
# from ucimlrepo import fetch_ucirepo 


def uci_air_quality(size = 1000, start = 0, name = "CO(GT)", **params):
    # air_quality = fetch_ucirepo(id=360) 
    # df = air_quality.data.features
    # df.to_csv(path)

    PATH = Path(__file__).parent.resolve()
    path = Path(PATH, "uci_data/air_quality.csv")

    df = pd.read_csv(path)
    df =  df[["Date", "Time", name]]
    # drop -200 they seem to be None values
    df = df[~(df == -200).any(axis=1)]
    
    ts = df[name].to_numpy()
    if size > len(ts):
        warnings.warn("Size is larger than the length of the time series. Returning full data.")
        return ts
    elif size + start > len(ts):
        warnings.warn("Size + start is larger than the length of the time series. Returning last size elements.")
        return ts[-size:]
    return ts[start: start + size]