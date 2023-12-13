import pandas as pd
from pathlib import Path
import warnings
  


def uci_electricity(size = 1000, start = 0, name = "Global_active_power", **params):

    PATH = Path(__file__).parent.resolve()
    # path = Path(PATH, "uci_data/household_power_consumption.csv")
    path_s = Path(PATH, "uci_data/household_power_consumption_s.csv")

    df = pd.read_csv(path_s)
    # df.replace('?', -200, inplace=True)
    # df = df.iloc[:100000]
    # df.to_csv(path_s)
    df =  df[["Date", "Time", name]]
    # drop "?" they seem to be None values
    df = df[~(df == -200).any(axis=1)]
    
    ts = df[name].astype(float).to_numpy()
    if size > len(ts):
        warnings.warn("Size is larger than the length of the time series. Returning full data.")
        return ts
    elif size + start > len(ts):
        warnings.warn("Size + start is larger than the length of the time series. Returning last size elements.")
        return ts[-size:]
    return ts[start: start + size]