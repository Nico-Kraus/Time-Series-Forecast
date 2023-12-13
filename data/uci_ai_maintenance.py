import pandas as pd
from pathlib import Path
import warnings
  


def uci_ai_maintenance(size = 1000, start = 0, name = "CO(GT)", **params):

    PATH = Path(__file__).parent.resolve()
    path = Path(PATH, "uci_data/ai4i2020.csv")

    df = pd.read_csv(path)
    # df.replace('?', -200, inplace=True)
    # df.to_csv(path)
    df =  df[["UDI", name]]
    
    ts = df[name].astype(float).to_numpy()
    if size > len(ts):
        warnings.warn("Size is larger than the length of the time series. Returning full data.")
        return ts
    elif size + start > len(ts):
        warnings.warn("Size + start is larger than the length of the time series. Returning last size elements.")
        return ts[-size:]
    return ts[start: start + size]