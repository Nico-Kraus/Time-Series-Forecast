import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data.repeating import repeating
from data.sinusoidal import sinusoidal
from data.multi_sinusoidal import multi_sinusoidal
from data.noise import noise
from data.piecewise_linear import piecewise_linear
from data.uniform_piecewise_linear import uniform_piecewise_linear

_c = {
    "repeating": repeating,
    "sinusoidal": sinusoidal,
    "multi_sinusoidal": multi_sinusoidal,
    "noise": noise,
    "piecewise_linear": piecewise_linear,
    "uniform_piecewise_linear": uniform_piecewise_linear,
}


class Data:
    def __init__(self, size, config, lookback):
        self.lookback = lookback
        self.size = size
        if lookback != None:
            size = size + lookback + 1
        self.data = pd.DataFrame({"values": np.zeros(size)}, index=range(size))
        for category, params in config.items():
            self.data["values"] += _c[category](**params)
        # self.data["values"] = MinMaxScaler().fit_transform(self.data[["values"]])

    def get(self, split=(0.8, 0.2)):
        if split:
            train = self.data.iloc[: int(self.size * split[0] + self.lookback)]
            val = self.data.iloc[-int(self.size * split[1] + self.lookback) :]
            return train, val
        else:
            return self.data
