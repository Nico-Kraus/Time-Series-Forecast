from sklearn.preprocessing import MinMaxScaler

from data.repeating import repeating
from data.sinusoidal import sinusoidal
from data.multi_sinusoidal import multi_sinusoidal
from data.random import random
from data.sinusoidal_with_noise import sinusoidal_with_noise
from data.piecewise_linear import piecewise_linear, uniform_piecewise_linear
from data.autoregressive import autoregressive

_c = {
    "repeating": repeating,
    "sinusoidal": sinusoidal,
    "multi_sinusoidal": multi_sinusoidal,
    "random": random,
    "sinusoidal_with_noise": sinusoidal_with_noise,
    "piecewise_linear": piecewise_linear,
    "uniform_piecewise_linear": uniform_piecewise_linear,
    "autoregressive": autoregressive,
}


class Data:
    def __init__(self, category, size, difficulty, lookback):
        self.lookback = lookback
        self.size = size
        if lookback != None:
            size = size + lookback + 1
        self.data = _c[category](difficulty=difficulty, size=size)
        self.data["values"] = MinMaxScaler().fit_transform(self.data[["values"]])

    def get(self, split=(0.8, 0.2)):
        if split:
            train = self.data.iloc[: int(self.size * split[0] + self.lookback)]
            val = self.data.iloc[-int(self.size * split[1] + self.lookback) :]
            return train, val
        else:
            return self.data
