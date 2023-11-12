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
    def __init__(self, size, config, lookback, seed):
        self.lookback = lookback
        self.size = size
        rng = np.random.default_rng(seed)
        if lookback != None:
            size = size + lookback + 1
        self.data = pd.DataFrame({"values": np.zeros(size)}, index=range(size))
        for category, params in sorted(config.items()):
            self.data["values"] += _c[category](size=size, rng=rng, **params)
        self.data["values"] = MinMaxScaler().fit_transform(self.data[["values"]])

    def get(self, split=(0.8, 0.1, 0.1)):
        if split:
            # train = self.data.iloc[: int(self.size * split[0] + self.lookback + 1)]
            # val = self.data.iloc[-int(self.size * split[1] + self.lookback + 1) :]
            # test = self.data.iloc[-int(self.size * split[2] + self.lookback + 1) :]

            # Training set
            train_end_idx = int(self.size * split[0]) + self.lookback + 1
            train = self.data.iloc[:train_end_idx]

            # Validation set
            val_start_idx = (
                train_end_idx - self.lookback - 1
            )  # Start with lookback data
            val_end_idx = val_start_idx + int(self.size * split[1]) + self.lookback
            val = self.data.iloc[val_start_idx:val_end_idx]

            # Test set
            test_start_idx = val_end_idx - self.lookback - 1  # Start with lookback data
            test = self.data.iloc[test_start_idx:]
            return train, val, test
        else:
            return self.data
