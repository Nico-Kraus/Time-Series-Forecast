from sklearn.preprocessing import MinMaxScaler

from data.repeating import repeating
from data.sinusoidal import sinusoidal
from data.multi_sinusoidal import multi_sinusoidal
from data.generate_random import generate_random
from data.sinus_linear_noise import sinus_linear_noise
from data.piecewise_linear import piecewise_linear

_c = {
    "repeating": repeating,
    "sinusoidal": sinusoidal,
    "multi_sinusoidal": multi_sinusoidal,
    "generate_random": generate_random,
    "sinus_linear_noise": sinus_linear_noise,
    "piecewise_linear": piecewise_linear,
}


class Data:
    def __init__(self, category, params, lookback):
        self.lookback = lookback
        self.size = params["size"]
        if lookback != None:
            params["size"] += lookback + 1
        self.data = _c[category](**params)
        scaler = MinMaxScaler()
        self.data["values"] = scaler.fit_transform(self.data[["values"]])

    def get(self, split=(0.8, 0.2)):
        train = self.data.iloc[: int(self.size * split[0] + self.lookback)]
        val = self.data.iloc[-int(self.size * split[1] + self.lookback) :]
        # train_size = int(self.lookback + self.size * split[0])
        # val_size = int(self.lookback + self.size * split[1])
        # print(train_size, train.shape[0], val_size, val.shape[0])
        return train, val
