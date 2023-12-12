import torch
from torch.utils.data import Dataset

from models.lstm import LSTM
from models.peephole_lstm import PeepholeLSTM
from models.open_lstm import OpenLSTM
from models.input_lstm import InputLSTM
from models.dnn import DNN
from models.cnn import CNN_XS, CNN_M
from models.transformer import Transformer

_model = {"lstm": LSTM, "peepholelstm": PeepholeLSTM, "openlstm": OpenLSTM, "inputlstm": InputLSTM, "dnn": DNN, "cnn_xs": CNN_XS, "cnn_m": CNN_M, "transformer": Transformer}
_loss = {"MSE": torch.nn.MSELoss, "L1": torch.nn.L1Loss}
_optimizer = {
    "Adam": torch.optim.Adam,
    "NAdam": torch.optim.NAdam,
    "AdamW": torch.optim.AdamW,
}


class SlidingWindowDataset(Dataset):
    def __init__(self, time_series, lookback):
        self.time_series = time_series
        self.lookback = lookback

    def __len__(self):
        return len(self.time_series) - self.lookback + 1

    def __getitem__(self, idx):
        x = self.time_series[idx : idx + self.lookback - 1].clone().detach()
        y = (
            self.time_series[idx + self.lookback - 1 : idx + self.lookback]
            .clone()
            .detach()
        )
        return x.float(), y.float()
