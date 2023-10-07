import torch
from torch.utils.data import Dataset

from models.lstm import LSTM

_model = {"lstm": LSTM}
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
        return len(self.time_series) - self.lookback

    def __getitem__(self, idx):
        x = self.time_series[idx : idx + self.lookback - 1].clone().detach()
        y = (
            self.time_series[idx + self.lookback - 1 : idx + self.lookback]
            .clone()
            .detach()
        )
        return x.float(), y.float()