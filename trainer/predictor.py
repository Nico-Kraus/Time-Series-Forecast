import torch
from torch.utils.data import DataLoader
from trainer.utils import SlidingWindowDataset, _loss


class Predictor:
    def __init__(self, method, loss, lookback):
        _m = {"last_value": self.last_value}
        self.__method = _m[method]
        self.__loss = _loss[loss]()
        self.__lookback = lookback

    def test(self, x_test, info=True):
        x_test = torch.tensor(x_test.values, dtype=torch.float)
        test_dataset = SlidingWindowDataset(x_test, self.__lookback + 1)
        test_loader = DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False
        )

        for batch_idx, (x_true, y_true) in enumerate(test_loader):
            pred = self.__method(x_true)[:,-1]
            loss = self.__loss(pred.view(y_true.shape[0], 1), y_true.view(y_true.shape[0], 1))

        avg_loss = loss.mean().item()
        if info:
            print(f"test loss = {avg_loss}")

        return avg_loss, pred

    def last_value(self, x):
        return x[:, :, 0]
