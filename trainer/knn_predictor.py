import torch
from torch.utils.data import DataLoader
from trainer.utils import SlidingWindowDataset, _loss
from knn_tspi import KNeighborsTSPI
import numpy as np


class KNN_Predictor:
    def __init__(self, loss, lookback):
        self.__loss = _loss[loss]()
        self.__lookback = lookback
        self.__k = 8

        self.__knn = KNeighborsTSPI(k=self.__k, len_query=self.__lookback )


    def test(self, x_train, x_test, info=True):
        x_test = torch.tensor(x_test.values, dtype=torch.float)
        test_dataset = SlidingWindowDataset(x_test, self.__lookback+1)
        test_loader = DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False
        )

        train_numpy = x_train["values"].to_numpy()

        for batch_idx, (x_true, y_true) in enumerate(test_loader):
            to_pred = y_true.squeeze().numpy()
            pred = self.knn_forward(train_numpy, to_pred)
            loss = self.__loss(pred.view(y_true.shape[0], 1), y_true.view(y_true.shape[0], 1))

        avg_loss = loss.mean().item()
        if info:
            print(f"test loss = {avg_loss}")

        return avg_loss, pred

    def knn_forward(self, x_train, x_to_pred):
        predictions = []
        for to_pred in x_to_pred:
            self.__knn.fit(x_train)
            y = self.__knn.predict(h=1)
            predictions.append(y["mean"][0])
            x_train = np.append(x_train, to_pred)
        return torch.tensor(predictions)
            


