import torch
from torch.utils.data import DataLoader

import numpy as np

from trainer.utils import SlidingWindowDataset, _model, _loss, _optimizer


class Trainer:
    def __init__(
        self,
        model: str = "lstm",
        input_dim: int = 1,
        hidden_dim: int = 32,
        n_layers: int = 2,
        output_dim: int = 1,
        epochs=10,
        lr: float = 0.01,
        batch_size: int = 10,
        loss: str = "L1",
        optimizer: str = "Adam",
        lookback: int = 10,
        device: str = "cpu",
    ):
        self.__model = _model[model](
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
        )
        self.__loss = _loss[loss]()
        self.__optimizer = _optimizer[optimizer](self.__model.parameters(), lr=lr)
        self.__batch_size = batch_size
        self.__lookback = lookback
        self.__device = device
        self.__epochs = epochs

    def train(self, x_train, info=True, stop_delta=0):
        losses = []
        last_epoch = None

        x_train = torch.tensor(x_train.values, dtype=torch.float)
        dataset = SlidingWindowDataset(x_train, self.__lookback + 1)
        dataloader = DataLoader(dataset, batch_size=self.__batch_size, shuffle=False)

        for epoch in range(self.__epochs):
            for batch_idx, (x_true, y_true) in enumerate(dataloader):
                self.__model.zero_grad()

                pred = self.__model(x_true)
                loss = self.__loss(pred, y_true.view(y_true.shape[0], 1))
                loss.backward()

                self.__optimizer.step()

            avg_loss = loss.mean().item()
            losses.append(avg_loss)
            if info:
                print(f"Epoch {epoch+1} / {self.__epochs}: Loss = {avg_loss:.3f}")
            if avg_loss < stop_delta:
                last_epoch = epoch
                if info:
                    print("finished")
                break

        return losses

    def val(self, x_val, info=True):
        x_val = torch.tensor(x_val.values, dtype=torch.float)
        dataset = SlidingWindowDataset(x_val, self.__lookback + 1)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        self.__model.eval()

        with torch.no_grad():
            for batch_idx, (x_true, y_true) in enumerate(dataloader):
                pred = self.__model(x_true)
                loss = self.__loss(pred, y_true.view(y_true.shape[0], 1))

        avg_loss = loss.mean().item()
        if info:
            print(f"val loss = {avg_loss}")

        self.__model.train()

        return avg_loss, pred
