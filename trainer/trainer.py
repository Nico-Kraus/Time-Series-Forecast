import os
import torch
import uuid
from torch.utils.data import DataLoader
import torch.optim as optim

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
        patience: int = 10,
        init_method: str = "normal",
        device: str = "cpu",
    ):
        self.__model = _model[model](
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            init_method=init_method,
        )
        self.__loss = _loss[loss]()
        self.__optimizer = _optimizer[optimizer](self.__model.parameters(), lr=lr)
        self.__batch_size = batch_size
        self.__lookback = lookback
        self.__device = device
        self.__epochs = epochs
        self.__patience = patience
        self.__scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.__optimizer, "min")

    def train(self, x_train, x_val, info=True):
        best_val_loss = float("inf")
        best_model_path = f"models/tmp_best_model_{uuid.uuid4()}.pth"
        patience_counter = 0

        train_losses = []
        val_losses = []

        x_train = torch.tensor(x_train.values, dtype=torch.float)
        train_ds = SlidingWindowDataset(x_train, self.__lookback + 1)
        train_loader = DataLoader(train_ds, batch_size=self.__batch_size, shuffle=False)

        x_val = torch.tensor(x_val.values, dtype=torch.float)
        val_ds = SlidingWindowDataset(x_val, self.__lookback + 1)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

        for epoch in range(self.__epochs):
            self.__model.train()
            train_loss = 0
            for batch_idx, (x_true, y_true) in enumerate(train_loader):
                self.__model.zero_grad()

                pred = self.__model(x_true)
                loss = self.__loss(pred, y_true.view(y_true.shape[0], 1))
                loss.backward()
                self.__optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            self.__model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_idx, (x_true, y_true) in enumerate(val_loader):
                    pred = self.__model(x_true)
                    loss = self.__loss(pred, y_true.view(y_true.shape[0], 1))
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            self.__scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.__model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.__patience:
                    if info:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            if info:
                print(
                    f"epoch: {epoch+1}\t train loss: {avg_train_loss:.5f}\t val loss: {avg_val_loss:.5f}"
                )

        self.__model.load_state_dict(torch.load(best_model_path))
        os.remove(best_model_path)

        return train_losses, val_losses, pred

    def test(self, x_test, info=True):
        x_test = torch.tensor(x_test.values, dtype=torch.float)
        test_dataset = SlidingWindowDataset(x_test, self.__lookback + 1)
        test_loader = DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False
        )

        self.__model.eval()

        with torch.no_grad():
            for batch_idx, (x_true, y_true) in enumerate(test_loader):
                pred = self.__model(x_true)
                loss = self.__loss(pred, y_true.view(y_true.shape[0], 1))

        avg_loss = loss.mean().item()
        if info:
            print(f"test loss = {avg_loss:.5f}")

        self.__model.train()

        return avg_loss, pred
