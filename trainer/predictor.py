import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from trainer.utils import SlidingWindowDataset, _loss
import pmdarima as pm


class Predictor:
    def __init__(self, method, loss, lookback):
        _m = {
            "last_value": self.last_value,
            "regression": self.regression, 
            "arima": self.arima,   
        }
        self.__method = _m[method]
        self.__loss = _loss[loss]()
        self.__lookback = lookback
        if method == "regression":
            self.model = LinearRegression()

    def test(self, x_test, info=True):
        x_test = torch.tensor(x_test.values, dtype=torch.float)
        test_dataset = SlidingWindowDataset(x_test, self.__lookback + 1)
        test_loader = DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False
        )

        for batch_idx, (x_true, y_true) in enumerate(test_loader):
            pred = self.__method(x_true)
            loss = self.__loss(pred.view(y_true.shape[0], 1), y_true.view(y_true.shape[0], 1))

        avg_loss = loss.mean().item()
        if info:
            print(f"test loss = {avg_loss}")

        return avg_loss, pred

    def last_value(self, x):
        return x[:, -1, 0]
    
    def regression(self, y_):
        x_ = torch.arange(self.__lookback).reshape(-1, 1)
        preds_ = []
        for t in range(y_.shape[0]):
            y = y_[t, :, 0].reshape(-1, 1)
            r_sq = self.model.fit(x_, y)
            slope, intercept = r_sq.coef_[0][0], r_sq.intercept_[0]
            pred = slope * self.__lookback + intercept
            pred = torch.tensor(pred, dtype=torch.float64)
            preds_.append(pred)
        return torch.stack(preds_)
    
    def arima(self, x):
        predictions = []
        batch_size = x.shape[0]
        for i in range(batch_size):
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module='pmdarima.arima.auto')
            warnings.filterwarnings("ignore", category=RuntimeWarning, module='statsmodels.tsa.statespace.sarimax')
            series = x[i, :, 0].numpy()  # Convert to numpy array and remove the last dimension
            model = pm.auto_arima(series, suppress_warnings=True, error_action='ignore')
            prediction = model.predict(n_periods=1)
            predictions.append(prediction[0])
        return torch.tensor(predictions)

