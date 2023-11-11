import pandas as pd
from tqdm import tqdm
import numpy as np

from plotting.plot import plot_linear_regression
from data.data import Data
from trainer.trainer import Trainer

trainer_params = {
    "model": "lstm",
    "input_dim": 1,
    "hidden_dim": 1,
    "n_layers": 1,
    "output_dim": 1,
    "epochs": 100,
    "lr": 0.01,
    "batch_size": 64,
    "loss": "L1",
    "optimizer": "Adam",
    "lookback": 10,
    "device": "cpu",
}
data_params = {"number_periods": 1, "min_value": 0, "max_value": 1, "size": 1000}
category = "multi_sinusoidal"

stop_delta = 0.01
repeats = 10
max_number = 50
max_hidden_dim = 64
info = False

results = {}
for number in range(1, max_number + 1):
    print(f"periods = {number}")
    hidden_dim_solved = []
    for repeat in range(repeats):
        for hidden_dim in range(1, max_hidden_dim + 1):
            data_params["number_periods"] = number
            trainer_params["hidden_dim"] = hidden_dim

            train_df, val_df = Data(
                category, data_params, lookback=trainer_params["lookback"]
            ).get(split=(0.8, 0.2))

            trainer = Trainer(**trainer_params)
            train_losses = trainer.train(x_train=train_df, info=info)
            val_loss = trainer.val(val_df, info=info)
            if val_loss <= stop_delta:
                hidden_dim_solved.append(hidden_dim)
                break
            elif hidden_dim == max_hidden_dim:
                print(f"Warning could not solve the problem with sl {number}")
    avg_hidden_dimensions = np.mean(hidden_dim_solved)
    results[number] = hidden_dim_solved

results_df = (
    pd.DataFrame(
        list(results.items()), columns=["Sequence length", "Hidden dimensions"]
    )
    .explode("Hidden dimensions")
    .apply(pd.to_numeric)
)

plot_linear_regression(
    results_df,
    f"linear_regression_sinusodial_multi_lb{trainer_params['lookback']}_maxn{max_number}_h{max_hidden_dim}",
)
