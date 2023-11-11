from plotting.plot import plot_linear_regression
from utils import run

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
data_params = {
    "period": 1,
    "number_periods": 1,
    "min_value": 0,
    "max_value": 1,
    "size": 1000,
}
category = "multi_sinusoidal"  # repeating, sinusoidal, multi_sinusoidal
increasing_param = "number_periods"

stop_delta = 0.01
repeats = 1
max_inc = 2
max_hidden_dim = 63
info = True


results_df = run(
    trainer_params,
    category,
    data_params,
    increasing_param,
    repeats,
    stop_delta,
    max_inc,
    max_hidden_dim,
    info,
)

plot_linear_regression(
    results_df,
    f"linear_regression_{category}_lb{trainer_params['lookback']}_maxinc{max_inc}_h{max_hidden_dim}",
)
