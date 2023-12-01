import pandas as pd
from utils import (
    get_params,
    get_model_test_loss,
    get_prediction_loss,
    get_sample_entropy,
    create_name,
    dict_to_string,
)
from data.data import Data
from plotting.plot import double_plot_linear_regressions

data_lookback = 10
loss_func = "L1"

data_params = {
    "size": 1000,
    "seed": 0,
    "config": {"piecewise_linear": {"num_seg": 1}},
}

max_difficulty = 100
increasing_type = "piecewise_linear"
increasing_param = "num_seg"
factor = 1
repeats = 10

models = ["lstm", "custom_lstm"]
predictors = ["last_value_loss"]

model_params = {}
for model in models:
    model_params[model] = get_params(model)
    model_params[model]["loss"] = loss_func
    model_params[model]["lookback"] = data_lookback    

loss_categories = models + predictors
metric_categories = ["entropy"]


data_params["lookback"] = data_lookback
columns = ["type", "seed", "difficulty"] + loss_categories + metric_categories
data_type = dict_to_string(data_params)
results = pd.DataFrame(columns=columns)
for difficulty in range(1, max_difficulty + 1):
    print(f"{difficulty} ", end="", flush=True)
    data_params["config"][increasing_type][increasing_param] = difficulty * factor
    losses, comp, entropies = [], [], []
    for i in range(repeats):
        data_params["seed"] += 1
        train_df, val_df, test_df = Data(**data_params).get(split=(0.8, 0.1, 0.1))
        entropy = get_sample_entropy(train_df, val_df, test_df, m=10, tau=0)
        model_results = {}
        for name, params in model_params.items():
            model_results[name] = get_model_test_loss(train_df, val_df, test_df, params)
        last_value_loss = get_prediction_loss(test_df, method="last_value", loss=loss_func, lookback=data_params["lookback"])

        row = pd.DataFrame(
            [
                {
                    "type": data_type,
                    "seed": data_params["seed"],
                    "difficulty": difficulty,
                    "entropy": entropy,
                    "lstm": model_results["lstm"],
                    "custom_lstm": model_results["custom_lstm"],
                    "last_value_loss": last_value_loss,
                }
            ],
            columns=columns,
        )
        results = pd.concat([results, row], ignore_index=True)

print()
name = create_name("entrophy", data_params, max_difficulty)
double_plot_linear_regressions(results, name, loss_categories, metric_categories)
