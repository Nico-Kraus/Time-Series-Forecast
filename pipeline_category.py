import pandas as pd
from utils import (
    get_params,
    get_model_test_loss,
    get_prediction_loss,
    get_sample_entropy,
    create_name,
    dict_to_string,
    df_to_csv,
)
from data.data import Data

data_lookback = 10
loss_func = "L1"

data_params = {
    "size": 1000,
    "seed": 0,
    "config": {"piecewise_linear": {"num_seg": 1}},
}

max_difficulty = 10
increasing_type = "piecewise_linear"
increasing_param = "num_seg"
factor = 1
repeats = 1

model_names = [
    "dnn_xs",
    "dnn_m",
    "lstm_xs",
    "open_lstm_m",
]
predictor_names = ["last_value", "regression", "arima"]
metrics = ["entropy"]

model_params = {}
for model in model_names:
    model_params[model] = get_params(model)
    model_params[model]["loss"] = loss_func
    model_params[model]["lookback"] = data_lookback


data_params["lookback"] = data_lookback
columns = ["type", "seed", "difficulty"] + model_names + predictor_names + metrics
data_type = dict_to_string(data_params)
results = pd.DataFrame(columns=columns)
for difficulty in range(1, max_difficulty + 1):
    print(f"{difficulty} ", end="", flush=True)
    data_params["config"][increasing_type][increasing_param] = difficulty * factor
    for i in range(repeats):
        data_params["seed"] += 1
        train_df, val_df, test_df = Data(**data_params).get(split=(0.8, 0.1, 0.1))
        entropy = get_sample_entropy(train_df, val_df, test_df, m=10, tau=0)
        model_results = {}
        for model_name, params in model_params.items():
            model_results[model_name] = get_model_test_loss(train_df, val_df, test_df, params)
        predictor_results = {}
        for predictor_name in predictor_names:
            predictor_results[predictor_name] = get_prediction_loss(test_df, method=predictor_name, loss=loss_func, lookback=data_lookback)

        row_data = {
            "type": data_type,
            "seed": data_params["seed"],
            "difficulty": difficulty,
            "entropy": entropy,
        }
        for predictor_name in predictor_names:
            row_data[predictor_name] = predictor_results.get(predictor_name, None)
        for model_name in model_names:
            row_data[model_name] = model_results.get(model_name, None)

        row = pd.DataFrame([row_data], columns=columns)
        results = pd.concat([results, row], ignore_index=True)

        name = create_name("entrophy", data_params, max_difficulty)
        df_to_csv(results, f"{name}.csv")

print()