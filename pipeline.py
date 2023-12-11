import pandas as pd

from utils import (
    get_data,
    get_params,
    get_model_test_loss,
    get_prediction_loss,
    get_sample_entropy,
    df_to_csv,
)
from data.data import Data

data, data_lookback, loss_func = get_data("data")
csv_name = "all_data_2"

model_names = [
    "lstm_xs", "open_lstm_m", "dnn_xs", "dnn_m", "cnn_xs", "cnn_m"
]
predictor_names = ["last_value", "regression", "arima", "knn"]
metrics = ["entropy"]

model_params = {}
for model in model_names:
    model_params[model] = get_params(model)
    model_params[model]["loss"] = loss_func
    model_params[model]["lookback"] = data_lookback

columns = ["name"] + model_names + predictor_names + metrics
results = pd.DataFrame(columns=columns)
for idx, (name, data_params) in enumerate(data.items()):
    print(f"{idx} ", end="", flush=True)
    train_df, val_df, test_df = Data(**data_params).get(split=(0.8, 0.1, 0.1))
    entropy = get_sample_entropy(train_df, val_df, test_df, m=10, tau=0)
    model_results = {}
    for model_name, params in model_params.items():
        model_results[model_name] = get_model_test_loss(train_df, val_df, test_df, params)
    predictor_results = {}
    for predictor_name in predictor_names:
        predictor_results[predictor_name] = get_prediction_loss(train_df, val_df, test_df, method=predictor_name, loss=loss_func, lookback=data_lookback)

    row_data = {
        "name": name,
        "entropy": entropy,
    }
    for predictor_name in predictor_names:
        row_data[predictor_name] = predictor_results.get(predictor_name, None)
    for model_name in model_names:
        row_data[model_name] = model_results.get(model_name, None)

    row = pd.DataFrame([row_data], columns=columns)
    results = pd.concat([results, row], ignore_index=True)

    df_to_csv(results, f"{csv_name}.csv")
print()