import pandas as pd
from utils import (
    get_params,
    get_model_test_loss,
    get_prediction_loss,
    get_characteristic,
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
    "config": {"fetch_stock_data" :{"symbol": "MSFT", "data_type": "Close", "interval": "1d", "start_date": "2000-01-01"}}
}

symbols = ["AAPL", "MSFT", "AMZN", "GOOG", "ALV.DE", "NVDA", "UNH", "TSM", "BMW.DE", "LLY"]
dates = ["2005-06-16", "2006-07-19", "2007-08-22", "2008-09-25", "2009-10-28",
        "2010-12-02", "2012-01-05", "2013-02-08", "2014-03-11", "2015-04-14"]
repeats = 10

model_names = ["lstm_xs", "lstm", "dnn_xs", "dnn", "cnn_xs", "cnn"]
predictor_names = ["last_value", "regression", "arima", "knn"]
characteristics = {"sample_entropy": {"m": 10, "tau": 0}, "mad": {}, "num_edges": {}, "compression": {"sax_word_size": 20, "sax_alphabet_size": 10, "div": 64}, 
                   "perm_entropy": {"max_n": 7}, "noise": {"model": "additive"}, "mean_trend": {"model": "additive"}, "trend_complexity": {"model": "additive"},
                   "period": {}, "period_complexity": {"model": "additive"}, "spectral_entropy_fft": {}, "spectral_entropy_psd": {}}

model_params = {}
for model in model_names:
    model_params[model] = get_params(model)
    model_params[model]["loss"] = loss_func
    model_params[model]["lookback"] = data_lookback


data_params["lookback"] = data_lookback
columns = ["name", "seed", "repeats", "stock", "date"] + model_names + predictor_names + list(characteristics.keys())
results = pd.DataFrame(columns=columns)
for symbol in symbols:
    for date in dates:
        print(f"{symbol}: {date} ", end="", flush=True)
        data_params["config"]["fetch_stock_data"]["symbol"] = symbol
        data_params["config"]["fetch_stock_data"]["start_date"] = date
        for repeat in range(repeats):
            data_params["seed"] += 1
            train_df, val_df, test_df = Data(**data_params).get(split=(0.8, 0.1, 0.1))
            characteristics_results = {}
            for c_name, c_params in characteristics.items():
                characteristics_results[c_name] = get_characteristic(train_df, val_df, test_df, data_lookback, c_name, c_params)
            model_results = {}
            for model_name, params in model_params.items():
                model_results[model_name] = get_model_test_loss(train_df, val_df, test_df, params)
            predictor_results = {}
            for predictor_name in predictor_names:
                predictor_results[predictor_name] = get_prediction_loss(train_df, val_df, test_df, method=predictor_name, loss=loss_func, lookback=data_lookback)

            row_data = {
                "name": dict_to_string(data_params),
                "seed": data_params["seed"],
                "repeats": repeat,
                "stock": symbol,
                "date": date
            }
            for c_name in characteristics.keys():
                value = characteristics_results.get(c_name, None)
                if value is not None:
                    row_data[c_name] = float(value)
                else:
                    row_data[c_name] = None
            for predictor_name in predictor_names:
                value = predictor_results.get(predictor_name, None)
                if value is not None:
                    row_data[predictor_name] = float(value)
                else:
                    row_data[predictor_name] = None
            for model_name in model_names:
                value = model_results.get(model_name, None)
                if value is not None:
                    row_data[model_name] = float(value)
                else:
                    row_data[model_name] = None

            row = pd.DataFrame([row_data], columns=columns)
            results = pd.concat([results, row], ignore_index=True)

            name = create_name("category_stock", data_params, 1)
            df_to_csv(results, f"{name}.csv")

print()