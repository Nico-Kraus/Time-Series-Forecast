from plotting.plot import plot_model_comparisons, plot_correlation_matrix, plot_all_ts, plot_box, plot_ridge, double_plot_linear_regressions
from utils import get_data
import pandas as pd

csv_name = "all_data"

data, data_lookback, loss_func = get_data("data")
plot_all_ts(data)
print(len(data))

results = pd.read_csv(f"results/{csv_name}.csv")
print(results)

dnn_categories = ["dnn_xs", "dnn_m"]
lstm_categories = ["lstm_xs", "open_lstm_m"]
method_categories = [ "last_value", "regression", "arima"]
model_categories = lstm_categories + dnn_categories + method_categories
metric_categories = ["entropy"]
plot_categories = model_categories + metric_categories

plot_model_comparisons(results, f"{csv_name}_scatter", plot_categories)
plot_correlation_matrix(results, f"{csv_name}_corr_mat", plot_categories)
plot_box(results, f"{csv_name}_box", model_categories)
plot_ridge(results, f"{csv_name}_ridge", model_categories)
double_plot_linear_regressions(results, f"{csv_name}_regressions",  [dnn_categories, lstm_categories, method_categories, metric_categories], path="all_results")
