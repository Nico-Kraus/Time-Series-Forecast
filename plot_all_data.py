from plotting.plot import plot_model_comparisons, plot_correlation_matrix, plot_all_ts, plot_box, plot_ridge, double_plot_linear_regressions
from utils import get_data
import pandas as pd

# csv_name = "all_data"
csv_name = "all_data_2"
# csv_name = "entrophy_piecewise_linear_100"
# csv_name = "entrophy_multi_sinusoidal_100"
data, data_lookback, loss_func = get_data("data")
plot_all_ts(data)
print(len(data))

results = pd.read_csv(f"results/{csv_name}.csv")

dnn_categories = ["dnn_xs", "dnn_m"]
cnn_categories = ["cnn_xs", "cnn_m"]
lstm_categories = ["lstm_xs", "open_lstm_m"]
method_categories = [ "last_value", "regression", "arima", "knn"]
model_categories = lstm_categories + dnn_categories + cnn_categories + method_categories
characteristics = ["mad", "num_edges"]
characteristics_l = ["sample_entropy", "perm_entropy", "compression"]
list_model_categories = [dnn_categories, cnn_categories, lstm_categories, method_categories, characteristics, characteristics_l]
plot_categories = model_categories + characteristics + characteristics_l
print(results[["name"] + plot_categories])

plot_model_comparisons(results, f"{csv_name}_scatter", plot_categories)
plot_correlation_matrix(results, f"{csv_name}_corr_mat", plot_categories)
plot_box(results, f"{csv_name}_box", model_categories)
plot_ridge(results, f"{csv_name}_ridge", model_categories)
double_plot_linear_regressions(results, f"{csv_name}_regressions", list_model_categories, path="all_results")
