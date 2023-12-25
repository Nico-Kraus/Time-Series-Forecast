from plotting.plot import plot_model_comparisons, plot_correlation_matrix, plot_all_ts, plot_box, plot_ridge, double_plot_linear_regressions
from utils import get_data
import pandas as pd

# csv_name = "all_data_1"
# csv_name = "all_data_lstm"
# csv_name = "category_repeating_500"
# csv_name = "category_repeating_noise_100"
# csv_name = "category_piecewise_linear_100"
# csv_name = "category_stock_fetch_stock_data_1"
csv_name = "category_multi_sinusoidal_100"

# data, data_lookback, loss_func = get_data("data")
# plot_all_ts(data)
# print(len(data))

results = pd.read_csv(f"results/{csv_name}.csv")

# dnn_categories = ["dnn_xs", "dnn_m"]
# cnn_categories = ["cnn_xs", "cnn_m"]
# lstm_categories = ["lstm_xs", "open_lstm_m"]
# ml_categories = ["classic_lstm","peephole_lstm","custom_lstm"] #
ml_categories = ["lstm", "dnn", "cnn"]
method_categories = [ "last_value", "regression", "arima", "knn"]
model_categories = ml_categories + method_categories
characteristics_s = ["mad", "num_edges", "compression"]
characteristics_e = ["sample_entropy", "perm_entropy", "spectral_entropy_psd", "spectral_entropy_fft"]
characteristics_d = ["noise", "period", "period_complexity"]
characteristics_t = ["mean_trend", "trend_complexity"]
list_model_categories = [ml_categories, method_categories, characteristics_s, characteristics_e, characteristics_d, characteristics_t]
plot_categories = model_categories + characteristics_s + characteristics_e + characteristics_d + characteristics_t
print(results[["name"] + plot_categories])

plot_model_comparisons(results, f"{csv_name}_scatter", plot_categories)
plot_correlation_matrix(results, f"{csv_name}_corr_mat", plot_categories)
plot_box(results, f"{csv_name}_box", model_categories)
plot_ridge(results, f"{csv_name}_ridge", model_categories)
double_plot_linear_regressions(results, f"{csv_name}_regressions", list_model_categories, path="all_results")
