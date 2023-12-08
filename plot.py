from plotting.plot import double_plot_linear_regressions, double_plot_linear_regressions_entropy
import pandas as pd

names = ["entrophy_piecewise_constant_100", "entrophy_piecewise_linear_100_lb_10"]
df_list = []
for name in names:
    df_list.append(pd.read_csv(f"results/{name}.csv"))
results = pd.concat(df_list)
print(results)

# plot_categories = [
#     ["lstm_xs", "peephole_lstm_xs", "open_lstm_xs", "input_lstm_xs"],
#     ["lstm_s", "peephole_lstm_s", "open_lstm_s", "input_lstm_s"],
#     ["lstm_m", "peephole_lstm_m", "open_lstm_m", "input_lstm_m"],
#     ["last_value_loss", "input_lstm_m", "open_lstm_m"]
# ]

# plot_categories = [
#     ["lstm", "very_small_lstm"],
#     ["entropy"],
#     ["last_value_loss"]
# ]

plot_categories = [
    ["last_value_loss", "lstm_m", "peephole_lstm_m"],
    ["last_value_loss", "input_lstm_m", "open_lstm_m"],
    ["input_lstm_m", "open_lstm_m"]
]

double_plot_linear_regressions_entropy(results, f"{name}_entropy", plot_categories)
