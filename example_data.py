from data.data import Data
from plotting.plot import plot_time_series, plot_return_distribution

size = 1000
seed = None
# config = {"piecewise_linear": {"num_seg": 10}, "noise": {"std_dev": 0.01}}
# config = {"sinusoidal": {"period": 500}, "noise": {"std_dev": 0.3}}
# config = {"sinusoidal": {"period": 500}, "noise": {"std_dev": 0.01}}
config = {"multi_sinusoidal": {"num_sin": 100}}
# config = {"multi_sinusoidal": {"num_sin": 2},"piecewise_linear": {"num_seg": 10}, "noise": {"std_dev": 0.01}}
# config = {"trend": {"change_rate": 1}}

df = Data(size=size, seed=seed, config=config, lookback=None).get(split=False)
print(df)
plot_time_series(df, f"timeseries")
plot_return_distribution(df, f"distribution")

