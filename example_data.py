from data.data import Data
from plotting.plot import plot_time_series

size = 200
seed = 42
config = {"multi_sinusoidal": {"num_sin": 2}, "noise": {"std_dev": 0.1}}

df = Data(size=size, seed=seed, config=config, lookback=None).get(split=False)
print(df)
plot_time_series(df, f"plot")
