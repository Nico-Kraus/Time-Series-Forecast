from data.data import Data
from plotting.plot import plot_time_series

size = 1000
config = {"sinusoidal": {"period": 100}, "noise": {"std_dev": 0.2}}

df = Data(size=size, config=config, lookback=None).get(split=False)
print(df)
plot_time_series(df, f"plot")
