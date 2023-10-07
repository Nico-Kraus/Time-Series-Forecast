from data.data import Data
from plotting.plot import plot_time_series

category = "sinusoidal_with_noise"
size = 1000
difficulty = 50
std_dev = 1

df = Data(category, size, difficulty, None).get(split=False)
print(df)
plot_time_series(df, f"{category}")
