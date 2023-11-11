import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

from data.data import Data
from plotting.plot import plot_time_series

data_params = {"number": 20, "min_value": 0, "max_value": 1, "size": 10000}
category = "multi_sinusoidal"

data_df = Data(category, data_params, lookback=None).get()
print(data_df)
plot_time_series(data_df, f"fourier_data")

yf = fft(data_df["values"].values)
xf = fftfreq(10000, 1 / 10000)

print(yf)

plt.clf()
plt.plot(np.abs(xf)[:100], np.abs(yf)[:100])
plt.grid()
plt.show()
