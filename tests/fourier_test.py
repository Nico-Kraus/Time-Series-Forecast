from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

N = 1000
T = 1.0 / 1000

x = np.linspace(0.0, N, N)
y = np.sin(200.0 * 2.0 * np.pi * x) + np.sin(80.0 * 2.0 * np.pi * x)

yf = fft(y)
xf = fftfreq(N, T)

plt.plot(np.abs(xf), np.abs(yf))
plt.grid()
plt.show()
