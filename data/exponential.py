import numpy as np

def exponential(factor, growth, size=1000,  **params):
    x = np.arange(size)
    return factor * (1 + growth / 100) ** x