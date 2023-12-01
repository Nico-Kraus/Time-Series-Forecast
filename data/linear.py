import numpy as np

def linear(slope, intercept, size, **params):
    x = np.arange(size)
    return slope * x + intercept