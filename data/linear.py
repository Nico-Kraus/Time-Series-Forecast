import numpy as np

def linear(size = 1000, slope = 1, intercept = 0, **params):
    x = np.arange(size)
    return slope * x + intercept