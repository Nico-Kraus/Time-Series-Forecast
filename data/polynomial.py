import numpy as np

def polynomial(coefficients, size =1000,  **params):
    x = np.arange(size)
    y = np.zeros(size)
    for i, coeff in enumerate(coefficients):
        y += coeff * x**i
    return y