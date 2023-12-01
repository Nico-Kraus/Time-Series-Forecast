import numpy as np

def logistic(max_capacity, growth, midpoint, size = 1000, **params):
    """
    Generate a logistic time series.

    :param max_capacity: The maximum value of the logistic function (e.g., total population susceptible).
    :param growth_rate: The steepness of the curve's increase.
    :param midpoint: The point of inflection where the growth rate starts to decrease.
    :param time_period: The total time period for the simulation (e.g., number of days).
    :return: Array representing the logistic time series.
    """
    time_steps = np.arange(size)
    return max_capacity / (1 + np.exp(-(growth/100) * (time_steps - midpoint)))
