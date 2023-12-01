from data.data import Data
from plotting.plot import plot_time_series, plot_return_distribution

size = 1000
seed = None
# config = {"linear": {"slope": 1, "intercept": 0}}
# config = {"polynomial": {"coefficients": [0,-1000,1],}} #quadratic
# config = {"polynomial": {"coefficients": [0,525000,-1500,1],}} #cubic
# config = {"exponential": {"factor": 1, "growth": 0.5}}
# config = {"logistic": {"max_capacity": 1,"growth": 1,"midpoint": 500}}
# config = {"sinusoidal": {"period": 500, "phase_offset": 100}}
# config = {"sinusoidal": {"period": 500, "phase_offset": 100}, "noise": {"std_dev": 0.1}}
# config = {"multi_sinusoidal": {"num_sin": 3}}
# config = {"multi_sinusoidal": {"num_sin": 100}}
# config = {"piecewise_linear": {"num_seg": 10}}
# config = {"uniform_piecewise_linear": {"num_seg": 10}}
# config = {"trend": {"max_return": 0.1, "trend": 0.2}}
config = {
    "logistic": {"max_capacity": 2,"growth": 1,"midpoint": 500}, 
    "multi_sinusoidal": {"num_sin": 2, "min_value": 0, "max_value": 0.4},
    "trend": {"max_return": 0.1, "trend": 0.1, "min_value":0, "max_value": 0.6},
    "noise": {"std_dev": 0.01}
}

df = Data(size=size, seed=seed, config=config, lookback=None).get(split=False)
print(df)
plot_time_series(df, f"timeseries")
plot_return_distribution(df, f"distribution")

