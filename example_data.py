from data.data import Data
from plotting.plot import plot_time_series, plot_return_distribution

size = 1000
seed = 40
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
# config = {"piecewise_constant": {"num_seg": 5}} # seed 39 for val loss 0
# config = {"uniform_piecewise_linear": {"num_seg": 10}}
# config = {"trend": {"max_return": 0.1, "trend": 0.2}}
# config = {
#     "logistic": {"max_capacity": 200,"growth": 1,"midpoint": 500}, 
#     "multi_sinusoidal": {"num_sin": 2, "min_value": 0, "max_value": 0.4},
#     "trend": {"max_return": 0.1, "trend": 0.1, "min_value":0, "max_value": 0.6},
#     "noise": {"std_dev": 0.01}
# }
# config = {"uci_synthetic_control": {"data_type": "cyclic", "number": 3}} # size = 60
# config = {"uci_gait": {"start": 20000}} # size = 181800
config = {"fetch_stock_data": {"symbol": "AAPL", "data_type": "Close", "interval": "1m", "start_date": "2023-12-04"}}
# config = {"fetch_stock_data": {"symbol": "AAPL", "data_type": "Close", "interval": "5d", "start_date": "2000-01-01"}}
# config = {"fetch_stock_data": {"symbol": "AAPL", "data_type": "Close", "interval": "1h", "start_date": "2022-01-01"}}
# config = {"probabilistic_discret": {"n": 10, "m": 5, "min_p": 0.1, "max_p": 0.9}}
# config = {"icmc_usp": {"name": "sd_constant_level", "start": 0}}
# config = {"uci_air_quality": {"start": 0, "name": "CO(GT)"}} # 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'
# config = {"uci_electricity": {"start": 0, "name": "Global_active_power"}} # 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
# config = {"uci_ai_maintenance": {"start": 0, "name": "Air temperature [K]"}} #  'Air temperature [K]','Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]','Tool wear [min]', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF','RNF'

train, val, test = Data(size=size, seed=seed, config=config, lookback=0).get(split=(0.8,0.1,0.1))
print(test)
plot_time_series(train, val, test, f"timeseries")

