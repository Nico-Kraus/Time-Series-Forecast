import numpy as np

def sigmoid(x, k=1):
    """ Sigmoid function"""
    return 1 / (1 + np.exp(-k * x)) - 0.5

def trend(rng, size, max_return=0.05, change_rate=0.1, limit=0.95):
    limit = max(0.01,min(0.99, limit))
    data = [rng.random()]
    prob_growth = 0.5
    consecutive_trend = 0
    last_direction = None

    for _ in range(1, size):
        r = rng.random() * max_return
        is_growth = rng.random() < prob_growth

        if is_growth:
            data.append(data[-1] * (1 + r))
            if last_direction == 'growth':
                consecutive_trend += 1
            else:
                consecutive_trend = 1
            last_direction = 'growth'
        else:
            data.append(data[-1] * (1 - r))
            if last_direction == 'decline':
                consecutive_trend += 1
            else:
                consecutive_trend = 1
            last_direction = 'decline'

        # Adjust probability using sigmoid function
        scaled_trend = limit * sigmoid(consecutive_trend, k=change_rate)
        prob_growth = 0.5 + (scaled_trend if last_direction == 'growth' else - scaled_trend)

    return np.array(data)