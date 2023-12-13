from data.data import Data
from trainer.trainer import Trainer
from plotting.plot import plot_pred
from utils import get_params, keys_to_string

params_file = "open_lstm_m"
trainer_params = get_params(params_file)
lookback = trainer_params["lookback"]

size = 1000
# seed = 42
# config = {"probabilistic_discret": {"n": 20, "m": 2, "min_p": 0.01, "max_p": 0.9, "first_p": 0.8}}
# seed = 42
# config = {"probabilistic_discret": {"n": 6, "m": 3, "min_p": 0.01, "max_p": 0.9, "first_p": 0.9}}
# seed = 44
# config = {"probabilistic_discret": {"n": 10, "m": 4, "min_p": 0.01, "max_p": 0.9, "first_p": 0.95}}
seed = 44
config = {"probabilistic_discret": {"n": 15, "m": 2, "min_p": 0.01, "max_p": 0.9, "first_p": 0.95}, "noise": {"std_dev": 0.1}}

data = Data(size=size, seed=seed, config=config, lookback=lookback)
train_df, val_df, test_df = data.get(split=(0.8, 0.1, 0.1))

trainer = Trainer(**trainer_params)
train_loss, val_loss, val_pred = trainer.train(train_df, val_df)
val_loss, val_pred = trainer.test(val_df)
test_loss, test_pred = trainer.test(test_df)

plot_pred(
    train_df=train_df,
    val_df=val_df,
    val_pred=val_pred,
    test_df=test_df,
    test_pred=test_pred,
    lookback=lookback,
    name=f"plot_{params_file}_1_{keys_to_string(config.keys())}",
)
