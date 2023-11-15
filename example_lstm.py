from data.data import Data
from trainer.trainer import Trainer
from plotting.plot import plot_pred
from utils import get_params

trainer_params = get_params("lstm_params.yaml")
lookback = trainer_params["lookback"]

size = 1000
seed = 44
config = {"multi_sinusoidal": {"num_sin": 3}}

data = Data(size=size, seed=seed, config=config, lookback=lookback)
train_df, val_df, test_df = data.get(split=(0.8, 0.1, 0.1))

trainer = Trainer(**trainer_params)
train_loss, val_loss, val_pred = trainer.train(train_df, val_df)
test_loss, test_pred = trainer.test(test_df)

plot_pred(
    train_df=train_df,
    val_df=val_df,
    val_pred=val_pred,
    test_df=test_df,
    test_pred=test_pred,
    lookback=lookback,
    name="plot",
)
