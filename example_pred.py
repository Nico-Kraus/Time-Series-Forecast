from data.data import Data
from trainer.predictor import Predictor
from plotting.plot import plot_pred
from utils import get_params, keys_to_string

lookback = 10
method = "last_value"
loss = "L1"

size = 1000
seed = 39
config = {"piecewise_constant": {"num_seg": 5}}

data = Data(size=size, seed=seed, config=config, lookback=lookback)
train_df, val_df, test_df = data.get(split=(0.8, 0.1, 0.1))

trainer = Predictor(lookback=lookback, method=method, loss=loss)
val_loss, val_pred = trainer.test(val_df)
test_loss, test_pred = trainer.test(test_df)

plot_pred(
    train_df=train_df,
    val_df=val_df,
    val_pred=val_pred,
    test_df=test_df,
    test_pred=test_pred,
    lookback=lookback,
    name=f"plot_pred_{keys_to_string(config.keys())}",
)
