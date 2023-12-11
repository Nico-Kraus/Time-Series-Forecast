from data.data import Data
from trainer.knn_predictor import KNN_Predictor
from plotting.plot import plot_pred
from utils import get_params, keys_to_string
import pandas as pd

lookback = 10
loss = "L1"

size = 100
seed = 42
config = {"sinusoidal": {"period": 20}}

data = Data(size=size, seed=seed, config=config, lookback=lookback)
train_df, val_df, test_df = data.get(split=(0.8, 0.1, 0.1))

trainer = KNN_Predictor(lookback=lookback, loss=loss)
val_loss, val_pred = trainer.test(train_df, val_df)
new_train_df = pd.concat([train_df, val_df])
new_train_df = new_train_df[~new_train_df.index.duplicated(keep='first')]
test_loss, test_pred = trainer.test(new_train_df, test_df)

plot_pred(
    train_df=train_df,
    val_df=val_df,
    val_pred=val_pred,
    test_df=test_df,
    test_pred=test_pred,
    lookback=lookback,
    name=f"plot_knn_{keys_to_string(config.keys())}",
)
