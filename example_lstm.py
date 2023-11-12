from data.data import Data
from trainer.trainer import Trainer
from utils import get_params
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

trainer_params = get_params("lstm_params.yaml")
lookback = trainer_params["lookback"]

size = 1000
seed = 42
config = {"piecewise_linear": {"num_seg": 10}}

data = Data(size=size, seed=seed, config=config, lookback=lookback)
train_df, val_df, test_df = data.get(split=(0.8, 0.1, 0.1))

trainer = Trainer(**trainer_params)
trainer.train(train_df)
val_loss, val_pred = trainer.val(val_df)
test_loss, test_pred = trainer.val(test_df)

train_df = train_df[lookback + 1 :].rename(columns={train_df.columns[0]: "train"})
val_df = val_df[lookback + 1 :].assign(pred=val_pred)
val_df = val_df.set_axis(["val", "val_pred"], axis=1)
test_df = test_df[lookback + 1 :].assign(pred=test_pred)
test_df = test_df.set_axis(["test", "test_pred"], axis=1)
final = pd.concat([train_df, val_df, test_df], ignore_index=True, sort=False)


sns.set_style("darkgrid")
plt.figure(figsize=(12, 7))
sns.lineplot(x=final.index, y=final["train"], label="real train")
sns.lineplot(x=final.index, y=final["val"], label="real val")
sns.lineplot(x=final.index, y=final["val_pred"], label="pred val")
sns.lineplot(x=final.index, y=final["test"], label="real test")
sns.lineplot(x=final.index, y=final["test_pred"], label="pred test")
plt.legend(loc="upper left")
plt.show()
