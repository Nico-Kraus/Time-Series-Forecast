from data.data import Data
from trainer.trainer import Trainer
from utils import get_params
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

trainer_params = get_params("lstm_params.yaml")
lookback = trainer_params["lookback"]

size = 1000
config = {"piecewise_linear": {"num_seg": 10}}

data = Data(size=size, config=config, lookback=lookback)
train_df, val_df = data.get(split=(0.8, 0.2))

trainer = Trainer(**trainer_params)
trainer.train(train_df)
val_loss, pred = trainer.val(val_df)


train_df = train_df[lookback + 1 :].rename(columns={train_df.columns[0]: "train"})
val_df = val_df[lookback + 1 :].assign(pred=pred).set_axis(["val", "pred"], axis=1)
final = pd.concat([train_df, val_df], ignore_index=True, sort=False)


sns.set_style("darkgrid")
plt.figure(figsize=(12, 7))
sns.lineplot(x=final.index, y=final["train"], label="train")
sns.lineplot(x=final.index, y=final["val"], label="val")
sns.lineplot(x=final.index, y=final["pred"], label="pred")
plt.legend(loc="upper left")
plt.show()
