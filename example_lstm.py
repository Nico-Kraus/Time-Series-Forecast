from data.data import Data
from trainer.trainer import Trainer
from utils import get_params
import matplotlib.pyplot as plt

trainer_params = get_params("lstm_params.yaml")

data_params = {"difficulty": 100, "size": 1000}
category = "piecewise_linear"

train_df, val_df = Data(category, data_params, lookback=trainer_params["lookback"]).get(
    split=(0.8, 0.2)
)

trainer = Trainer(**trainer_params)
trainer.train(train_df)
trainer.val(val_df)

plt.figure(figsize=(10, 6))
train_df.plot()
plt.title("Generated Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
# plt.show()
