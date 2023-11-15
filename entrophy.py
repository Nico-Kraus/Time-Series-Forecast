from utils import get_params, get_model_test_loss, get_sample_entropy
from data.data import Data
from plotting.plot import plot_linear_regressions


trainer_params = get_params("lstm_params.yaml")

data_params = {"size": 1000, "seed": 44, "config": {"repeating": {"period": 3}}}
data_params["lookback"] = trainer_params["lookback"]

max_difficulty = 100
repeats = 10

results = {}
for difficulty in range(1, max_difficulty + 1):
    print(f"{difficulty} ", end="", flush=True)
    data_params["config"]["repeating"]["period"] = difficulty
    losses, entropies = [], []
    for i in range(repeats + 1):
        train_df, val_df, test_df = Data(**data_params).get(split=(0.8, 0.1, 0.1))
        entropies.append(get_sample_entropy(train_df, val_df, test_df, m=10, tau=0))
        losses.append(get_model_test_loss(train_df, val_df, test_df, trainer_params))

    results[difficulty] = {
        "entropy": sum(entropies) / len(entropies),
        "val_loss": sum(losses) / len(losses),
    }

print()
name = f"entrophy_{data_params['category']}_{max_difficulty}"
plot_linear_regressions(results, name)
