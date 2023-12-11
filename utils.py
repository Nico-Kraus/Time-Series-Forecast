import yaml
import os

import pandas as pd
import numpy as np
import EntropyHub as EH

from trainer.trainer import Trainer
from trainer.predictor import Predictor
from trainer.knn_predictor import KNN_Predictor

def df_to_csv(df,name):
    create_dir("results")
    df.to_csv(f"results/{name}")


def create_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


def keys_to_string(keys):
    name = f"{list(keys)}"
    symbols_to_remove = "[]'"
    for symbol in symbols_to_remove:
        name = name.replace(symbol, "")
    return name.replace(",", "_")


def dict_to_string(d, parent_key=""):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}_{k}" if parent_key else k
        if isinstance(v, dict):
            items.append(
                dict_to_string(v, new_key)
            )  # Recursive call for nested dictionaries
        else:
            items.append(f"{new_key}_{v}")
    return "_".join(items)


def create_name(type, data_params, max_difficulty):
    return f"{type}_{keys_to_string(data_params['config'].keys())}_{max_difficulty}"


def scale(l):
    max_value = max(l)
    return [x / max_value for x in l]

def get_data(name):
    with open(f"{name}.yaml", "r") as file:
        pre_data = yaml.safe_load(file)
    result = {}
    for name, config in pre_data["configs"].items():
        seed = config["seed"] if "seed" in config else pre_data["seed"]
        config.pop("seed", None)
        result[name] = {"size": pre_data["size"], "seed": seed, "lookback": pre_data["data_lookback"], "config": config}
    return result, pre_data["data_lookback"], pre_data["loss_func"]


def get_params(name, folder="params"):
    if folder == None:
        with open(f"{name}.yaml", "r") as file:
            return yaml.safe_load(file)
    with open(f"params/{name}.yaml", "r") as file:
        return yaml.safe_load(file)


def get_model_test_loss(train_df, val_df, test_df, trainer_params):
    trainer = Trainer(**trainer_params)
    trainer.train(train_df, val_df, info=False)
    test_loss, pred = trainer.test(test_df, info=False)
    return test_loss


def get_prediction_loss(train_df, val_df, test_df, method, loss, lookback):
    if method == "knn":
        predictor = KNN_Predictor(lookback=lookback, loss=loss)
        full_train_df = pd.concat([train_df, val_df])
        full_train_df = full_train_df[~full_train_df.index.duplicated(keep='first')]
        test_loss, test_pred = predictor.test(full_train_df, test_df, info=False)
        return test_loss
    else:
        predictor = Predictor(lookback=lookback, method=method, loss=loss)
        test_loss, test_pred = predictor.test(test_df, info=False)
        return test_loss


def get_sample_entropy(train_df, val_df, test_df, m, tau):
    X = pd.concat([train_df["values"], val_df["values"], test_df["values"]]).to_numpy()
    Samp, Phi1, Phi2 = EH.SampEn(X, m=10, **({} if tau == 0 else {"tau": tau}))
    mod_Sample = [0 if np.isinf(s) or np.isnan(s) else s for s in Samp]
    return sum(mod_Sample)
