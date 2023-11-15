import yaml
import os

import pandas as pd
import numpy as np
import EntropyHub as EH

from trainer.trainer import Trainer


def create_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


def scale(l):
    max_value = max(l)
    return [x / max_value for x in l]


def get_params(name):
    with open(name, "r") as file:
        return yaml.safe_load(file)


def get_model_test_loss(train_df, val_df, test_df, trainer_params):
    trainer = Trainer(**trainer_params)
    trainer.train(train_df, val_df, info=False)
    test_loss, pred = trainer.test(test_df, info=False)
    return test_loss


def get_sample_entropy(train_df, val_df, test_df, m, tau):
    X = pd.concat([train_df["values"], val_df["values"], test_df["values"]]).to_numpy()
    Samp, Phi1, Phi2 = EH.SampEn(X, m=10, **({} if tau == 0 else {"tau": tau}))
    mod_Sample = [0 if np.isinf(s) or np.isnan(s) else s for s in Samp]
    return sum(mod_Sample)
