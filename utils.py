import yaml

import pandas as pd
import numpy as np
import EntropyHub as EH

from trainer.trainer import Trainer


def scale(l):
    max_value = max(l)
    return [x / max_value for x in l]


def get_params(name):
    with open(name, "r") as file:
        return yaml.safe_load(file)


def get_model_val_loss(train_df, val_df, trainer_params):
    trainer = Trainer(**trainer_params)
    trainer.train(train_df, info=False)
    return trainer.val(val_df, info=False)


def get_sample_entropy(train_df, val_df, m, tau):
    X = pd.concat([train_df["values"], val_df["values"]]).to_numpy()
    Samp, Phi1, Phi2 = EH.SampEn(X, m=10, **({} if tau == 0 else {"tau": tau}))
    mod_Sample = [0 if np.isinf(s) or np.isnan(s) else s for s in Samp]
    return sum(mod_Sample)
