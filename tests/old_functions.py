import numpy as np

from data import Data
from trainer import Trainer


def run(
    trainer_params,
    category,
    data_params,
    increasing_param,
    repeats=1,
    stop_delta=0.01,
    max_inc=10,
    max_hidden_dim=63,
    info=True,
):
    results = {}
    for inc in range(1, max_inc + 1):
        hidden_dim_solved = []
        for repeat in range(repeats):
            for hidden_dim in range(1, max_hidden_dim + 1):
                data_params[increasing_param] = inc
                trainer_params["hidden_dim"] = hidden_dim

                repeating_df = Data(
                    category, data_params, lookback=trainer_params["lookback"]
                ).get()

                trainer = Trainer(**trainer_params)
                last_epoch, losses = trainer.train(
                    x_train=repeating_df, stop_delta=stop_delta, info=False
                )
                if last_epoch != None:
                    hidden_dim_solved.append(hidden_dim)
                    break
                elif hidden_dim == max_hidden_dim:
                    print(f"Warning could not solve the problem with sl {inc}")
        avg_hidden_dimensions = np.mean(hidden_dim_solved)
        results[inc] = hidden_dim_solved
        if info:
            if last_epoch != None:
                print(
                    f"=== Problem with increasing {increasing_param}: {inc} solved with average {avg_hidden_dimensions} ==="
                )

    return (
        pd.DataFrame(
            list(results.items()), columns=["Sequence length", "Hidden dimensions"]
        )
        .explode("Hidden dimensions")
        .apply(pd.to_numeric)
    )
