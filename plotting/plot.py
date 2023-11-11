import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plotting import PATH
from pathlib import Path

from utils import scale

from utils import create_dir


def plot_time_series(df, name):
    sns.set_style("darkgrid")
    sns.lineplot(data=df)
    create_dir(Path(PATH, "out/examples"))
    plt.savefig(Path(PATH, "out/examples", f"{name}.png"))


def plot_linear_regressions(results, name):
    x_values = list(results.keys())
    y1_values = scale([res["entropy"] for res in results.values()])
    y2_values = scale([res["val_loss"] for res in results.values()])
    df1 = pd.DataFrame({"difficulty": x_values, "entropy": y1_values})
    df2 = pd.DataFrame({"difficulty": x_values, "val_loss": y2_values})
    with warnings.catch_warnings():
        sns.set_style("darkgrid")
        warnings.simplefilter("ignore")
        sns.set_theme()
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        sns.regplot(data=df1, x="difficulty", y="entropy", ax=ax, label="Entropy")
        sns.regplot(
            data=df2,
            x="difficulty",
            y="val_loss",
            ax=ax,
            label="Validation Loss",
            color="r",
        )

        ax.set_xlabel("Difficulty")
        ax.set_ylabel("Value")
        ax.set_title("Regression Plots")
        ax.legend()

        plt.savefig(Path(PATH, "out/results", f"{name}.png"))


def plot_two_graphs(results, name):
    x_values = list(results.keys())
    y1_values = scale([res["entropy"] for res in results.values()])
    y2_values = scale([res["val_loss"] for res in results.values()])

    sns.lineplot(x=x_values, y=y1_values)
    sns.lineplot(x=x_values, y=y2_values)

    plt.title("Line Plot of the Results Dictionary")
    plt.xlabel("Index")
    plt.ylabel("Blue is Value, Orange is Loss")

    plt.savefig(Path(PATH, "out/results", f"{name}.png"))


def plot_linear_regression(df, name):
    x_name = df.columns[0]
    y_name = df.columns[1]
    with warnings.catch_warnings():
        sns.set_style("darkgrid")
        warnings.simplefilter("ignore")
        sns.set_theme()
        g = sns.lmplot(data=df, x=x_name, y=y_name, height=5)
        g.set_axis_labels(x_name, y_name)
        plt.savefig(Path(PATH, "out/results", f"{name}.png"))
