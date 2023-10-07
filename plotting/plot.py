import matplotlib.pyplot as plt
import seaborn as sns
from plotting import PATH
from pathlib import Path
import warnings


def plot_time_series(df, name):
    sns.set_style("darkgrid")
    sns.lineplot(data=df)
    plt.savefig(Path(PATH, "out", f"{name}.png"))


def plot_linear_regression(df, name):
    x_name = df.columns[0]
    y_name = df.columns[1]
    with warnings.catch_warnings():
        sns.set_style("darkgrid")
        warnings.simplefilter("ignore")
        sns.set_theme()
        g = sns.lmplot(data=df, x=x_name, y=y_name, height=5)
        g.set_axis_labels(x_name, y_name)
        plt.savefig(Path(PATH, "out", f"{name}.png"))
