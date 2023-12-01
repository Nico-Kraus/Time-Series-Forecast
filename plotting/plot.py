import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from plotting import PATH
from pathlib import Path

from utils import scale, create_dir

def plot_return_distribution(ts_df, name):
    """
    Plot the distribution of returns from the given time series.

    :param time_series: Array of time series data representing stock prices.
    """
    time_series = ts_df["values"]
    # Calculate returns as percentage change
    returns = np.diff(time_series) / time_series[:-1]

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.histplot(returns, kde=True, bins=50)
    plt.title('Distribution of Returns')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    create_dir(Path(PATH, "out/examples"))
    plt.savefig(Path(PATH, "out/examples", f"{name}.png"))



def plot_time_series(df, name):
    sns.set_style("darkgrid")
    sns.lineplot(data=df)
    create_dir(Path(PATH, "out/examples"))
    plt.savefig(Path(PATH, "out/examples", f"{name}.png"))


def plot_pred(train_df, val_df, val_pred, test_df, test_pred, lookback, name):
    train_df = train_df[lookback:].rename(columns={train_df.columns[0]: "train"})
    val_df = val_df[lookback:].assign(pred=val_pred)
    val_df = val_df.set_axis(["val", "val_pred"], axis=1)
    test_df = test_df[lookback:].assign(pred=test_pred)
    test_df = test_df.set_axis(["test", "test_pred"], axis=1)
    final = pd.concat([train_df, val_df, test_df], ignore_index=True, sort=False)

    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        x=final.index,
        y=final["train"],
        label="real train",
        color=(0, 0, 0.8, 0.5),
        linewidth=1,
    )
    sns.lineplot(
        x=final.index,
        y=final["val"],
        label="real val",
        color=(0, 0, 0.8, 0.5),
        linewidth=1,
    )
    sns.lineplot(
        x=final.index,
        y=final["val_pred"],
        label="pred val",
        color=(0, 0.8, 0, 0.5),
        linewidth=1,
    )
    sns.lineplot(
        x=final.index,
        y=final["test"],
        label="real test",
        color=(0, 0, 0.8, 0.5),
        linewidth=1,
    )
    sns.lineplot(
        x=final.index,
        y=final["test_pred"],
        label="pred test",
        color=(0.8, 0, 0, 0.5),
        linewidth=1,
    )
    plt.legend(loc="upper left")
    create_dir(Path(PATH, "out/predictions"))
    plt.savefig(Path(PATH, "out/predictions", f"{name}.png"), dpi=600)


def plot_linear_regressions(results, name, categories):
    with warnings.catch_warnings():
        sns.set_style("darkgrid")
        warnings.simplefilter("ignore")
        sns.set_theme()
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        for category in categories:
            if category in results.columns:
                df = results[["difficulty", category]]
                df = df.groupby("difficulty").mean().reset_index()
                sns.regplot(data=df, x="difficulty", y=category, ax=ax, label=category)
            else:
                warnings.warn(f"Column '{category}' does not exist in DataFrame.")

        ax.set_xlabel("Difficulty")
        ax.set_ylabel("Value")
        ax.set_title("Regression Plots")
        ax.legend()
        create_dir(Path(PATH, "out/results"))
        plt.savefig(Path(PATH, "out/results", f"{name}.png"))


def double_plot_linear_regressions(results, name, loss_categories, metric_categories):
    with warnings.catch_warnings():
        sns.set_style("darkgrid")
        warnings.simplefilter("ignore")
        sns.set_theme()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for category in loss_categories:
            if category in results.columns:
                df = results[["difficulty", category]]
                df = df.groupby("difficulty").mean().reset_index()
                sns.regplot(data=df, x="difficulty", y=category, ax=ax1, label=category)
            else:
                warnings.warn(f"Column '{category}' does not exist in DataFrame.")

        for category in metric_categories:
            if category in results.columns:
                df = results[["difficulty", category]]
                df = df.groupby("difficulty").mean().reset_index()
                sns.regplot(data=df, x="difficulty", y=category, ax=ax2, label=category)
            else:
                warnings.warn(f"Column '{category}' does not exist in DataFrame.")

        ax1.set_xlabel("Difficulty")
        ax2.set_xlabel("Difficulty")
        ax1.set_ylabel("Loss")
        ax2.set_ylabel("Metric")
        ax1.legend()
        ax2.legend()
        create_dir(Path(PATH, "out/results"))
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
    create_dir(Path(PATH, "out/results"))
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
        create_dir(Path(PATH, "out/results"))
        plt.savefig(Path(PATH, "out/results", f"{name}.png"))
