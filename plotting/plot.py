import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from plotting import PATH
from pathlib import Path

from utils import scale, create_dir
from data.data import Data

def plot_box(df, filename, categories):
    melted_df = df.melt(value_vars=categories, var_name='Category', value_name='Value')

    medians = melted_df.groupby('Category')['Value'].median().sort_values()
    sorted_categories = medians.index.tolist()

    plt.figure(figsize=(10, 6))
    # sns.boxplot(data=melted_df, x='Category', y='Value')
    sns.boxplot(data=melted_df, x='Category', y='Value', order=sorted_categories)

    plt.tight_layout()
    plt.ylim(0,0.12)

    create_dir(Path(PATH, "out/all_results"))
    plt.savefig(Path(PATH, "out/all_results", f"{filename}.png"))

def plot_ridge(df, filename, categories):
    melted_df = df.melt(value_vars=categories, var_name='Category', value_name="Value")

    # Calculate mean values and sort categories
    category_means = melted_df.groupby('Category')["Value"].mean().sort_values()
    sorted_categories = category_means.index.tolist()

    # Filter melted_df based on sorted categories
    melted_df['Category'] = pd.Categorical(melted_df['Category'], categories=sorted_categories, ordered=True)
    melted_df = melted_df.sort_values('Category')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        g = sns.FacetGrid(melted_df, row="Category", hue="Category", aspect=15, height=1.5, palette="viridis")

        g.map(sns.kdeplot, "Value", bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
        g.map(sns.kdeplot, "Value", clip_on=False, color="w", lw=2, bw_adjust=0.5)
        g.map(plt.axhline, y=0, lw=2, clip_on=False)

        def plot_mean(data, **kwargs):
            plt.axvline(data.mean(), ymin=0, ymax=0.6, color="r", linestyle="--")

        g.map(plot_mean, "Value")

        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, 0.2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

        g.map(label, "Value")

    g.fig.subplots_adjust(hspace=-0.4)
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    plt.xlim(-0.1,0.4)

    create_dir(Path(PATH, "out/all_results"))
    plt.savefig(Path(PATH, "out/all_results", f"{filename}.png"))

def plot_correlation_matrix(df, filename, categories):
    mean_df = df.groupby("name").mean()
    corr = mean_df[categories].corr()

    for category in categories:
        category_df = df[["name", "repeats", category]]
        pivoted_df = category_df.pivot(index="name", columns="repeats", values=category)
        pivoted_df = pivoted_df.dropna()
        if pivoted_df.shape[0] > 1:  # Check to ensure there are at least 2 rows for correlation calculation
            category_corr = pivoted_df.corr()
            avg_corr = (category_corr.sum().sum() - category_corr.shape[0]) / (category_corr.size - category_corr.shape[0])
            corr.loc[category, category] = avg_corr

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, vmin=-1, vmax=1)

    plt.tight_layout()

    create_dir(Path(PATH, "out/all_results"))
    plt.savefig(Path(PATH, "out/all_results", f"{filename}.png"))

def plot_model_comparisons(df, filename, plot_categories):
    num_plots = len(plot_categories)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))  # Adjust figsize as needed

    df['index'] = range(len(df))
    for idx, category in enumerate(plot_categories):
        ax = axes[idx]

        sns.scatterplot(data=df, x="index", y=category, ax=ax, s=20)  # Adjust point size with 's'

        ax.set_title(category)
        ax.set_xticks([])

    plt.tight_layout()

    create_dir(Path(PATH, "out/all_results"))
    plt.savefig(Path(PATH, "out/all_results", f"{filename}.png"))

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

def plot_all_ts(data):
    sns.set_style("darkgrid")

    num_series = len(data)
    fig, axes = plt.subplots(num_series, 1, figsize=(10, 1.8*num_series)) 

    for idx, (name, data_params) in enumerate(data.items()):
        # Generate data for each time series
        train_df, val_df, test_df = Data(**data_params).get(split=(0.8, 0.1, 0.1))

        # Select the current axis
        ax = axes[idx] if num_series > 1 else axes

        # Plot each time series on its own subplot
        sns.lineplot(x=train_df.index, y=train_df["values"], label="train", color=(0, 0, 0.8, 0.8), ax=ax)
        sns.lineplot(x=val_df.index, y=val_df["values"], label="val", color=(0, 0.8, 0, 0.8), ax=ax)
        sns.lineplot(x=test_df.index, y=test_df["values"], label="test", color=(0.8, 0, 0, 0.8), ax=ax)

        # Set the ylabel for each subplot
        ax.set_ylabel(name)
        ax.set_xlabel("")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(Path(PATH, "out/examples", "all_time_series.png"))


def plot_time_series(train, val, test, name):
    sns.set_style("darkgrid")
    sns.lineplot(x=train.index, y=train["values"], label="train", color=(0, 0, 0.8, 0.8))
    sns.lineplot(x=val.index, y=val["values"], label="val", color=(0, 0.8, 0, 0.8))
    sns.lineplot(x=test.index, y=test["values"], label="test", color=(0.8, 0, 0, 0.8))
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
        plt.savefig(Path(PATH, "out/results", f"{name}.png"), dpi=600)
        
def double_plot_linear_regressions_entropy(results, name, categories):
    with warnings.catch_warnings():
        sns.set_style("darkgrid")
        warnings.simplefilter("ignore")
        sns.set_theme()
        num_cat_lists = len(categories)
        rows = int(np.ceil(num_cat_lists / 2))
        fig, axs = plt.subplots(rows, 2, figsize=(16, 8))

        i = 0
        for categories_n in categories:
            for category in categories_n:
                if category in results.columns:
                    df = results[["entropy", category]]
                    sns.regplot(data=df, x="entropy", y=category, ax=axs[i//2][i%2], label=category, order=2,
                                line_kws={"linewidth":1}, scatter_kws={"s":3})
                else:
                    warnings.warn(f"Column '{category}' does not exist in DataFrame.")
            i=i+1;

        for axs_ in axs:
            for ax in axs_:
                ax.set_ylim(0, 0.08)
                ax.set_xlabel("Entropy")
                ax.set_ylabel("Metric")
                ax.legend()
        create_dir(Path(PATH, "out/results"))
        plt.savefig(Path(PATH, "out/results", f"{name}.png"),dpi=600)

def double_plot_linear_regressions(results, name, categories, path="results"):
    with warnings.catch_warnings():
        sns.set_style("darkgrid")
        warnings.simplefilter("ignore")
        sns.set_theme()
        num_cat_lists = len(categories)
        rows = int(np.ceil(num_cat_lists / 2))
        fig, axs = plt.subplots(rows, 2, figsize=(16, 8))

        i = 0
        for categories_n in categories:
            for category in categories_n:
                if category in results.columns:
                    if 'difficulty' in results.columns:
                        df = results[["difficulty", category]]
                        df = df.groupby("difficulty").mean().reset_index()
                    else:
                        df = results[[category]].copy()
                        df.index.name = 'difficulty'
                        df = df.reset_index()
                    df = df.rename(columns={'difficulty': 'x'})
                    sns.regplot(data=df, x="x", y=category, ax=axs[i//2][i%2], label=category, order=1,
                                line_kws={"linewidth":1}, scatter_kws={"s":3})
                else:
                    warnings.warn(f"Column '{category}' does not exist in DataFrame.")
            i=i+1;

        for axs_ in axs:
            for ax in axs_:
                ax.set_xlabel("Changing parameter")
                ax.set_ylabel("Metric")
                ax.legend()
        create_dir(Path(PATH, f"out/{path}"))
        plt.savefig(Path(PATH, f"out/{path}", f"{name}.png"),dpi=600)


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
