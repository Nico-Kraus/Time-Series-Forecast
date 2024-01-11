import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re

# Define the machine learning and method categories
ml_categories = ["lstm", "dnn", "cnn"]
method_categories = ["last_value", "regression", "arima", "knn"]

# Path to the directory containing the CSV files
directory_path = 'results/'

# Pattern to match the CSV files starting with 'category'
file_pattern = directory_path + 'category*.csv'

# List to store all dataframes
all_data = []

# Iterate over each CSV file
for file in glob.glob(file_pattern):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)

    # Extract the file name or category from the file path and clean it up
    category_name = re.sub(r'category_|_\d+', '', file.split('/')[-1].replace('.csv', ''))
    category_name = category_name.replace('_', ' ')

    # Keep only the relevant columns
    df = df[ml_categories + method_categories]
    df = df.melt(var_name='Model', value_name='Performance')
    df['Category'] = category_name

    # Append to the list
    all_data.append(df)

# Combine all data into a single DataFrame
combined_df = pd.concat(all_data)

# Define the models
models = combined_df['Model'].unique()

# Generate a Seaborn color palette with a distinct color for each model
palette = sns.color_palette("tab10", n_colors=len(models))

# Update the model_colors dictionary to map each model to a color from the palette
model_colors = {model: color for model, color in zip(models, palette)}

# Calculate mean performance for each model within each category
mean_performance = combined_df.groupby(['Category', 'Model'], as_index=False).mean()

# Sort the models within each category by mean performance
sorted_models = mean_performance.sort_values(by=['Category', 'Performance']).groupby('Category')['Model'].apply(list)

# Initialize a large figure to plot all categories
plt.figure(figsize=(20, 12))

# Get unique categories
categories = combined_df['Category'].unique()

max_performance = mean_performance['Performance'].max()

# Plot each category individually
for i, category in enumerate(categories):
    # Filter data for the category
    category_data = combined_df[combined_df['Category'] == category].copy()

    # Sort models in the order of performance
    category_data['Model'] = pd.Categorical(category_data['Model'], categories=sorted_models[category], ordered=True)
    category_data.sort_values(by='Model', inplace=True)

    # Plot using the updated model_colors for each model
    ax = plt.subplot(1, len(categories), i + 1)
    barplot = sns.barplot(x='Model', y='Performance', data=category_data, 
                          palette=[model_colors[model] for model in sorted_models[category]], ax=ax)

    # Customize x-axis labels
    plt.xticks(rotation=70, fontsize=10)
    ax.set_ylim(0, max_performance * 1.11)

    # Annotate bars with mean values
    for bar in barplot.patches:
        # Calculate the height to place the text: bar height + a small offset
        text_height = bar.get_height() + 0.0008
        # Annotate with the mean value
        ax.text(bar.get_x() + bar.get_width() / 2, text_height, 
                f'     {bar.get_height():.4f}', ha='center', va='bottom', fontsize=10, rotation=90)


    plt.xlabel('Models')
    plt.ylabel('Performance' if i == 0 else "")
    plt.title(category)

# Adjust subplots
plt.subplots_adjust(wspace=0.3)

# Adding overall title
plt.suptitle('Performance (L1 loss) of Models Across Different Categories (Sorted by Performance)')

# Creating a unified legend
handles = [plt.Rectangle((0,0),1,1, color=model_colors[model]) for model in model_colors]
plt.legend(handles, model_colors.keys(), title='Models', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

# Show the plot
plt.show()

