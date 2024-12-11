import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from adjustText import adjust_text

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text

def simi_sampler(data, class_label, compare_with=0, plot=False, sample_size=None):
    # Identify the relevant feature columns, excluding 'class' and 'flag'
    # add flag column equal to index if not present
    if 'flag' not in data.columns:
        data['flag'] = data.index
        
    vars = data.columns[(data.columns != 'class') & (data.columns != 'flag') & (data.columns != 'Sample')]
    

    # Convert feature columns to numeric, coerce errors to NaN
    sampler_data = data[vars].apply(pd.to_numeric, errors='coerce')
    # Check for NaN and infinite values before handling
    nan_count = sampler_data.isna().sum().sum()
    inf_count = np.isinf(sampler_data.values).sum()

    sampler_data.fillna(sampler_data.mean(), inplace=True)
  

    # Remove columns with zero variance (constant columns)
    zero_var_cols = sampler_data.columns[sampler_data.nunique() <= 1]
    if len(zero_var_cols) > 0:
        print(f"Dropping zero variance columns: {zero_var_cols.tolist()}")
        sampler_data.drop(columns=zero_var_cols, inplace=True)
        vars = sampler_data.columns  # Update vars
    else:
        print("No zero variance columns to drop.")

    # Check for NaN and infinite values after handling
    nan_count_after = sampler_data.isna().sum().sum()
    inf_count_after = np.isinf(sampler_data.values).sum()
    print(f"Total NaN values after handling: {nan_count_after}")
    print(f"Total infinite values after handling: {inf_count_after}")

    # Proceed with scaling
    scaler = StandardScaler()
   
    sampler_data_scaled = pd.DataFrame(scaler.fit_transform(sampler_data), columns=sampler_data.columns)
    print(f"Scaled data shape: {sampler_data_scaled.shape}")
   
    # Get unique classes
    unique_classes = data['class'].unique()
    print(f"Unique classes: {unique_classes}")

    # Compute mean vectors and magnitudes for each class
    class_vectors = {}
    class_magnitudes = {}
    for cls in unique_classes:
        class_data = sampler_data_scaled[data['class'] == cls]
        vec = class_data.mean(axis=0).values
        mag = np.linalg.norm(vec)
        class_vectors[f'class_{cls}_vector'] = vec
        class_magnitudes[f'class_{cls}_mag'] = mag
      

    # Compute similarity of each instance with its own class
    sampler_data_scaled['0'] = np.nan
    for idx in sampler_data_scaled.index:
        row = sampler_data_scaled.loc[idx, vars].values
        current_class = data.loc[idx, 'class']
        vec = class_vectors[f'class_{current_class}_vector']
        mag = class_magnitudes[f'class_{current_class}_mag']
        row_norm = np.linalg.norm(row)
        if mag == 0 or row_norm == 0:
            similarity = 0
            print(f"Zero magnitude encountered at index {idx}, assigning similarity 0.")
        else:
            similarity = np.dot(vec, row) / (mag * row_norm)
        sampler_data_scaled.loc[idx, '0'] = similarity

    # Compute similarity between instances and all classes
    simi_df = pd.DataFrame(index=data.index)
    for cls in unique_classes:
        vec = class_vectors[f'class_{cls}_vector']
        mag = class_magnitudes[f'class_{cls}_mag']
        sim_column = []
        for idx in sampler_data_scaled.index:
            row = sampler_data_scaled.loc[idx, vars].values
            row_norm = np.linalg.norm(row)
            if mag == 0 or row_norm == 0:
                similarity = 0
            else:
                similarity = np.dot(vec, row) / (mag * row_norm)
            sim_column.append(similarity)
        simi_df[str(cls)] = sim_column
       
    # Prepare similarity table
    simi_table = pd.concat([sampler_data_scaled['0'], simi_df], axis=1)
    simi_table['class'] = data['class'].values
    simi_table['Name'] = data.iloc[:,0].values
    simi_table['flag'] = data['flag'].values
   

    # Plot similarity before truncation (if requested)
    if plot:
        x_column = str(compare_with )
        print(f"Plotting similarity for class {class_label} compared with class {compare_with }.")
        
        # Plot before truncation
        plot_data_before = simi_table.dropna(subset=[x_column, 'class'])
        plot_data_before = plot_data_before[~np.isinf(plot_data_before[x_column])]
        if not plot_data_before.empty:
            plt.figure(figsize=(10, 5))
            sns.scatterplot(
                x=plot_data_before[x_column],
                y=plot_data_before['class'],
                hue=plot_data_before['class'],
                
            )
            texts = []
            for i in range(plot_data_before.shape[0]):
                texts.append(
                    plt.text(
                        plot_data_before[x_column][i],
                        plot_data_before['class'][i],
                        plot_data_before['Name'][i],
                        fontsize=9
                    )
                )

            # Adjust text to avoid overlap
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

            plt.title('Similarity Before Group Truncation')
            plt.xlabel(f"Similarity with Class {compare_with }")
            plt.ylabel("Class")
            plt.legend()
            plt.show()

    # Sampling based on similarity
    simi_class = simi_table[simi_table['class'] == class_label][str(compare_with )]
    sample_size = sample_size if sample_size else len(simi_class)
    steps = np.linspace(simi_class.min(), simi_class.max(), sample_size)
    dis_mat = np.abs(np.subtract.outer(simi_class.to_numpy(), steps))

    keep_flags = []
    simi_class_indices = simi_class.index.tolist()
    if len(steps) < len(simi_class):
        for i in range(len(steps)):
            drop = np.argmin(dis_mat[:, i])
            idx = simi_class.index[drop]
            keep_flags.append(simi_table.loc[idx, 'flag'])
            dis_mat = np.delete(dis_mat, drop, axis=0)
            simi_class_indices.pop(drop)
            simi_class = simi_class.drop(idx)
            
    else:
        keep_flags = simi_table[simi_table['class'] == class_label]['flag'].tolist()

    # Plot similarity after truncation
    if plot:
        # Filter the table for the class we are plotting
        simi_table_after = simi_table[simi_table['flag'].isin(keep_flags)]
        
        # Separate the data: 
        # - Keep all data for classes other than the one we're truncating
        # - Keep only the truncated data for the specific class
        other_classes = simi_table[simi_table['class'] != class_label]  # All other classes, full data
        truncated_class = simi_table_after[simi_table_after['class'] == class_label]  # Only 'keep' samples of truncated class

        # Combine the data back together for plotting
        plot_data_after = pd.concat([other_classes, truncated_class])

        # Ensure no NaNs or infinite values in the selected columns
        plot_data_after = plot_data_after.dropna(subset=[x_column, 'class'])
        plot_data_after = plot_data_after[~np.isinf(plot_data_after[x_column])]
   
        if not plot_data_after.empty:
            plt.figure(figsize=(10, 5))
            sns.scatterplot(
                x=plot_data_after[x_column],
                y=plot_data_after['class'],
                hue=plot_data_after['class']
            )
            texts = []
            for cls in plot_data_after['class'].unique():
                class_data = plot_data_after[plot_data_after['class'] == cls]
                for i in range(class_data.shape[0]):
                    texts.append(
                        plt.text(
                            class_data[x_column].iloc[i],
                            class_data['class'].iloc[i],
                            class_data['Name'].iloc[i],
                            fontsize=9
                        )
                    )
            # Adjust text to avoid overlap
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
                
            plt.title('Similarity After Group Truncation')
            plt.xlabel(f"Similarity with Class {compare_with }")
            plt.ylabel("Class")
            # legend always on the right
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            plt.show()

    return keep_flags




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

def stratified_sampling(df, group, size):
    """
    Performs stratified sampling on a DataFrame based on the grouping variable.

    Parameters:
    - df (pd.DataFrame): The DataFrame to sample from.
    - group (str): The name of the column to group by for stratification.
    - size (int, float, dict): Desired sample size:
                               - If int, it specifies a fixed number of samples per group.
                               - If float < 1, it specifies the proportion to sample from each group.
                               - If dict, it specifies the exact number of samples for each group.

    Returns:
    - pd.DataFrame: A stratified sample of the original DataFrame.
    """
    
    # Group by the specified column
    grouped = df.groupby(group)

    # Initialize an empty DataFrame to store the stratified sample
    stratified_sample = pd.DataFrame()

    # Iterate through each group in the DataFrame
    for name, group_data in grouped:
        group_size = len(group_data)

        # Determine the number of samples based on the type of `size`
        if isinstance(size, int):
            n_samples = min(size, group_size)  # Take all if group size is smaller than `size`
        elif isinstance(size, float) and size < 1:
            n_samples = max(1, int(size * group_size))  # Take a proportion
        elif isinstance(size, dict):
            n_samples = min(size.get(name, 0), group_size)  # Take from dict, but ensure we don’t exceed group size
        else:
            raise ValueError("size must be an integer, a float < 1, or a dictionary")

        # Sample from the group
        sampled_data = group_data.sample(n=n_samples, replace=False)
        stratified_sample = pd.concat([stratified_sample, sampled_data])

    # Reset the index of the sampled DataFrame
    stratified_sample = stratified_sample.reset_index(drop=True)
    
    return stratified_sample


def plot_distribution_scatter(data, group, x_column='index', title='Class Distribution'):
    """
    Plots the class distribution using a scatter plot, with text annotations and adjusted labels.

    Parameters:
    - data (pd.DataFrame): The DataFrame whose distribution to plot.
    - group (str): The name of the column representing classes/groups.
    - x_column (str): Column to use for x-axis (default is 'index' for basic class scatter).
    - title (str): The title of the plot.
    """
    # Ensure the index column exists for x-axis if not provided
    if x_column == 'index':
        data = data.reset_index()

    # Prepare plot
    plt.figure(figsize=(10, 5))
    
    # Scatterplot with class hue
    sns.scatterplot(
        x=data[x_column],
        y=data[group],
        hue=data[group]
    )
    
    # Add text labels to each point, adjusting for overlaps
    texts = []
    for cls in data[group].unique():
        class_data = data[data[group] == cls]
    
        for i in range(class_data.shape[0]):
            texts.append(
                plt.text(
                    class_data[x_column].iloc[i],
                    class_data[group].iloc[i],
                    class_data.iloc[:,1].iloc[i],  # Use index or add a 'Name' column if needed
                    fontsize=9
                )
            )
    
    # Adjust text positions to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    
    # Set title, labels, and legend
    plt.title(title)
    plt.xlabel(f"{x_column}")
    plt.ylabel(f"{group}")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Legend on the right side
    
    # Show the plot
    plt.show()


def stratified_sampling_with_plots(df, size ,group='class' , plot=True):
    """
    Performs stratified sampling and plots the distribution before and after sampling.

    Parameters:
    - df (pd.DataFrame): The DataFrame to sample from.
    - group (str): The name of the column to group by for stratification.
    - size (int, float, dict): Desired sample size (same as stratified_sampling function).
    
    Returns:
    - pd.DataFrame: A stratified sample of the original DataFrame.
    """
    if plot:
    # Plot distribution before sampling
        plot_distribution_scatter(df, group, title="Class Distribution Before Stratified Sampling")

    # Perform stratified sampling
    stratified_sample = stratified_sampling(df, group, size)
    
    if plot:
    # Plot distribution after sampling
        plot_distribution_scatter(stratified_sample, group, title="Class Distribution After Stratified Sampling")
    
    return stratified_sample
