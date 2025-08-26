import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from adjustText import adjust_text
from itertools import combinations
import sqlite3
import os 
from typing import Iterable, List, Sequence, Set, Tuple, Union, Optional


def plot_similarity_scatter(
    simi_table: pd.DataFrame,
    x_column: str,
    title: str,
    annotate: bool = True,
) -> None:
    """
    Scatter plot of similarity vs. class. Uses seaborn if available; falls back to matplotlib.
    simi_table must contain columns: x_column, 'class', 'Label'.
    """
    # Clean data for plotting
    dfp = simi_table[[x_column, 'class', 'Label']].copy()
    dfp = dfp.dropna(subset=[x_column, 'class'])
    dfp = dfp[np.isfinite(dfp[x_column])]
    if dfp.empty:
        print("[plot] Nothing to plot.")
        return

    # Try seaborn, else use matplotlib only
    try:
        
        ax = sns.scatterplot(data=dfp, x=x_column, y='class', hue='class')
    except Exception:
        ax = plt.gca()
        for cls, sub in dfp.groupby('class'):
            ax.scatter(sub[x_column], sub['class'], label=str(cls), s=32, alpha=0.9)
        ax.legend()

    if annotate:
        texts = []
        for _, row in dfp.iterrows():
            texts.append(plt.text(row[x_column], row['class'], str(row['Label']), fontsize=9))
        # adjust text if adjustText is installed
        try:
            
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        except Exception:
            pass

    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel("class")
    # put legend to the right if seaborn created one
    leg = ax.get_legend()
    if leg is not None:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


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



def create_results_table_classification(db_path='results.db'):
    """Create the classification_results table if it does not already exist."""
    db_exists = os.path.isfile(db_path)

    if db_exists:
        print(f"Database already exists at: {db_path}")
    else:
        print(f"Database does not exist. It will be created at: {db_path}")

    # Create table
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS classification_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            combination TEXT,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            mcfadden_r2 REAL,
            avg_mcfadden_r2 REAL,
            avg_accuracy REAL,
            avg_f1_score REAL,
            threshold REAL
        );
    ''')
    print("Table 'classification_results' has been ensured to exist.")
    
    conn.commit()
    conn.close()

def insert_result_into_db_classification(db_path, combination, results, threshold, csv_path='classification_results.csv'):
    """
    Insert classification metrics into the SQLite database and append to a CSV file.

    Args:
        db_path (str): Path to SQLite database.
        combination (str): Feature combination used.
        results (dict): Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1_score', 'mcfadden_r2'.
        threshold (float): Threshold used.
        csv_path (str): Path to CSV file for backup/logging.
    """
    accuracy = results.get('accuracy')
    precision = results.get('precision')
    recall = results.get('recall')
    f1 = results.get('f1_score')
    mcfadden_r2 = results.get('mcfadden_r2')
    avg_mcfadden_r2 = results.get('avg_mcfadden_r2')
    avg_accuracy = results.get('avg_accuracy')
    avg_f1_score = results.get('avg_f1_score')
    # Insert into SQLite
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        INSERT INTO classification_results (
            combination, accuracy, precision, recall, f1_score, mcfadden_r2, threshold, avg_accuracy, avg_f1_score, avg_mcfadden_r2
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    ''', (str(combination), accuracy, precision, recall, f1, mcfadden_r2, threshold, avg_accuracy, avg_f1_score, avg_mcfadden_r2))
    conn.commit()
    conn.close()

    # Append to CSV
    result_dict = {
        'combination': [str(combination)],
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1],
        'mcfadden_r2': [mcfadden_r2],
        'avg_mcfadden_r2': [avg_mcfadden_r2],
        'avg_accuracy': [avg_accuracy],
        'avg_f1_score': [avg_f1_score],
        'threshold': [threshold]
    }

    result_df = pd.DataFrame(result_dict)
   
    if not os.path.isfile(csv_path):
        result_df.to_csv(csv_path, index=False, mode='w')
    else:
        result_df.to_csv(csv_path, index=False, mode='a', header=False)



def create_results_table(db_path='results.db'):
    """Create the regression_results table if it does not already exist."""
    # Check if DB file already exists
    db_exists = os.path.isfile(db_path)
    
    if db_exists:
        print(f"Database already exists at: {db_path}")
    else:
        print(f"Database does not exist. It will be created at: {db_path}")

    # Connect and create table if needed
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS regression_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            combination TEXT,
            r2 REAL,
            q2 REAL,
            mae REAL,
            rmsd REAL,

            threshold REAL,
            model TEXT,
            predictions TEXT
        );
    ''')
    print("Table 'regression_results' has been ensured to exist.")
    
    conn.commit()
    conn.close()


def insert_result_into_db_regression(db_path, combination, r2, q2, mae, rmsd, threshold, model, predictions, csv_path='results.csv'):
    """
    Insert one row of results into the SQLite database and append to a CSV file.
    
    Args:
        db_path (str): Path to the SQLite database.
        combination (str): Feature combination.
        formula (str): Model formula.
        r2 (float): R-squared value.
        q2 (float): Q-squared value.
        mae (float): Mean Absolute Error.
        rmsd (float): Root Mean Squared Deviation.
        threshold (float): Threshold used.
        csv_path (str): Path to the CSV file.
    """
    # Insert into SQLite database
    # print(f'Inserting results for combination: {combination} | R2: {r2} | Q2: {q2}')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        INSERT INTO regression_results (combination, r2, q2, mae, rmsd, threshold, model, predictions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    ''', (str(combination), r2, q2, mae, rmsd, threshold, str(model), str(predictions)))
    conn.commit()
    conn.close()

    # Prepare data for CSV
    result_dict = {
        'combination': [str(combination)],
        'r2': [r2],
        'q2': [q2],
        'mae': [mae],
        'rmsd': [rmsd],
        'threshold': [threshold],
        'model': [model],
        'predictions': [predictions]
    }
  
    result_df = pd.DataFrame(result_dict)
    
    # Check if CSV exists; if not, write header
    if not os.path.isfile(csv_path):
        result_df.to_csv(csv_path, index=False, mode='w')
     
    else:
        result_df.to_csv(csv_path, index=False, mode='a', header=False)
       


def load_results_from_db(db_path, table='regression_results'):
    """
    Load the entire results table from the SQLite database.

    Args:
        db_path (str): Path to the SQLite database.
        table   (str): Table name ('regression_results' or 'classification_results').

    Returns:
        List[dict]: One dict per row, with column names as keys.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
  
    conn.close()
    return df

def sample(x, size=None, replace=False, prob=None, random_state=None):
    """
    Draw random samples from a population.

    Parameters:
        x (int or sequence): 
            - If an integer, represents the range 1 to `x` inclusive.
            - If a sequence (list, tuple, numpy array), samples are drawn from the elements of the sequence.
        size (int, optional): 
            Number of samples to draw. 
            - If `None`, defaults to the length of `x` (only applicable when `replace=False`).
        replace (bool, optional): 
            Whether the sampling is with replacement. Defaults to `False`.
        prob (list or numpy.ndarray, optional): 
            A list of probabilities associated with each element in `x`. Must be the same length as `x` if provided.
        random_state (int, numpy.random.Generator, or None, optional): 
            Seed or random number generator for reproducibility.

    Returns:
        list: A list of sampled elements.
    """
    # Handle random state
    rng = None
    if isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    elif random_state is not None:
        raise ValueError("random_state must be an int, numpy.random.Generator, or None.")
    
    # If x is an integer, interpret as 1 to x inclusive
    if isinstance(x, int):
        if x < 1:
            raise ValueError("When 'x' is an integer, it must be greater than or equal to 1.")
        population = list(range(1, x + 1))
    elif isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        population = list(x)
    else:
        raise TypeError("x must be an integer or a sequence (list, tuple, numpy array, or pandas Series).")
    
    # Determine default size
    if size is None:
        if replace:
            size = len(population)
        else:
            size = len(population)
    
    # Validate size
    if not replace and size > len(population):
        raise ValueError("Cannot take a larger sample than population when 'replace' is False.")
    
    # Handle probabilities
    if prob is not None:
        if len(prob) != len(population):
            raise ValueError("'prob' must be the same length as 'x'.")
        prob = np.array(prob)
        if not np.isclose(prob.sum(), 1):
            raise ValueError("The sum of 'prob' must be 1.")
    
    # Perform sampling
    if rng is not None:
        sampled = rng.choice(population, size=size, replace=replace, p=prob)
    else:
        sampled = np.random.choice(population, size=size, replace=replace, p=prob)
    
    return sampled.tolist()

def assign_folds_no_empty(n_samples, n_folds, random_state=None):
    """
    Assign each data point to a fold ensuring that no fold is empty.

    Parameters:
        n_samples (int): Number of data points.
        n_folds (int): Number of folds.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        list: List of fold assignments (1-based indexing).
    """
    if n_folds > n_samples:
        raise ValueError("Number of folds cannot exceed number of samples.")

    # Assign one unique data point to each fold
    initial_assignments = sample(n_samples, size=n_folds, replace=False, random_state=random_state)
    # initial_assignments are unique data points indices (1-based)

    # Initialize all assignments to 0
    fold_assignments = [0] * n_samples

    # Assign each fold its unique data point
    for fold_num, idx in enumerate(initial_assignments, start=1):
        fold_assignments[idx - 1] = fold_num  # Convert to 1-based indexing

    # Assign the remaining data points to any fold using the sample function
    remaining_indices = [i for i in range(1, n_samples + 1) if fold_assignments[i - 1] == 0]
    if remaining_indices:
        # Define uniform probabilities for simplicity; modify if needed
        fold_probs = [1.0 / n_folds] * n_folds
        sampled_folds = sample(n_folds, size=len(remaining_indices), replace=True, prob=fold_probs, random_state=random_state)
        for idx, fold_num in zip(remaining_indices, sampled_folds):
            fold_assignments[idx - 1] = fold_num

    return fold_assignments


# --- your main function, refactored -----------------------------------------

def r_squared(pred, obs, formula="corr", na_rm=False):
    """
    Compute R-squared between observed and predicted values.

    Args:
        pred (array-like): Predicted values.
        obs (array-like): Observed (actual) values.
        formula (str): Method for R² calculation: "corr" (default) or "traditional".
        na_rm (bool): If True, remove NaN values before computation.

    Returns:
        float: R-squared value.
    """
    # Convert inputs to numpy arrays for easier computation
    pred = np.array(pred)
    obs = np.array(obs)
    
    # Handle missing values (NaN) if na_rm is True
    if na_rm:
        mask = ~np.isnan(pred) & ~np.isnan(obs)
        pred = pred[mask]
        obs = obs[mask]

    n = len(pred)

    if formula == "corr":
        # Correlation-based R²
        corr_matrix = np.corrcoef(obs, pred)
        r_squared_value = corr_matrix[0, 1] ** 2

    elif formula == "traditional":
        # Traditional R²: 1 - (SS_res / SS_tot)
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = (n - 1) * np.var(obs, ddof=1)  # Sample variance
        r_squared_value = 1 - (ss_res / ss_tot)

    else:
        raise ValueError("Invalid formula type. Choose 'corr' or 'traditional'.")

    return r_squared_value


def set_max_features_limit(total_features_num, max_features_num=None):
    if max_features_num is None:
        max_features_num = int(total_features_num / 5.0)
    return max_features_num

def get_feature_combinations(features, min_features_num=2, max_features_num=None):
    max_features_num = set_max_features_limit(len(features), max_features_num)
    total_combinations = 0  # Counter to track the total number of combinations generated
    for current_features_num in range(min_features_num, max_features_num + 1):
        count_combinations = 0  # Counter for each specific number of features
        for combo in combinations(features, current_features_num):
            count_combinations += 1
            total_combinations += 1
            yield combo

def _parse_tuple_string(s: str):

    return [x.strip(" '") for x in s.strip("()").split(",")]