# -*- coding: latin-1 -*-
import time
from tqdm import tqdm
from itertools import combinations
from joblib import Parallel, delayed
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error ,make_scorer, accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, cross_val_score, train_test_split, LeaveOneOut ,RepeatedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import cProfile
import pstats
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
import matplotlib.pyplot as plt
import PIL
from scipy.stats import t
import seaborn as sns
import sys
import os
from tkinter import filedialog, messagebox
from joblib import Parallel, delayed
import multiprocessing
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
import random
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from plot import *
    from modeling_utils import simi_sampler, stratified_sampling_with_plots
    from modeling_utils import *
except:
    from M3_modeler.plot import *
    from M3_modeler.modeling_utils import simi_sampler, stratified_sampling_with_plots
    from M3_modeler.modeling_utils import *




def create_results_table(db_path='results.db'):
    """Create the regression_results table if it does not already exist."""
    print('Creating table at location:', db_path)
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
            threshold REAL
        );
    ''')
    boolean_is_file = os.path.isfile(db_path)
    print('Table has been created successfully at location:', db_path, '\nCreated flag:', boolean_is_file)
    conn.commit()
    conn.close()

def insert_result_into_db(db_path, combination, r2, q2, mae,rmsd, threshold, csv_path='results.csv'):
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
        INSERT INTO regression_results (combination, r2, q2, mae, rmsd,threshold)
        VALUES (?, ?, ?, ?, ?, ?);
    ''', (str(combination), r2, q2, mae,rmsd, threshold))
    conn.commit()
    conn.close()

    # Prepare data for CSV
    result_dict = {
        'combination': [str(combination)],
        'r2': [r2],
        'q2': [q2],
        'mae': [mae],
        'rmsd': [rmsd],
        'threshold': [threshold]
    }

    result_df = pd.DataFrame(result_dict)

    # Check if CSV exists; if not, write header
    if not os.path.isfile(csv_path):
        result_df.to_csv(csv_path, index=False, mode='w')
        # print(f'CSV created and result saved at: {csv_path}')
    else:
        result_df.to_csv(csv_path, index=False, mode='a', header=False)
        # print(f'Result appended to existing CSV at: {csv_path}')

def load_results_from_db(db_path=None):
    """Load the entire results table from the SQLite database into a DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('SELECT * FROM regression_results', conn)
  
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
        
        # Print the count of combinations for each number of features
def fit_and_evaluate_single_combination_classification(model, combination, threshold=0.5, return_probabilities=False):
        selected_features = model.features_df[list(combination)]
        X = selected_features.to_numpy()
        y = model.target_vector.to_numpy()

        # Fit the model
        model.fit(X, y)

        # Evaluate the model
        evaluation_results = model.evaluate(X, y)
      
        # Check if accuracy is above the threshold
        if evaluation_results['mc_fadden_r2'] > threshold:
            avg_accuracy, avg_f1, avg_r2 = model.cross_validation(X, y , model.n_splits) ## , avg_auc
            evaluation_results['avg_accuracy'] = avg_accuracy
            evaluation_results['avg_f1_score'] = avg_f1
            evaluation_results['mc_fadden_r2'] = avg_r2
            # evaluation_results['avg_auc'] = avg_auc

        results={
            'combination': combination,
            'scores': evaluation_results,
            'models': model
        }

        if return_probabilities:
            if model.ordinal:
                cumulative_probs = model.result.predict(exog=X)
                # Calculate class probabilities from cumulative probabilities
                # For class i: P(Y = i) = P(Y <= i) - P(Y <= i - 1)
                class_probs = np.zeros_like(cumulative_probs)
                
                # First class probability is the cumulative probability of being in the first class
                class_probs[:, 0] = cumulative_probs[:, 0]
                
                # Intermediate class probabilities
                for i in range(1, cumulative_probs.shape[1]):
                    class_probs[:, i] = cumulative_probs[:, i] - cumulative_probs[:, i - 1]
                
                # Last class probability is 1 - cumulative probability of being in the previous class
                class_probs[:, -1] = 1 - cumulative_probs[:, -2]
                prob_df = pd.DataFrame(probabilities, columns=[f'Prob_Class_{i+1}' for i in range(probabilities.shape[1])])
                prob_df['Predicted_Class'] = model.model.predict(X)
                prob_df['True_Class'] = y
                return results, prob_df
            else:
                probabilities = model.model.predict_proba(X)
                # Creating a DataFrame for probabilities
                prob_df = pd.DataFrame(probabilities, columns=[f'Prob_Class_{i+1}' for i in range(probabilities.shape[1])])
                prob_df['Predicted_Class'] = model.model.predict(X)
                prob_df['True_Class'] = y

                return results, prob_df 
           
        return results


def fit_and_evaluate_single_combination_regression(model, combination, r2_threshold=0.85):
    selected_features = model.features_df[list(combination)]
    X = selected_features.to_numpy()
    y = model.target_vector.to_numpy()
   
    # Fit the model
    t0 = time.time()
    model.fit(X, y)
    fit_time=time.time()-t0
    # Evaluate the model
    t1=time.time()
   
    evaluation_results = model.evaluate(X, y)
    
    eval_time=time.time()-t1
    coefficients,intercepts = model.get_coefficients_from_trained_estimator()

    # Check if R-squared is above the threshold
    t3=time.time()
    if evaluation_results['r2'] > r2_threshold:
        q2, mae,rmsd = model.calculate_q2_and_mae(X, y, n_splits=1)
        evaluation_results['Q2'] = q2
        evaluation_results['MAE'] = mae
        evaluation_results['RMSD'] = rmsd

    q2_time=time.time()-t3
    # arrange the results based on highest q2
    # sorted_evaluation_results = sorted(evaluation_results, key=lambda x: x['Q2'], reverse=True)

    # Store results
    result = {
        'combination': combination,
        'scores': evaluation_results,
        'intercept': intercepts,
        'coefficients': coefficients,
        'models': model
    }
    if  'scores' in result:
        
        r2 = result['scores'].get('r2', float('-inf'))
        q2 = result['scores'].get('Q2', float('-inf'))
        mae = result['scores'].get('MAE', float('inf'))
        rmsd = result['scores'].get('RMSD', float('inf'))
        csv_path=model.db_path.replace('.db','.csv')
        # Insert into DB
        insert_result_into_db(
            db_path=model.db_path,
            combination=combination,
            r2=r2,
            q2=q2,
            mae=mae,
            rmsd=rmsd,
            threshold=r2_threshold,
            csv_path=csv_path
        )
    return result




class PlotModel:
    def __init__(self, model):
        self.model = model
    
    


class LinearRegressionModel:

    def __init__(
            self, 
            csv_filepaths, 
            process_method='one csv', 
            output_name='output', 
            leave_out=None, 
            min_features_num=2, 
            max_features_num=None, 
            n_splits=5, 
            metrics=None, 
            return_coefficients=False, 
            model_type='linear',    # <--- Choose 'linear' or 'lasso'
            alpha=1.0,
            app=None,
            db_path='results',               # <--- If lasso, this is the regularization strength
    ):
        self.csv_filepaths = csv_filepaths
        self.process_method = process_method
        self.output_name = output_name
        self.leave_out = leave_out
        self.min_features_num = min_features_num
        self.max_features_num = max_features_num
        self.metrics = metrics if metrics is not None else ['r2', 'neg_mean_absolute_error']
        self.return_coefficients = return_coefficients
        self.model_type = model_type
        self.alpha = alpha
        self.app = app
        self.n_splits = n_splits
        name=name = os.path.splitext(os.path.basename(self.csv_filepaths.get('features_csv_filepath')))[0]
       
        
        self.db_path = db_path + f'_{name}.db'
        create_results_table(self.db_path)
        
        if self.model_type.lower() == 'linear':
            self.model = LinearRegression()
            print('linear model selected')
        elif model_type.lower() == 'lasso':
            self.model = Lasso(alpha=alpha)
            print('lasso model selected')
        else:
            raise ValueError("Invalid model_type. Please specify 'linear' or 'lasso'.")

        if csv_filepaths:
            if process_method == 'one csv':
                self.process_features_csv(csv_filepaths.get('features_csv_filepath'), output_name=output_name)
            elif process_method == 'two csvs':
                self.process_features_csv(csv_filepaths.get('features_csv_filepath'))
                self.process_target_csv(csv_filepaths.get('target_csv_filepath'))

            self.compute_correlation()
            self.leave_out_samples(leave_out)
            self.determine_number_of_features()
            # self.get_feature_combinations()
            self.scaler = StandardScaler()
            
            self.features_df = pd.DataFrame(self.scaler.fit_transform(self.features_df), columns=self.features_df.columns)
            self.feature_names = self.features_df.columns.tolist()

    def compute_multicollinearity(self, vif_threshold=5.0):
        """
        Compute the Variance Inflation Factor (VIF) for each feature in the dataset.
        """
        # Compute VIF
        vif_results = self._compute_vif(self.features_df)
        
        # Identify features with high VIF
        high_vif_features = vif_results[vif_results['VIF'] > vif_threshold]
        
        return vif_results
    

    def process_features_csv(self, csv_filepath, output_name):

        df = pd.read_csv(csv_filepath)
        ## change the name of the first column to 0
        first_colindex = df.columns[0]
        df.rename(columns={first_colindex: '0'}, inplace=True)
        self.molecule_names = df['0'].tolist()
        self.features_df = df.drop(columns=['0'])
        self.target_vector = df[output_name]
        self.features_df= self.features_df.drop(columns=[output_name])
        self.features_list = self.features_df.columns.tolist()
       
    def compute_correlation(self,correlation_threshold=0.8, vif_threshold=5.0):
        app=self.app
        
        self.corr_matrix = self.features_df.corr()

        # Identify highly-correlated features above correlation_threshold
        high_corr_features = self._get_highly_correlated_features(
            self.corr_matrix, threshold=correlation_threshold
        )

    


        if high_corr_features and app is not None:
            # Show correlation report
            app.show_result(f"\n--- Correlation Report ---\n")
            app.show_result(
                f"Features with correlation above {correlation_threshold}:\n"
                f"{list(high_corr_features)}\n"
            )
            
            visualize_corr = messagebox.askyesno(
            title="Visualize Correlated Features?",
            message=(
                "Would you like to see a heatmap of the correlation among these features?"
            )
        )
            if visualize_corr:
                # Subset the correlation matrix for the correlated features only
                sub_corr = self.corr_matrix.loc[list(high_corr_features), list(high_corr_features)]
                
                # Create a heatmap with Seaborn
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(sub_corr, annot=False, cmap='coolwarm', square=True, ax=ax)
                ax.set_title(f"Correlation Heatmap (>{correlation_threshold})")
                plt.tight_layout()
                plt.show()

            # Ask user if they want to drop them (yes/no)
            drop_corr = messagebox.askyesno(
                title="Drop Correlated Features?",
                message=(
                    f"Features above correlation {correlation_threshold}:\n"
                    f"{list(high_corr_features)}\n\n"
                    "Do you want to randomly drop some of these correlated features?"
                )
            )
            if drop_corr:
                # Decide how many to drop. (Here: half the set, randomly)
                count_to_drop = len(high_corr_features) // 2
                features_to_drop = random.sample(list(high_corr_features), k=count_to_drop)

                app.show_result(f"\nRandomly selected {count_to_drop} features to drop:")
                app.show_result(f"{features_to_drop}\n")

                # Remove from DataFrame
                self.features_df.drop(columns=features_to_drop, inplace=True)
                self.features_list = self.features_df.columns.tolist()

                app.show_result(f"Remaining features: {self.features_list}\n")
            else:
                app.show_result("\nCorrelated features were not dropped.\n")
        else:
            if app is not None:
                app.show_result("\nNo features exceeded the correlation threshold.\n")


    def _compute_vif(self, df):
        """
        Compute the Variance Inflation Factor for each column in df.
        """
        # Check for non-numeric data
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df_scaled = df.select_dtypes(include=[np.number])

        # Handle missing or infinite values
        df_scaled = df_scaled.replace([np.inf, -np.inf], np.nan).dropna()

        # Check matrix rank
        rank = np.linalg.matrix_rank(df_scaled.values)
        if rank < df_scaled.shape[1]:
            print(f"Warning: Linear dependence detected. Matrix rank: {rank}")

        # Standardize data
        

        # Check correlation matrix
        correlation_matrix = df_scaled.corr()
        

        # Compute VIF
        vif = pd.DataFrame()
        vif["variables"] = df_scaled.columns
        vif["VIF"] = [
            variance_inflation_factor(df_scaled.values, i)
            for i in range(df_scaled.shape[1])
        ]
        
     
        return vif

    
    def _get_highly_correlated_features(self, corr_matrix, threshold=0.8):
        """
        Identify any features whose pairwise correlation is above the threshold.
        Returns a set of the implicated feature names.
        """
        corr_matrix_abs = corr_matrix.abs()
        columns = corr_matrix_abs.columns
        high_corr_features = set()

        # We only need to look at upper triangular part to avoid duplication
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                if corr_matrix_abs.iloc[i, j] > threshold:
                    # Add both columns in the pair
                    high_corr_features.add(columns[i])
                    high_corr_features.add(columns[j])
        
        return high_corr_features


    def process_target_csv(self, csv_filepath):
        target_vector_unordered = pd.read_csv(csv_filepath)[self.output_name]
        self.target_vector = target_vector_unordered.loc[self.molecule_names]

    def leave_out_samples(self, leave_out=None):
        
        if leave_out:
            if len(leave_out) == 1:
                leave_out = int(leave_out[0])
                self.molecule_names_predict = self.molecule_names[leave_out] if leave_out else None
                self.molecule_names = [name for i, name in enumerate(self.molecule_names) if i != leave_out] if leave_out else self.molecule_names
            else:
                self.molecule_names_predict = [self.molecule_names[i] for i in leave_out] if leave_out else None
                self.molecule_names = [name for i, name in enumerate(self.molecule_names) if i not in leave_out] if leave_out else self.molecule_names

            self.predict_features_df = self.features_df.loc[leave_out] if leave_out else None
            self.predict_target_vector = self.target_vector.loc[leave_out] if leave_out else None

            

        # add debugging print
        print(f'leave_out: {leave_out}')
        self.leftout_features= self.features_df.loc[leave_out] if leave_out else None
        self.leftout_target_vector = self.target_vector.loc[leave_out] if leave_out else None
        self.features_df = self.features_df.drop(index=leave_out) if leave_out else self.features_df
        self.target_vector = self.target_vector.drop(index=leave_out) if leave_out else self.target_vector

        

    def determine_number_of_features(self):
        total_features_num = self.features_df.shape[0]
        self.max_features_num = set_max_features_limit(total_features_num, self.max_features_num)
  


    def get_feature_combinations(self):
       
        self.features_combinations = list(get_feature_combinations(self.features_list, self.min_features_num, self.max_features_num))
   

    # def calculate_q2_and_mae(self, X, y, n_splits=None):
    #     """
    #     Calculate Q², MAE, and RMSD using fold-by-fold cross-validation.

    #     Args:
    #         X (np.ndarray): Feature matrix of shape (n_samples, n_features).
    #         y (np.ndarray): Target vector of shape (n_samples,).
    #         n_splits (int): Number of splits (folds) for cross-validation.
    #                     If None, defaults to self.n_splits.

    #     Returns:
    #         tuple (q2, mae, rmsd):
    #             q2  -> The average R² (i.e., Q²) across the folds
    #             mae -> The average Mean Absolute Error across the folds
    #             rmsd-> The average Root Mean Squared Deviation across the folds
    #     """
    #     if n_splits is None:
    #         n_splits = self.n_splits

    #     n_samples = X.shape[0]
    #     indices = np.arange(n_samples)
    #     np.random.shuffle(indices)

    #     fold_size = n_samples // n_splits

    #     # Lists to store metrics for each fold
    #     fold_r2_scores = []
    #     fold_mae_scores = []
    #     fold_rmsd_scores = []

    #     for i in range(n_splits):
    #         # Determine start/end of this fold
    #         start = i * fold_size
    #         end = start + fold_size if i != n_splits - 1 else n_samples
            
    #         test_indices = indices[start:end]
    #         train_indices = np.concatenate([indices[:start], indices[end:]])

    #         # Split into training and test
    #         X_train, y_train = X[train_indices], y[train_indices]
    #         X_test, y_test = X[test_indices], y[test_indices]

    #         # Fit on training
    #         self.fit(X_train, y_train)

    #         # Predict on test
    #         y_pred_fold = self.predict(X_test)

    #         # Compute fold metrics
    #         r2_fold = r2_score(y_test, y_pred_fold)
    #         mae_fold = mean_absolute_error(y_test, y_pred_fold)
    #         rmsd_fold = np.sqrt(mean_squared_error(y_test, y_pred_fold))

    #         fold_r2_scores.append(r2_fold)
    #         fold_mae_scores.append(mae_fold)
    #         fold_rmsd_scores.append(rmsd_fold)

    #     # Average metrics across all folds
    #     q2 = np.mean(fold_r2_scores)
    #     mae = np.mean(fold_mae_scores)
    #     rmsd = np.mean(fold_rmsd_scores)

    #     return q2, mae, rmsd

    def calculate_q2_and_mae(self, X, y, n_splits=None, test_size=0.1, random_state=84, n_iterations=100):
        """
        Calculate Q², MAE, and RMSD using scikit-learn's cross-validation or single train/test split.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
            n_splits (int, optional): Number of splits (folds) for cross-validation.
                                    If None, defaults to self.n_splits.
                                    If 1, performs a single train/test evaluation on the entire dataset.

        Returns:
            tuple: (q2, mae, rmsd)
                q2   -> The average R² (i.e., Q²) across the folds or single evaluation.
                mae  -> The average Mean Absolute Error across the folds or single evaluation.
                rmsd -> The average Root Mean Squared Deviation across the folds or single evaluation.
        """
        if n_splits is None:
            n_splits = self.n_splits

        if n_splits < 1:
            raise ValueError("n_splits must be at least 1.")

       ## verify the X is normalized and normalize if not, check the variance of the X
        if np.var(X) > 1:
            X = StandardScaler().fit_transform(X)


        if n_splits == 1:
            loo = LeaveOneOut()
                
            # Initialize an array to store predictions
            y_pred = np.empty_like(y, dtype=float)
            estimator = self.model
            # Iterate through each split
            for fold, (train_index, test_index) in enumerate(loo.split(X), 1):
                # Split the data into training and testing sets
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # Fit the model on the training data
                estimator.fit(X_train, y_train)     
                # Predict the target for the test data
                y_pred[test_index] = estimator.predict(X_test)
                
            # Compute evaluation metrics on the aggregated predictions
            q2 = r_squared(y, y_pred, formula="corr")
            mae = mean_absolute_error(y, y_pred)
            rmsd = np.sqrt(mean_squared_error(y, y_pred))

            return q2, mae, rmsd
        
        else:
            # Define a custom scorer for RMSD
            def rmsd_scorer(y_true, y_pred):
                return np.sqrt(mean_squared_error(y_true, y_pred))
            # Create a scorer object for RMSD
            rmsd_score = make_scorer(rmsd_scorer, greater_is_better=False)  # Lower RMSD is better
            r2_score = make_scorer(r_squared, greater_is_better=True)  # Higher R² is better
            # Define the cross-validation strategy
            cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_iterations, random_state=random_state)
            # Define the scoring metrics
            scoring = {
                'r2': r2_score,
                'mae': 'neg_mean_absolute_error',  # scikit-learn uses negative MAE for maximization
                'rmsd': rmsd_score
            }
            q2_list = []
            mae_list = []
            rmsd_list = []

            n_samples = len(y)

            for iteration in range(n_iterations):
                # Assign each data point to a fold ensuring no fold is empty
                random_seed = random_state + iteration
                fold_assignments = assign_folds_no_empty(n_samples, n_splits, random_seed)
         
                # print(f"Fold assignments in iteration {iteration + 1}: {fold_assignments}")
                # Initialize arrays to store predictions
                predictions = np.empty(n_samples)
                predictions[:] = np.nan  # Initialize with NaN

                for fold in range(1, n_splits + 1):
                    # Define training and testing indices
                    test_indices = [i for i, x in enumerate(fold_assignments) if x == fold]
                    train_indices = [i for i, x in enumerate(fold_assignments) if x != fold]

                    # Handle potential empty folds (shouldn't occur)
                    if len(train_indices) == 0 or len(test_indices) == 0:
                        print(f"Warning: Fold {fold} has no training or testing samples in iteration {iteration + 1}.")
                        continue

                    # Define training and testing sets
                    X_train, X_test = X[train_indices], X[test_indices]
                    y_train, y_test = y[train_indices], y[test_indices]

                    # Train the model
                    self.model.fit(X_train, y_train)

                    # Predict on the test set
                    y_pred = self.model.predict(X_test)

                    # Store predictions
                    predictions[test_indices] = y_pred

                # After all folds, compute metrics
                # Ensure no NaN predictions
                valid = ~np.isnan(predictions)
                if not np.all(valid):
                    print(f"Warning: Some samples were not assigned to any fold in iteration {iteration + 1}.")

                # Calculate metrics
                mae = mean_absolute_error(y[valid], predictions[valid])
                q2 = r_squared(y[valid], predictions[valid])
                rmsd = np.sqrt(mean_squared_error(y[valid], predictions[valid]))
               
                # Append to lists
                mae_list.append(mae)
                q2_list.append(q2)
                rmsd_list.append(rmsd)

            # Compute the overall average metrics across all iterations
            average_q2 = np.mean(q2_list)
            average_mae = np.mean(mae_list)
            average_rmsd = np.mean(rmsd_list)

            return average_q2, average_mae, average_rmsd



    def fit(self, X, y, alpha=1e-5):
        """
        Train the linear regression model using the normal equation with regularization.

        Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        alpha (float): Regularization parameter. Default is 1e-5.

        Returns:
        self: Fitted model.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add a bias term
        A = X_b.T.dot(X_b)
        I = np.eye(A.shape[0])
        self.theta = np.linalg.inv(A + alpha * I).dot(X_b.T).dot(y)
        return self
    
    def predict(self, X, calc_covariance_matrix=False,confidence_level=0.95):
        """
        Make predictions using the trained linear regression model.
        
        Optionally, calculate and store the covariance matrix of the model's coefficients.

        Args:
        X (np.ndarray): Feature matrix.
        calc_covariance_matrix (bool): Whether to calculate and store the covariance matrix. Default is False.

        Returns:
        np.ndarray: Predicted values.
        """
        # Add bias (intercept) term
        
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of ones for the intercept
        

        predictions = X_b.dot(self.theta)
        
        # Optionally calculate the covariance matrix
        if calc_covariance_matrix:
            # Assuming residuals have already been calculated in the model fitting
            # If not, you'll need to pass the true y values to calculate residuals
            # Calculate residual variance (sigma^2)
            residuals = self.target_vector - X_b.dot(self.theta)  # Assuming target_vector and features_df are stored
            self.residual_variance = np.sum(residuals ** 2) / (X_b.shape[0] - X_b.shape[1])
            # Calculate and store the covariance matrix of the coefficients
            self.model_covariance_matrix = self.residual_variance * np.linalg.inv(X_b.T @ X_b)
            t_value = t.ppf(1 - (1 - confidence_level) / 2, df=self.features_df.shape[0] - self.features_df.shape[1])
            variance_terms = np.array([
                np.sqrt(self.residual_variance * (1 + X_b[i, :].T @ np.linalg.inv(X_b.T @ X_b) @ X_b[i, :]))
                for i in range(X_b.shape[0])
            ])
            
            # Upper and lower bounds of the prediction intervals
            lower_bounds = predictions - t_value * variance_terms
            upper_bounds = predictions + t_value * variance_terms

            return predictions, lower_bounds, upper_bounds

        return predictions
    
    def predict_for_leftout(self, X, calc_covariance_matrix=False, confidence_level=0.95):
        """
        Make predictions using the trained linear regression model.

        Optionally, calculate and store the covariance matrix of the model's coefficients.

        Args:
            X (np.ndarray or pd.DataFrame): Feature matrix.
            calc_covariance_matrix (bool): Whether to calculate and store the covariance matrix.
            confidence_level (float): Confidence level for prediction intervals.

        Returns:
            np.ndarray: Predicted values, or a tuple (predictions, lower_bounds, upper_bounds)
                        if calc_covariance_matrix is True.
        """
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm
        from scipy.stats import t

        # Ensure X has the same features as the training data
        if isinstance(X, pd.DataFrame):
            if hasattr(self, 'feature_names'):
                try:
                    X = X[self.feature_names].values
                except KeyError as e:
                    raise KeyError(f"Missing training features in input X: {e}")
            else:
                X = X.values  # fallback if feature names aren't stored

        # Ensure X is at least 2D
        X = np.atleast_2d(X)

        # Check dimensionality match (excluding intercept)
        expected_features = self.theta.shape[0] - 1
        if X.shape[1] != expected_features:
            raise ValueError(
                f"Expected input with {expected_features} features, but got {X.shape[1]}."
            )

        # Add bias (intercept) column
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Make predictions
        predictions = X_b.dot(self.theta)

        # If not calculating intervals, return plain predictions
        if not calc_covariance_matrix:
            return predictions

        # Compute covariance matrix and prediction intervals
        if hasattr(self, 'feature_names'):
            X_train = self.features_df[self.feature_names].values
        else:
            X_train = self.features_df
        X_train_b = sm.add_constant(X_train)

        # Residuals and variance from training
        residuals = self.target_vector - X_train_b.dot(self.theta)
        n_train, p = X_train_b.shape
        self.residual_variance = np.sum(residuals ** 2) / (n_train - p)

        # Covariance matrix
        xtx_inv = np.linalg.inv(X_train_b.T.dot(X_train_b))
        self.model_covariance_matrix = self.residual_variance * xtx_inv

        # t-value for the confidence interval
        t_value = t.ppf(1 - (1 - confidence_level) / 2, df=n_train - p)

        # Prediction variance terms
        variance_terms = np.array([
            np.sqrt(self.residual_variance * (1 + x_i.T @ xtx_inv @ x_i))
            for x_i in X_b
        ])

        # Prediction bounds
        lower_bounds = predictions - t_value * variance_terms
        upper_bounds = predictions + t_value * variance_terms

        return predictions, lower_bounds, upper_bounds

    def evaluate(self, X, y):
        
        y_pred = self.predict(X)
        results = {}
       
        if 'r2' in self.metrics:
            results['r2'] = r2_score(y, y_pred)
        # if 'neg_mean_absolute_error' in self.metrics:
        #     results['neg_mean_absolute_error'] = -mean_absolute_error(y, y_pred)

        return results

    def cross_validate(self, X, y):
        n_splits=self.n_splits 
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred = cross_val_predict(self.model, X, y, cv=cv)
        scores = cross_validate(self.model, X, y, cv=cv, scoring=self.metrics)
        return y_pred, scores
    
    def get_coefficients_from_trained_estimator(self):
        intercept = self.theta[0]
        coefficients = self.theta[1:]  
        return intercept, coefficients
    
    def get_covariace_matrix(self,features):
        intercept = self.theta[0]
        coefficients = self.theta[1:]

        # Assume self.model_covariance_matrix is calculated during training and is available
        # This would be the covariance matrix of the parameter estimates
        cov_matrix = self.model_covariance_matrix
        
        # Calculate standard errors from the diagonal of the covariance matrix
        std_errors = np.sqrt(np.diag(cov_matrix))
        
        # Calculate t-values
        t_values = coefficients / std_errors[1:]  # Ignoring intercept for t-value
        
        # Degrees of freedom (n - p - 1), where n is the number of observations, p is the number of predictors
        degrees_of_freedom = len(self.target_vector) - len(coefficients) - 1
        
        # Calculate p-values using the t-distribution
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df=degrees_of_freedom))
        
        # Include intercept values for std_error, t_value, and p_value
        intercept_std_error = std_errors[0]
        intercept_t_value = intercept / intercept_std_error
        intercept_p_value = 2 * (1 - stats.t.cdf(np.abs(intercept_t_value), df=degrees_of_freedom))
        
        # Combine intercept and coefficients into a table
        coefficient_table = {
            'Estimate': np.concatenate([[intercept], coefficients]),
            'Std. Error': std_errors,
            't value': np.concatenate([[intercept_t_value], t_values]),
            'p value': np.concatenate([[intercept_p_value], p_values])
        }

        # Convert to DataFrame
        coef_df = pd.DataFrame(coefficient_table)
        coef_df.index = ['(Intercept)'] + features  # Assuming feature_names is a list of your feature names

        
        
        return coef_df



    def fit_and_evaluate_combinations(self, top_n=50, n_jobs=-1, initial_r2_threshold=0.85 ,bool_parallel=False): ## parallel is not working
        app=self.app
        self.get_feature_combinations()
        if multiprocessing.cpu_count() == 1 or bool_parallel==False:
            n_jobs = 1
        
        print(f"Using {n_jobs} jobs for evaluation. found {multiprocessing.cpu_count()} cores")

        def is_all_inf(results):
            return all(x['scores'].get('Q2', float('-inf')) == float('-inf') for x in results)

        def evaluate_with_threshold(threshold):
            """
            Perform evaluation with the specified R2 threshold. 
            Runs either in parallel or single thread depending on the final value of n_jobs.
            """
            try:
                results=load_results_from_db(self.db_path)
            except:
                results=[]
                
            if n_jobs == 1:
                # Non-parallel execution
                results = []
                for combination in tqdm(self.features_combinations,
                                        desc=f'Calculating combos with threshold {threshold} (single-core)'):
                    
                    res = fit_and_evaluate_single_combination_regression(self,combination, threshold) # fit_and_evaluate_single_combination_with_prints
                    results.append(res)
            else:
                # Parallel execution
                Parallel(n_jobs=n_jobs)(
                    delayed(fit_and_evaluate_single_combination_regression)(self, combination, threshold) # fit_and_evaluate_single_combination_with_prints
                    for combination in tqdm(self.features_combinations, 
                                            desc=f'Calculating combos with threshold {threshold} (parallel)')
                )
                
        
                results = load_results_from_db(self.db_path)
            
           
            # Filter out None entries if any exist
            return results

        def get_highest_r2(results):
            # Extract the highest R2 value from the results that are not -inf
            r2_values = [x['scores'].get('r2') for x in results ]
            if r2_values:
                
                return max(r2_values)
                
            return None

        # Initial evaluation with the initial R2 threshold
        sorted_results = evaluate_with_threshold(initial_r2_threshold)
        sorted_results = sorted(sorted_results, key=lambda x: x['scores'].get('Q2', float('-inf')), reverse=True)
        

        # Check if all Q2 values are -inf and recalibrate if necessary
        if is_all_inf(sorted_results):
            print("All Q2 values are -inf, recalculating with a new R2 threshold...")
            highest_r2 = get_highest_r2(sorted_results)
            if highest_r2 is not None:
                new_threshold = highest_r2 - 0.15  # Reduce the highest found R2 by 0.15
            else:
                new_threshold = initial_r2_threshold - 0.15  # Default lowering if no R2 found
            print('new threshold',new_threshold)
            sorted_results = evaluate_with_threshold(new_threshold)
            sorted_results = sorted(sorted_results, key=lambda x: x['scores'].get('Q2', float('-inf')), reverse=True)
            sorted_results = sorted_results[:top_n]

        sorted_results = sorted_results[:top_n]
        # Print models regression table if sorted results are valid
        if sorted_results:
            print_models_regression_table(sorted_results,app)
        

        if self.leave_out:
            X = self.predict_features_df.to_numpy()
            y = self.predict_target_vector.to_numpy()
            fit=self.fit(X,y)
            predictions = self.predict(X)
            result_dict={'sample_name':self.molecule_names_predict,'predictions':predictions,'true':y}
            if app:
                app.show_result('\n\n Predictions on left out samples\n\n')
                app.show_result(pd.DataFrame(result_dict).to_markdown(tablefmt="pipe", index=False))
            else:
                print(pd.DataFrame(result_dict).to_markdown(tablefmt="pipe", index=False))

        return sorted_results
    
    def fit_and_evaluate_single_combination_with_prints(self, combination, r2_threshold):
        print(f"Starting evaluation for combination: {combination}")
        start_time = time.time()
        
        # Call the original evaluation method
        result = fit_and_evaluate_single_combination_regression(self, combination, r2_threshold=r2_threshold)
        
        end_time = time.time()
        print(f"Finished evaluation for combination: {combination} in {end_time - start_time:.2f} seconds")
        return result

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from statsmodels.miscmodels.ordinal_model import OrderedModel

class OrdinalLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, distr='logit'):
        """
        Custom wrapper for OrderedModel from statsmodels.
        
        Parameters:
        ------------
        distr: str, optional (default='logit')
            The distribution function for the ordinal logistic regression.
            Available options: 'logit', 'probit', 'loglog', 'cloglog', 'cauchit'.
        """
        self.distr = distr
        self.model = None
        self.result = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model = OrderedModel(y, X, distr=self.distr)
        self.result = self.model.fit(method='bfgs', maxiter=1000, disp=False)
        return self

    def predict(self, X):
        # Get predicted probabilities for each class
        class_probs = self.result.predict(exog=X, which='prob')
        # Get the index of the class with the highest probability
        y_pred_indices = class_probs.argmax(axis=1)
        # Map indices to class labels
        y_pred = self.classes_[y_pred_indices]
        return y_pred



    def predict_proba(self, X):
        # Get predicted probabilities for each class
        class_probs = self.result.predict(exog=X, which='prob')
      
        return class_probs


    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Returns:
        ------------
        params: dict
            Dictionary of parameters.
        """
        return {"distr": self.distr}

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters:
        ------------
        params: dict
            Dictionary of parameters to set.
            
        Returns:
        ------------
        self: object
            Returns self.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ClassificationModel:
    def __init__(self, csv_filepaths, process_method='one csv', output_name='class', leave_out=None, min_features_num=2, max_features_num=None,n_splits=5, metrics=None, return_coefficients=False,ordinal=False, exclude_columns=None,app=None):
        self.csv_filepaths = csv_filepaths
        self.process_method = process_method
        self.output_name = output_name
        self.leave_out = leave_out
        self.min_features_num = min_features_num
        self.max_features_num = max_features_num
        self.metrics = metrics if metrics is not None else ['accuracy','precision','recall' ,'f1', 'roc_auc','mc_fadden_r2']
        self.return_coefficients = return_coefficients
        self.n_splits = n_splits
        self.ordinal=ordinal
        self.app=app

        if csv_filepaths:
      
            if process_method == 'one csv':
                self.process_features_csv(csv_filepaths.get('features_csv_filepath'),  output_name=output_name)
            elif process_method == 'two csvs':
                self.process_features_csv(csv_filepaths.get('features_csv_filepath'))
                self.process_target_csv(csv_filepaths.get('target_csv_filepath'))
            self.compute_correlation()
            self.leave_out_samples(leave_out)
            self.determine_number_of_features()
            self.get_feature_combinations()
            self.scaler = StandardScaler()
            if exclude_columns is None:
                exclude_columns = []

            # Identify columns to scale
            columns_to_scale = [col for col in self.features_df.columns if col not in exclude_columns]

            # Apply scaling only to selected columns
            self.scaler = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), columns_to_scale)  # Scale only these columns
                ],
                remainder='passthrough'  # Keep excluded columns as they are
            )

            # Fit and transform the data
            self.features_df = pd.DataFrame(self.scaler.fit_transform(self.features_df), columns=self.features_df.columns)
            

        if ordinal:
            self.model = OrdinalLogisticRegression()
        else:    
            self.model = LogisticRegression(solver='lbfgs', random_state=42)

    def simi_sampler_(self,class_label, compare_with=0, plot=False, sample_size=None):
        data=pd.concat([self.features_df,self.target_vector],axis=1)
        ## if self.indices exist , append to it else create it
        if hasattr(self, 'indices'):
            self.indices = self.indices + simi_sampler(data, class_label, compare_with, plot, sample_size)
        else:
            self.indices = simi_sampler(data, class_label, compare_with, plot, sample_size)
        
    def apply_simi_sampler(self):
        self.features_df = self.features_df.loc[self.indices]
        self.target_vector = self.target_vector.loc[self.indices]
        
    def stratified_sampling_(self,plot=False, sample_size=None):
        data=pd.concat([self.features_df,self.target_vector],axis=1)
        df = stratified_sampling_with_plots(data, plot, sample_size)
        self.features_df = df.drop(columns=[self.output_name])
        self.target_vector = df[self.output_name]

    def get_feature_combinations(self):
        self.features_combinations = list(get_feature_combinations(self.features_list, self.min_features_num, self.max_features_num))


    def determine_number_of_features(self):
        total_features_num = self.features_df.shape[0]
        self.max_features_num = set_max_features_limit(total_features_num, self.max_features_num)
    

    def leave_out_samples(self, leave_out=None):
        self.predict_features_df = self.features_df.loc[leave_out] if leave_out else self.features_df
        self.predict_target_vector = self.target_vector.loc[leave_out] if leave_out else self.target_vector
        self.features_df = self.features_df.drop(index=leave_out) if leave_out else self.features_df
        self.target_vector = self.target_vector.drop(index=leave_out) if leave_out else self.target_vector

        
    def process_features_csv(self, csv_filepath, output_name):
        df = pd.read_csv(csv_filepath)
        self.molecule_names = df.iloc[:, 0].tolist()
        self.features_df = df.drop(columns=[df.columns[0]])
        self.target_vector = df[output_name]
        self.features_df= self.features_df.drop(columns=[output_name])
       
        self.features_list = self.features_df.columns.tolist()
    
    def compute_multicollinearity(self,correlation_threshold=0.8):
        app=self.app
        self.corr_matrix = self.features_df.corr()

        # Identify highly-correlated features above correlation_threshold
        high_corr_features = self._get_highly_correlated_features(
            self.corr_matrix, threshold=correlation_threshold
        )


        if high_corr_features:
            # Show correlation report
            app.show_result(f"\n--- Correlation Report ---\n")
            app.show_result(
                f"Features with correlation above {correlation_threshold}:\n"
                f"{list(high_corr_features)}\n"
            )
            visualize_corr = messagebox.askyesno(
            title="Visualize Correlated Features?",
            message=(
                "Would you like to see a heatmap of the correlation among these features?"
            )
        )
            if visualize_corr:
                # Subset the correlation matrix for the correlated features only
                sub_corr = self.corr_matrix.loc[list(high_corr_features), list(high_corr_features)]
                
                # Create a heatmap with Seaborn
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(sub_corr, annot=False, cmap='coolwarm', square=True, ax=ax)
                ax.set_title(f"Correlation Heatmap (>{correlation_threshold})")
                plt.tight_layout()
                plt.show()

            
            # Ask user if they want to drop them (yes/no)
            drop_corr = messagebox.askyesno(
                title="Drop Correlated Features?",
                message=(
                    f"Features above correlation {correlation_threshold}:\n"
                    f"{list(high_corr_features)}\n\n"
                    "Do you want to randomly drop some of these correlated features?"
                )
            )
            if drop_corr:
                # Decide how many to drop. (Here: half the set, randomly)
                count_to_drop = len(high_corr_features) // 2
                features_to_drop = random.sample(list(high_corr_features), k=count_to_drop)

                app.show_result(f"\nRandomly selected {count_to_drop} features to drop:")
                app.show_result(f"{features_to_drop}\n")

                # Remove from DataFrame
                self.features_df.drop(columns=features_to_drop, inplace=True)
                self.features_list = self.features_df.columns.tolist()

                app.show_result(f"Remaining features: {self.features_list}\n")
            else:
                app.show_result("\nCorrelated features were not dropped.\n")
        else:
            app.show_result("\nNo features exceeded the correlation threshold.\n")

       


    def _compute_vif(self, df):
        """
        Compute the Variance Inflation Factor for each column in df.
        """
        # Check for non-numeric data
        df = df.select_dtypes(include=[np.number])

        # Handle missing or infinite values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # Check matrix rank
        rank = np.linalg.matrix_rank(df.values)
        if rank < df.shape[1]:
            print(f"Warning: Linear dependence detected. Matrix rank: {rank}")

        # Standardize data
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        # Check correlation matrix
        correlation_matrix = df_scaled.corr()
        print('Checking correlation matrix:\n', correlation_matrix)

        # Compute VIF
        vif = pd.DataFrame()
        vif["variables"] = df_scaled.columns
        vif["VIF"] = [
            variance_inflation_factor(df_scaled.values, i)
            for i in range(df_scaled.shape[1])
        ]
        
        print('Checking VIF values:\n', vif)
        return vif

    def _get_highly_correlated_features(self, corr_matrix, threshold=0.8):
        """
        Identify any features whose pairwise correlation is above the threshold.
        Returns a set of the implicated feature names.
        """
        corr_matrix_abs = corr_matrix.abs()
        columns = corr_matrix_abs.columns
        high_corr_features = set()

        # We only need to look at upper triangular part to avoid duplication
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                if corr_matrix_abs.iloc[i, j] > threshold:
                    # Add both columns in the pair
                    high_corr_features.add(columns[i])
                    high_corr_features.add(columns[j])
        
        return high_corr_features
        

    def process_target_csv(self, csv_filepath):
        target_vector_unordered = pd.read_csv(csv_filepath)[self.output_name]
        self.target_vector = target_vector_unordered.loc[self.molecule_names]

    def fit(self, X, y):
        # Train the classifier
        # if self.ordinal:
        #     self.model = OrderedModel(y, X, distr='logit')  # Using logit for ordinal regression
        #     self.result = self.model.fit(method='bfgs')  # Using BFGS optimizer
        # else:
        self.model.fit(X, y)

    def predict(self, X):
        # Make predictions using the classifier
        # if self.ordinal:
        #     cumulative_probs = self.result.model.predict(self.result.params, exog=X)
        #     # Convert cumulative probabilities into class predictions (argmax approach)
        #     predicted_class = cumulative_probs.argmax(axis=1) + 1  # Adjust index for class labels
        #     return predicted_class
        # else:

        return self.model.predict(X)

    def compute_multicollinearity(self, vif_threshold=5.0):
        """
        Compute the Variance Inflation Factor (VIF) for each feature in the dataset.
        """
        # Compute VIF
        vif_results = self._compute_vif(self.features_df)
        
        # Identify features with high VIF
        high_vif_features = vif_results[vif_results['VIF'] > vif_threshold]
        
        return vif_results
    
    def calculate_vif(self):
        X = self.features_df
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data

    def evaluate(self, X, y):
        # Evaluate the classifier using different metrics
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred,average='weighted', zero_division=0)
        recall = recall_score(y, y_pred,average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred,average='weighted', zero_division=0)
        # auc = roc_auc_score(y, self.model.predict_proba(X)[:, 1])
        mcfadden_r2_var = self.mcfadden_r2(X, y)
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mc_fadden_r2': mcfadden_r2_var
            #'auc': auc
        }
        return results

    # def cross_validation(self, X, y):
    #     n_splits=self.n_splits
    #     # Perform k-fold cross-validation
    #     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
    #     # Define custom scorers
    #     f1_scorer = make_scorer(f1_score, average='weighted', zero_division=0)  # Choose 'macro', 'micro', or 'weighted'
    #     roc_auc_scorer = make_scorer(roc_auc_score, multi_class='ovr', average='weighted', zero_division=0)  # Choose 'ovr' or 'ovo', and averaging method

    #     # Perform cross-validation
    #     accuracy = cross_val_score(self.model, X, y, cv=kf, scoring='accuracy')
    #     f1 = cross_val_score(self.model, X, y, cv=kf, scoring=f1_scorer)
    #     # auc = cross_val_score(self.model, X, y, cv=kf, scoring=roc_auc_scorer)
        
    #     # Calculate mean values
    #     avg_accuracy = np.mean(accuracy)
    #     avg_f1 = np.mean(f1)
    #     # avg_auc = np.mean(auc)

    #     # calculate mcfadden r2
    #     r2_scorer = make_scorer(self.mcfadden_r2, greater_is_better=True)
    #     mcfadden_r2_var = cross_val_score(self.model, X, y, cv=kf, scoring=r2_scorer)
    #     avg_mcfadden_r2 = np.mean(mcfadden_r2_var)
        
        
    #     return avg_accuracy, avg_f1, avg_mcfadden_r2


    def cross_validation(self, X, y, n_splits=5):
        
        # Initialize lists to store metrics
        accuracy_list = []
        f1_list = []
        mcfadden_r2_list = []

        # Use appropriate KFold splitter
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Loop over the folds manually
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit the model on the training data
            if self.ordinal:
               
                result = self.model.fit(X_train, y_train)
                # Predict probabilities and classes
                y_pred = result.predict(X_test)
                # y_pred = np.argmax(y_pred_prob, axis=1)
                mcfadden_r2= self.mcfadden_r2(X_test, y_test)
                mcfadden_r2_list.append(mcfadden_r2)

            else:
                
                result = self.model.fit(X_train, y_train)
                y_pred = result.predict(X_test)
                mcfadden_r2 = self.mcfadden_r2(X_test, y_test)
                mcfadden_r2_list.append(mcfadden_r2)

            # Compute accuracy and F1-score
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            accuracy_list.append(accuracy)
            f1_list.append(f1)

        # Calculate mean values
        avg_accuracy = np.mean(accuracy_list)
        avg_f1 = np.mean(f1_list)
        avg_mcfadden_r2 = np.mean(mcfadden_r2_list)

        return avg_accuracy, avg_f1, avg_mcfadden_r2


    def fit_and_evaluate_combinations(self,top_n, n_jobs=-1,threshold=0.5,bool_parallel=False):
        app=self.app
        results,vif_df=[fit_and_evaluate_single_combination_classification(self,combination, threshold=threshold) for combination in tqdm(self.features_combinations, desc='Calculating combinations')]
        # print('results',results)
        sorted_results = sorted(results, key=lambda x: x['scores'].get('mc_fadden_r2', 0), reverse=True)
        sorted_results = sorted_results[:top_n]
        print_models_classification_table(sorted_results,app)
        self.models_list=[result['models'] for result in sorted_results]
        self.combinations_list=[result['combination'] for result in sorted_results]

        if self.leave_out:
            X = self.predict_features_df.to_numpy()
            y = self.predict_target_vector.to_numpy()
            self.fit(X, y)
            predictions = self.predict(X)
            result_dict={'sample_name':self.molecule_names_predict,'predictions':predictions,'true':y}
            if app:
                app.show_result('\n\n Predictions on left out samples\n\n')
                app.show_result(pd.DataFrame(result_dict).to_markdown(tablefmt="pipe", index=False))
            else:
                print(pd.DataFrame(result_dict).to_markdown(tablefmt="pipe", index=False))


        return sorted_results
    
    def iterative_cross_validation(self, combination, n_iter=10, n_splits=5):
        """
        Perform iterative cross-validation on a selected model and feature combination.

        Parameters:
        ------------
        model: statsmodels model object
            The fitted model to be evaluated.

        combination: tuple or list
            The feature combination used in the model.

        n_iter: int, optional (default=10)
            Number of iterations for cross-validation.

        n_splits: int, optional (default=5)
            Number of folds in each cross-validation iteration.

        Returns:
        ------------
        results: dict
            A dictionary containing:
            - 'overall_avg_accuracy': Overall average accuracy across all iterations.
            - 'overall_avg_f1_score': Overall average F1 score across all iterations.
            - 'overall_avg_mcfadden_r2': Overall average McFadden's R-squared across all iterations.
            - 'iteration_metrics': List of metrics for each iteration.
        """
        # Initialize lists to store overall metrics
        overall_accuracy_list = []
        overall_f1_list = []
        overall_mcfadden_r2_list = []
        iteration_metrics = []

        for i in range(n_iter):
            # For each iteration, perform cross-validation with a different random state
            random_state = 42 + i  # Change the random state for each iteration
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

            # Initialize lists for this iteration
            accuracy_list = []
            f1_list = []
            mcfadden_r2_list = []

            selected_features = self.features_df[list(combination)]
            X = selected_features.to_numpy()
            y = self.target_vector.to_numpy()
   
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                if self.ordinal:
                
                    result = self.model.fit(X_train, y_train)
                
                    # Predict probabilities and classes
                    y_pred = result.predict(X_test)
                    mcfadden_r2 = self.mcfadden_r2(X_test, y_test)
                else:
                    result = self.model.fit(X_train, y_train)
             
                    y_pred = result.predict(X_test)
                    mcfadden_r2 = self.mcfadden_r2(X_test, y_test)
                    mcfadden_r2_list.append(mcfadden_r2)


                # Compute accuracy and F1-score
                accuracy_cv = accuracy_score(y_test, y_pred)
                f1_cv = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                accuracy_list.append(accuracy_cv)
                f1_list.append(f1_cv)

            # Calculate average metrics for this iteration
            avg_accuracy = np.mean(accuracy_list)
            avg_f1 = np.mean(f1_list)
            avg_mcfadden_r2 = np.mean(mcfadden_r2_list)

            # Store the metrics for this iteration
            iteration_metrics.append({
                'iteration': i + 1,
                'avg_accuracy': avg_accuracy,
                'avg_f1_score': avg_f1,
                'avg_mcfadden_r2': avg_mcfadden_r2
            })

            # Add to overall metrics
            overall_accuracy_list.append(avg_accuracy)
            overall_f1_list.append(avg_f1)
            overall_mcfadden_r2_list.append(avg_mcfadden_r2)

        # Calculate overall average metrics
        overall_avg_accuracy = np.mean(overall_accuracy_list)
        overall_avg_f1 = np.mean(overall_f1_list)
        overall_avg_mcfadden_r2 = np.mean(overall_mcfadden_r2_list)

        results = {
            'overall_avg_accuracy': overall_avg_accuracy,
            'overall_avg_f1_score': overall_avg_f1,
            'overall_avg_mcfadden_r2': overall_avg_mcfadden_r2,
            'iteration_metrics': iteration_metrics
        }

        return results


    
    def mcfadden_r2(self, X, y):
        """
        Calculate McFadden's R-squared for the fitted model.

        Parameters:
        ------------
        X: array-like, shape (n_samples, n_features)
            Features dataset.

        y: array-like, shape (n_samples,)
            True labels.

        Returns:
        ------------
        mcfadden_r2: float
            McFadden's R-squared value.
        """
        if self.ordinal:
            mcfadden_r2 = self.model.result.prsquared

        else:
            # For regular logistic regression
            # Existing code remains the same
            y_prob = self.model.predict_proba(X)
            # Ensure y is encoded as integers starting from 0
            unique_classes = np.unique(y)
            class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
            y_indices = np.array([class_to_index[cls] for cls in y])

            # Compute log-likelihood for the fitted model
            LL_model = self._compute_log_likelihood(y_indices, y_prob)

            # Compute LL_null
            # Null model predicts the observed class probabilities
            class_counts = np.bincount(y_indices)
            class_probs = class_counts / np.sum(class_counts)
            y_null_prob = np.tile(class_probs, (X.shape[0], 1))
            LL_null = self._compute_log_likelihood(y_indices, y_null_prob)
            mcfadden_r2 = 1 - (LL_model / LL_null)

        
        
        return mcfadden_r2

    def _compute_log_likelihood(self, y_indices, y_prob):
        """
        Compute the log-likelihood of the model.

        Parameters:
        ------------
        y_indices: array-like, shape (n_samples,)
            True labels encoded as integers starting from 0.

        y_prob: array-like, shape (n_samples, n_classes)
            Predicted probabilities for each class.

        Returns:
        ------------
        log_likelihood: float
            The log-likelihood value.
        """
        # Clip probabilities to avoid log(0)
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)

        # Get the probabilities of the true classes
        prob_true_class = y_prob[np.arange(len(y_indices)), y_indices]

        # Compute log-likelihood
        log_likelihood = np.sum(np.log(prob_true_class))
        return log_likelihood


import os
# Usage





import time
if __name__ == "__main__":
    
    os.chdir(r'C:\Users\edens\Documents\GitHub\lucas_project\feather_example')
    csv_filepaths = {
    'features_csv_filepath': r'output_test.csv',
    'target_csv_filepath': ''
    }

    # model=ClassificationModel(csv_filepaths)
    # start_time = time.time()
    # results = model.fit_and_evaluate_combinations(n_jobs=-1)
    # end_time = time.time()
    # # print(results)
    # print_models_classification_table(results)


    model = LinearRegressionModel(
    csv_filepaths=csv_filepaths,
    process_method='one csv',
    output_name='output',
    leave_out=None,
    min_features_num=2,
    max_features_num=None,
    n_splits=5,
    return_coefficients=True
)

    results = model.fit_and_evaluate_combinations(n_jobs=-1)
#     end_time = time.time()

#     elapsed_time = end_time - start_time
#     print(f"Elapsed time: {elapsed_time:.2f} seconds")
    


    # print_models_regression_table(results)




