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
from sklearn.linear_model import BayesianRidge
import pymc as pm
import arviz as az
import multiprocessing
from typing import Iterable, List, Sequence, Set, Tuple, Union, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from plot import *
    from modeling_utils import simi_sampler, stratified_sampling_with_plots
    from modeling_utils import *
except:
    from M3_modeler.plot import *
    from M3_modeler.modeling_utils import simi_sampler, stratified_sampling_with_plots
    from M3_modeler.modeling_utils import *




import sqlite3
import os

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
    avg_accuracy = results.get('avg_accuracy')
    avg_f1_score = results.get('avg_f1_score')
    # Insert into SQLite
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        INSERT INTO classification_results (
            combination, accuracy, precision, recall, f1_score, mcfadden_r2, threshold, avg_accuracy, avg_f1_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    ''', (str(combination), accuracy, precision, recall, f1, mcfadden_r2, threshold, avg_accuracy, avg_f1_score))
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
        'avg_accuracy': [avg_accuracy],
        'avg_f1_score': [avg_f1_score],
        'threshold': [threshold]
    }

    result_df = pd.DataFrame(result_dict)
    print(result_df.head())
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
            avg_accuracy REAL,
            avg_f1_score REAL,
            threshold REAL,
            model TEXT
        );
    ''')
    print("Table 'regression_results' has been ensured to exist.")
    
    conn.commit()
    conn.close()


def insert_result_into_db(db_path, combination, r2, q2, mae,rmsd, threshold,model, csv_path='results.csv'):
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
        INSERT INTO regression_results (combination, r2, q2, mae, rmsd, threshold, model)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    ''', (str(combination), r2, q2, mae, rmsd, threshold, str(model)))
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
        'model': [model]
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

# --- small utilities ---------------------------------------------------------

def _combo_key(combo: Sequence[str]) -> str:
    """Canonical string key for a combo (order-insensitive)."""
    return ",".join(sorted(map(str, combo)))

def _to_df(maybe_df) -> pd.DataFrame:
    """Coerce list/None/DataFrame to a DataFrame."""
    
    if isinstance(maybe_df, pd.DataFrame):
        return maybe_df
    if maybe_df is None:
        return pd.DataFrame()
    if isinstance(maybe_df, list):
        return pd.DataFrame(maybe_df)
    return pd.DataFrame(maybe_df)

def _ensure_numeric(s: pd.Series) -> pd.Series:
    """Coerce to numeric; invalid -> NaN."""
    return pd.to_numeric(s, errors="coerce")

def _extract_series(df: pd.DataFrame, names: Tuple[str, ...]) -> Optional[pd.Series]:
    """Return the first matching column (case-insensitive), else None."""
    if df.empty:
        return None
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return df[lower[n.lower()]]
    return None

def _extract_q2(df: pd.DataFrame) -> Optional[pd.Series]:
    """Get Q2 (or q2) as numeric; if only in 'scores' dicts, pull from there."""
    if df.empty:
        return None
    s = _extract_series(df, ("q2", "Q2"))
    if s is not None:
        return _ensure_numeric(s)
    # scores column path
    scores = _extract_series(df, ("scores",))
    if scores is not None:
        s = scores.apply(lambda d: (d or {}).get("q2", (d or {}).get("Q2", np.nan)))
        return _ensure_numeric(s)
    return None

def _extract_r2(df: pd.DataFrame) -> Optional[pd.Series]:
    s = _extract_series(df, ("r2", "R2"))
 
    return _ensure_numeric(s) if s is not None else None

def _is_all_q2_neg_inf(df: pd.DataFrame) -> bool:
    """True iff Q2 exists and every non-NaN value is exactly -inf (NaN => not -inf)."""
    q2 = _extract_q2(df)
    if q2 is None or q2.dropna().empty:
        return False
    return bool((q2.dropna() == -np.inf).all())

def _best_r2(df: pd.DataFrame) -> Optional[float]:
    r2 = _extract_r2(df)
    if r2 is None or r2.dropna().empty:
        return None
    return float(r2.max())

def _sort_results(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by q2 (desc) if present, then r2; else by r2 only."""

    if df.empty:
        return df
    q2 = _extract_q2(df)
    r2 = _extract_r2(df)

    temp = df.copy()
    if q2 is not None:
        temp["_q2__"] = q2
    if r2 is not None:
        temp["_r2__"] = r2
    by = []
    if "_q2__" in temp.columns:
        by.append("_q2__")
    if "_r2__" in temp.columns:
        by.append("_r2__")
    if not by:
        return df
    temp = temp.sort_values(by=by, ascending=[False]*len(by))
    return temp.drop(columns=[c for c in ("_q2__", "_r2__") if c in temp.columns])

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
    # "('L_11-6', 'buried_volume')" -> ['L_11-6','buried_volume']
    return [x.strip(" '") for x in s.strip("()").split(",")]
   
        # Print the count of combinations for each number of features
def fit_and_evaluate_single_combination_classification(model, combination, threshold=0.5, return_probabilities=False):
    selected_features = model.features_df[_parse_tuple_string(combination)]
    X = selected_features.to_numpy()
    y = model.target_vector.to_numpy()

    # Fit the model
    model.fit(X, y)

    # Evaluate the model
    evaluation_results, y_pred = model.evaluate(X, y)
    
    # Check if accuracy is above the threshold
    if evaluation_results['mcfadden_r2'] > threshold:
        avg_accuracy, avg_f1, avg_r2 = model.cross_validation(X, y , model.n_splits) ## , avg_auc
        evaluation_results['avg_accuracy'] = avg_accuracy
        evaluation_results['avg_f1_score'] = avg_f1
        evaluation_results['mcfadden_r2'] = avg_r2
   

    results={
        'combination': combination,
        'scores': evaluation_results,
        'model': model,
        'predictions': y_pred
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


def fit_and_evaluate_single_combination_regression(model, combination, r2_threshold=0.7):
    selected_features = model.features_df[list(combination)]
    X = selected_features.to_numpy()
    y = model.target_vector.to_numpy()
  
    # Fit the model
    t0 = time.time()
    model.fit(X, y)
    ## check
    model.model.fit(X, y)
    fit_time=time.time()-t0
    # Evaluate the modeld
    t1=time.time()
    model._trained_features = list(combination)
    evaluation_results, y_pred = model.evaluate(X, y)
    eval_time=time.time()-t1
    coefficients,intercepts = model.get_coefficients_from_trained_estimator()
 
    # Check if R-squared is above the threshold
    t3=time.time()
    if evaluation_results['r2'] > r2_threshold:
        q2, mae,rmsd = model.calculate_q2_and_mae(X, y, n_splits=1)
        evaluation_results['Q2'] = q2
        evaluation_results['MAE'] = mae
        evaluation_results['RMSD'] = rmsd
        print(f'R2:{evaluation_results["r2"]:.3f} Q2: {q2:.3f}, MAE: {mae:.3f}, RMSD: {rmsd:.3f} for combination: {combination}')

    q2_time=time.time()-t3
    # arrange the results based on highest q2
    # sorted_evaluation_results = sorted(evaluation_results, key=lambda x: x['Q2'], reverse=True)

    # Store results
    result = {
        'combination': combination,
        'scores': evaluation_results,
        'intercept': intercepts,
        'coefficients': coefficients,
        'model': model,
        'predictions': y_pred
    }

    if  'scores' in result:
        try:
            r2 = result['scores'].get('r2', float('-inf'))
            q2 = result['scores'].get('Q2', float('-inf'))
            mae = result['scores'].get('MAE', float('inf'))
            rmsd = result['scores'].get('RMSD', float('inf'))
            csv_path=model.db_path.replace('.db','.csv')
            model = result['model']
            # Insert into DB
            result_dict = {
                'combination': str(combination),
                'r2': r2,
                'q2': q2,
                'mae': mae,
                'rmsd': rmsd,
                'threshold': r2_threshold,
                'model': model
            }
            # print(type(model),'type of model variable')
            insert_result_into_db(
                db_path=model.db_path,
                combination=combination,
                r2=r2,
                q2=q2,
                mae=mae,
                rmsd=rmsd,
                threshold=r2_threshold,
                csv_path=csv_path,
                model=model
            )
            return result_dict
        except Exception as e:
            print(f'failed to insert combination : {combination}')
            print(e)

    return result

from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro, probplot

def check_linear_regression_assumptions(X,y):
    # Load data
    
    
    # Fit linear regression model
    model = sm.OLS(y, sm.add_constant(X)).fit()
    residuals = model.resid
    predictions = model.predict(sm.add_constant(X))


    print("\n----- Independence of Errors (Durbin-Watson) -----")
    dw_stat = durbin_watson(residuals)
    print(f"Durbin-Watson statistic: {dw_stat:.3f}")
    if 1.5 < dw_stat < 2.5:
        print("✅ No autocorrelation detected.")
    else:
        print("⚠️ Possible autocorrelation in residuals.")

    print("\n----- Homoscedasticity (Breusch-Pagan Test) -----")
    bp_test = het_breuschpagan(residuals, model.model.exog)
    p_value_bp = bp_test[1]
    print(f"Breusch-Pagan p-value: {p_value_bp:.3f}")
    if p_value_bp > 0.05:
        print("✅ Homoscedasticity assumed (good).")
    else:
        print("⚠️ Heteroscedasticity detected (bad).")

    print("\n----- Normality of Errors (Shapiro-Wilk Test) -----")
    shapiro_stat, shapiro_p = shapiro(residuals)
    print(f"Shapiro-Wilk p-value: {shapiro_p:.3f}")
    if shapiro_p > 0.05:
        print("✅ Residuals appear normally distributed.")
    else:
        print("⚠️ Residuals may not be normally distributed.")

    print("\n----- Normality of Errors (Q-Q Plot) -----")
    plt.figure()
    probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q plot of residuals')
    plt.show()


class PlotModel:
    def __init__(self, model):
        self.model = model
    
    


class LinearRegressionModel:

    def __init__(
            self, 
            csv_filepaths, 
            process_method='one csv', 
            names_column=None,
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
            db_path='results', 
            scale=True              # <--- If lasso, this is the regularization strength
    ):
        self.csv_filepaths = csv_filepaths
        self.process_method = process_method
        self.output_name = output_name
        self.names_column = names_column
        self.leave_out = leave_out
        self.min_features_num = min_features_num
        self.max_features_num = max_features_num
        self.metrics = metrics if metrics is not None else ['r2', 'neg_mean_absolute_error']
        self.return_coefficients = return_coefficients
        self.model_type = model_type
        self.alpha = alpha
        self.app = app
        self.n_splits = n_splits
        self.scale=scale
        self.name=os.path.splitext(os.path.basename(self.csv_filepaths.get('features_csv_filepath')))[0]


        self.db_path = db_path + f'_{self.name}.db'
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
                self.process_features_csv()
            elif process_method == 'two csvs':
                self.process_features_csv()
                self.process_target_csv(csv_filepaths.get('target_csv_filepath'))

            self.scaler = StandardScaler()
            self.original_features_df = self.features_df.copy()
            self.features_df = pd.DataFrame(self.scaler.fit_transform(self.features_df), columns=self.features_df.columns)
            self.feature_names=self.features_df.columns.tolist()
            
            self.compute_correlation()
            self.leave_out_samples(self.leave_out)
            self.determine_number_of_features()

            

            self.theta       = None
            self.X_b_train   = None
            self.sigma2      = None
            self.XtX_inv     = None

    

    def check_linear_regression_assumptions(self):
        return check_linear_regression_assumptions(self.features_df, self.target_vector)

    
    def compute_multicollinearity(self, vif_threshold=5.0):
        """
        Compute the Variance Inflation Factor (VIF) for each feature in the dataset.
        """
        # Compute VIF
        vif_results = self._compute_vif(self.features_df)
        
        # Identify features with high VIF
        high_vif_features = vif_results[vif_results['VIF'] > vif_threshold]
        
        return vif_results
    

    def process_features_csv(self):
        csv_filepath=self.csv_filepaths.get('features_csv_filepath')
        output_name=self.output_name
        names_column=self.names_column if hasattr(self, 'names_column') else None
        df = pd.read_csv(csv_filepath)
        if names_column is None:
            self.molecule_names = df.iloc[:, 0].tolist()
        else:
            self.molecule_names = df[names_column].tolist()
        self.features_df = df.drop(columns=[df.columns[0]])
        self.target_vector = df[output_name]
        self.features_df= self.features_df.drop(columns=[output_name])
       
        self.features_list = self.features_df.columns.tolist()

    def run_spike_and_slab_selection(self, slab_sd: float = 5.0, n_samples: int = 2000, n_tune: int = 1000,
                                     target_accept: float = 0.95, p_include: float = 0.5, verbose: bool = True):
        """
        Run spike-and-slab Bayesian variable selection using PyMC.
        Stores inclusion probabilities and posterior in self.spike_and_slab_result.
        """
        X = self.features_df.values
        y = self.target_vector.values if hasattr(self.target_vector, 'values') else self.target_vector

        p = X.shape[1]
        features = self.features_df.columns

        with pm.Model() as model:
            # Slab: wide Gaussian for nonzero
            spike = pm.Bernoulli('spike', p=p_include, shape=p)
            betas_slab = pm.Normal('betas_slab', mu=0, sigma=slab_sd, shape=p)
            betas = pm.Deterministic('betas', betas_slab * spike)
            intercept = pm.Normal('intercept', mu=0, sigma=10)
            sigma = pm.HalfNormal('sigma', sigma=1)

            mu = intercept + pm.math.dot(X, betas)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            trace = pm.sample(n_samples, tune=n_tune, target_accept=target_accept, chains=2, random_seed=self.random_state, progressbar=verbose)

        inclusion_probs = trace.posterior['spike'].mean(dim=("chain", "draw")).values

        self.spike_and_slab_result = {
            'trace': trace,
            'inclusion_probs': inclusion_probs,
            'features': features,
        }

        if verbose:
            print("Spike-and-Slab Feature Inclusion Probabilities:")
            for fname, prob in zip(features, inclusion_probs):
                print(f"{fname:20s}  P(included) = {prob:.3f}")

    def get_selected_features_spike_and_slab(self, threshold: float = 0.5):
        """
        Return features whose spike-and-slab inclusion probability exceeds the threshold.
        """
        if not hasattr(self, 'spike_and_slab_result'):
            raise ValueError("Run run_spike_and_slab_selection() first.")
        inclusion_probs = self.spike_and_slab_result['inclusion_probs']
        features = self.spike_and_slab_result['features']
        selected = [f for f, prob in zip(features, inclusion_probs) if prob >= threshold]
        return selected
    
    


    def compute_correlation(self, correlation_threshold=0.8, vif_threshold=5.0):
        app = self.app

        self.corr_matrix = self.features_df.corr()

        # Identify highly-correlated features above correlation_threshold
        high_corr_features = self._get_highly_correlated_features(
            self.corr_matrix, threshold=correlation_threshold
        )

        if high_corr_features:
            # Report findings
            msg = (
                f"\n--- Correlation Report ---\n"
                f"Features with correlation above {correlation_threshold}:\n"
                f"{list(high_corr_features)}\n"
            )
            if app:
                app.show_result(msg)
            print(msg)

            # === Always visualize if no app ===
            visualize = True
            if app:
                visualize = messagebox.askyesno(
                    title="Visualize Correlated Features?",
                    message="Would you like to see a heatmap of the correlation among these features?"
                )

            if visualize:
                sub_corr = self.corr_matrix.loc[
                    list(high_corr_features), list(high_corr_features)
                ]
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(sub_corr, annot=False, cmap='coolwarm', square=True, ax=ax)
                ax.set_title(f"Correlation Heatmap (>{correlation_threshold})")
                plt.tight_layout()
                plt.show()

            # === Ask to drop correlated features ===
            drop_corr = False
            if app:
                drop_corr = messagebox.askyesno(
                    title="Drop Correlated Features?",
                    message=(
                        f"Features above correlation {correlation_threshold}:\n"
                        f"{list(high_corr_features)}\n\n"
                        "Do you want to randomly drop some of these correlated features?"
                    )
                )

            if drop_corr:
                count_to_drop = len(high_corr_features) // 2
                features_to_drop = random.sample(list(high_corr_features), k=count_to_drop)

                drop_msg = (
                    f"\nRandomly selected {count_to_drop} features to drop:\n"
                    f"{features_to_drop}\n"
                )
                if app:
                    app.show_result(drop_msg)
                print(drop_msg)

                self.features_df.drop(columns=features_to_drop, inplace=True)
                self.features_list = self.features_df.columns.tolist()

                remaining_msg = f"Remaining features: {self.features_list}\n"
                if app:
                    app.show_result(remaining_msg)
                print(remaining_msg)
            else:
                no_drop_msg = "\nCorrelated features were not dropped.\n"
                if app:
                    app.show_result(no_drop_msg)
                print(no_drop_msg)

        else:
            msg = "\nNo features exceeded the correlation threshold.\n"
            if app:
                app.show_result(msg)
            print(msg)



    def _compute_vif(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the Variance Inflation Factor for each numeric column in df.
        """
        # 1) Select only numeric columns
        df_num = df.select_dtypes(include=[np.number]).copy()

        # 2) Handle missing or infinite values
        df_num.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_num.dropna(axis=1, how='any', inplace=True)
        df_num.dropna(axis=0, how='any', inplace=True)  # ensure no NaNs in rows
       
        # 3) Check matrix rank for multicollinearity warnings
        rank = np.linalg.matrix_rank(df_num.values)
        if rank < df_num.shape[1]:
            print(f"Warning: Linear dependence detected. Matrix rank: {rank} < {df_num.shape[1]}")

        # 4) Standardize data before VIF calculation
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_num),
            columns=df_num.columns,
            index=df_num.index
        )

        # 5) Compute VIF for each variable
        vif_data = {
            "variable": [],
            "VIF": []
        }
        for i, col in enumerate(df_scaled.columns):
            vif_val = variance_inflation_factor(df_scaled.values, i)
            vif_data["variable"].append(col)
            vif_data["VIF"].append(vif_val)

        vif_df = pd.DataFrame(vif_data)
        return vif_df


    
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
    def leave_out_samples(self, leave_out=None, keep_only=False):
        """
        Remove or retain specific rows from features_df/target_vector based on indices or molecule names.
        Stores the removed (or retained) samples into:
            - self.leftout_samples
            - self.leftout_target_vector
            - self.molecule_names_predict

        Parameters:
            leave_out (int, str, list[int|str]): Indices or molecule names to leave out (or to keep if keep_only=True).
            keep_only (bool): If True, only the given indices/names are kept, all others are left out.
        """
        if leave_out is None:
            return

        # Normalize input to a list
        if isinstance(leave_out, (int, np.integer, str)):
            leave_out = [leave_out]
        else:
            leave_out = list(leave_out)

        # Determine indices from mixed inputs (names or indices)
        indices = []
        name_to_index = {name: idx for idx, name in enumerate(self.molecule_names)}

        for item in leave_out:
            if isinstance(item, (int, np.integer)):
                indices.append(int(item))
            elif isinstance(item, str):
                try:
                    indices.append(name_to_index[item])
                except KeyError as e:
                    raise ValueError(f"Molecule name '{item}' not found in molecule_names.") from e
            else:
                raise ValueError("Items in leave_out must be integers or strings.")

        selected_features = self.features_df.iloc[indices].copy()
        selected_target = self.target_vector.iloc[indices].copy()
        selected_names = [self.molecule_names[i] for i in indices]

        # Ensure we have the full set of columns in the right order
        if hasattr(self, 'feature_names'):
            selected_features = selected_features.reindex(
                columns=self.feature_names,
                fill_value=0
            )

        # locate index labels for dropping
        self.idx_labels = self.features_df.index[indices]

        if keep_only:
            # Everything *not* in indices becomes the "left out" set
            self.leftout_samples = self.features_df.drop(index=self.idx_labels).copy()
            self.leftout_target_vector = self.target_vector.drop(index=self.idx_labels).copy()
            self.molecule_names_predict = [
                n for i, n in enumerate(self.molecule_names) if i not in indices
            ]

            # The new main set is just our selected rows (already reindexed)
            self.features_df = selected_features
            self.target_vector = selected_target
            self.molecule_names = selected_names
            self.indices_predict = indices
        else:
            # The selected rows become the "left out" set (already reindexed)
            self.leftout_samples = selected_features
            self.leftout_target_vector = selected_target
            self.molecule_names_predict = selected_names

            # Drop them from the main set
            self.features_df = self.features_df.drop(index=self.idx_labels)
            self.target_vector = self.target_vector.drop(index=self.idx_labels)
            self.molecule_names = [
                n for i, n in enumerate(self.molecule_names) if i not in indices
            ]
            self.indices_predict = indices


        

    def determine_number_of_features(self):
        total_features_num = self.features_df.shape[0]
        self.max_features_num = set_max_features_limit(total_features_num, self.max_features_num)
  


    def get_feature_combinations(self):
       
        self.features_combinations = list(get_feature_combinations(self.features_list, self.min_features_num, self.max_features_num))
   

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut, RepeatedKFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
    import numpy as np

    # analyze_shap_values(model, X, feature_names=None, target_name="output", n_top_features=10):

    def plot_shap_values(self, X, feature_names=None, target_name="output", n_top_features=10):
        """
        Plot SHAP values for the model's predictions.
        
        Args:
            X (np.ndarray): Feature matrix.
            feature_names (list, optional): Names of the features.
            target_name (str, optional): Name of the target variable.
            n_top_features (int, optional): Number of top features to display.
        """
        model= self.model
        
        analyze_shap_values(model, X, feature_names=feature_names, target_name=target_name, n_top_features=n_top_features)
        
    def calculate_q2_and_mae(self, X, y,
                         n_splits=None,
                         test_size=0.1,
                         random_state=42,
                         n_iterations=500,
                         exclude_leftout: bool = True):
        """
        Calculate Q² (cross-validated R²), MAE, and RMSD using scikit-learn's cross-validation or single train/test split.
        Q² is only meaningful when n_splits=1 (LOO CV).

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            n_splits (int, optional): Number of CV splits. If 1, does Leave-One-Out CV.
        Returns:
            tuple: (q2, mae, rmsd)
        """
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        scaler = StandardScaler()
        # Normalize if variance suggests it's needed
        if np.var(X) > 1: 
            X = scaler.fit_transform(X)
     
        if n_splits is None or n_splits == 0:
            print(f'Using single train/test split with test size {test_size} and random state {random_state}')
            self.model.fit(X, y)
            y_pred = self.model.predict(X)
            r2   = r2_score(y, y_pred)
            mae  = mean_absolute_error(y, y_pred)
            rmsd = np.sqrt(mean_squared_error(y, y_pred))
            return r2, mae, rmsd

        if n_splits < 1:
            raise ValueError("n_splits must be at least 1.")

        # --- LOO case ---
        if n_splits == 1:
            cv = LeaveOneOut()
            print("Using Leave-One-Out cross-validation (LOO)...")
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            print(f"Using {n_splits}-Fold cross-validation...")


        if exclude_leftout and hasattr(self, 'molecule_names_predict'):
            # build a mask of indices to KEEP in X, y
            keep_mask = [
                name not in self.molecule_names_predict
                for name in self.molecule_names
            ]
            X = X[keep_mask]
            y = y[keep_mask]

        y_pred = np.empty_like(y, dtype=float)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self.model.fit(X_train, y_train)
            y_pred[test_idx] = self.model.predict(X_test)

   
        q2   = r2_score(y, y_pred)
        mae  = mean_absolute_error(y, y_pred)
        rmsd = np.sqrt(mean_squared_error(y, y_pred))
        return q2, mae, rmsd




    def fit(self, X, y, alpha=1e-5):
        """
        Train linear regression with bias and L2 regularization,
        and precompute everything for predict().
        """
        # 1) build training design matrix with bias
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.X_b_train = X_b

        # 2) normal eqn with regularization (no penalty on bias term)
        A = X_b.T @ X_b
        I = np.eye(A.shape[0])
        I[0, 0] = 0
        self.theta = np.linalg.inv(A + alpha * I) @ (X_b.T @ y)

        # 3) residual variance σ² = SSE / (n – p)
        residuals = y - X_b.dot(self.theta)
        n, p = X_b.shape
        self.sigma2 = (residuals**2).sum() / (n - p)

        # 4) store (XᵀX)⁻¹ for later interval widths
        self.XtX_inv = np.linalg.inv(X_b.T @ X_b)

        return self

    def predict(self, X, return_interval=False, cl=0.95):
        """
        Make point predictions, and optionally return (lower, upper)
        prediction intervals at confidence level cl.
        """
        # 1) build new design matrix
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # 2) point predictions
        yhat = X_b.dot(self.theta)

        if not return_interval:
            return yhat
        residuals = self.target_vector - X_b.dot(self.theta)  # Assuming target_vector and features_df are stored
        self.residual_variance = np.sum(residuals ** 2) / (X_b.shape[0] - X_b.shape[1])
        # Calculate and store the covariance matrix of the coefficients
        self.model_covariance_matrix = self.residual_variance * np.linalg.inv(X_b.T @ X_b)
        # 3) t‐value for two-sided interval
        df   = self.X_b_train.shape[0] - self.X_b_train.shape[1]
        tval = t.ppf(1 - (1 - cl) / 2, df)

        # 4) standard errors: sqrt(σ² * (1 + xᵀ (XᵀX)⁻¹ x))
        #    vectorized for all rows
        #    (X_b @ XtX_inv) has shape [n_pred, p]; elementwise * X_b then sum across axis=1
        inside = 1 + np.sum((X_b @ self.XtX_inv) * X_b, axis=1)
        se_pred = np.sqrt(self.sigma2 * inside)

        # 5) intervals
        lower = yhat - tval * se_pred
        upper = yhat + tval * se_pred
        return yhat, lower, upper



    def predict_for_leftout(self, X, y=None, X_train=None, y_train=None,
                        calc_interval=False, confidence_level=0.95):
        """
        Predict on left-out samples using the already-fitted or freshly refitted model.

        Args:
            X (pd.DataFrame or np.ndarray): Left-out feature matrix.
            y (optional): Left-out target values.
            X_train, y_train (optional): If provided, will retrain model on this data before predicting.
            calc_interval (bool): If True, compute prediction intervals.
            confidence_level (float): Confidence level for prediction intervals.

        Returns:
            Prediction results as described earlier.
        """
        import numpy as np
        import pandas as pd
        import copy

        try:
            import statsmodels.api as sm
        except ImportError:
            sm = None

        # --- Step 1: Confirm feature list ---
        if not hasattr(self, "_trained_features"):
            raise AttributeError("Model is missing `_trained_features`. Set this after fitting.")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self._trained_features)
        else:
            X = X.copy()

        X = X[self._trained_features]
    
        X_arr = X.values
        if calc_interval:
            preds, lower, upper = self.predict(X_arr, return_interval=True, cl=confidence_level)
        else:
            preds = self.model.predict(X_arr)


        # --- Step 6: Compute errors ---
        errors = None
        if y is not None:
            y_arr = y.values.ravel() if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y).ravel()
            if preds.shape != y_arr.shape:
                raise ValueError(f"Prediction shape {preds.shape} != y shape {y_arr.shape}")
            errors = y_arr - preds

        # --- Step 7: Return ---
        if calc_interval:
            return (preds, lower, upper, errors) if errors is not None else (preds, lower, upper)
        else:
            return (preds, errors) if errors is not None else preds
        
    def evaluate(self, X, y):
        
        ## must use model.predict() to get the predictions
        y_pred = self.model.predict(X)
        results = {}
        if 'r2' in self.metrics:
            results['r2'] = r2_score(y, y_pred)
        return results , y_pred

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
    
    def get_covariance_matrix(self,features):
        intercept = self.theta[0]
        coefficients = self.theta[1:]
        cov_matrix = self.model_covariance_matrix
        std_errors = np.sqrt(np.diag(cov_matrix))
        t_values = coefficients / std_errors[1:]  # Ignoring intercept for t-value
        degrees_of_freedom = len(self.target_vector) - len(coefficients) - 1
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df=degrees_of_freedom))
        # Include intercept values for std_error, t_value, and p_value
        intercept_std_error = std_errors[0]
        intercept_t_value = intercept / intercept_std_error
        intercept_p_value = 2 * (1 - stats.t.cdf(np.abs(intercept_t_value), df=degrees_of_freedom))
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



    def search_models(
        self,
        top_n: int = 20,
        n_jobs: int = -1,
        initial_r2_threshold: float = 0.7,
        bool_parallel: bool = False,
        required_features: Optional[Iterable[str]] = None,
        min_models_to_keep: int = 5,
        threshold_step: float = 0.05,
        min_threshold: float = 0.2,
    ):
        """
        Fit and evaluate feature combinations with an optional set of required features.
        Threshold is relaxed automatically if all Q2 are -inf.

        Returns
        -------
        pd.DataFrame
            Top results, sorted by q2 then r2, limited to top_n.
        """
        app = self.app

        # Generate combinations up-front
        self.get_feature_combinations()
        all_combos: List[Tuple[str, ...]] = list(self.features_combinations or [])
        req: Set[str] = set(required_features or [])

        # Decide parallel
        cpu_count = multiprocessing.cpu_count()
        effective_jobs = 1 if (cpu_count == 1 or not bool_parallel) else (n_jobs if n_jobs != -1 else cpu_count)
        print(f"Using {effective_jobs} jobs for evaluation. Found {cpu_count} cores.")

        # Load existing results (avoid re-doing work)
        try:
            existing_results = _to_df(load_results_from_db(self.db_path))
            print(f"Loaded {len(existing_results)} existing results from DB.")
            # normalize combination keys
            if "combination" in existing_results.columns:
                done_combos = set(map(str, existing_results["combination"].tolist()))
            else:
                done_combos = set()
        except Exception:
            existing_results = pd.DataFrame()
            done_combos = set()

        def _filter_new_combos() -> List[Tuple[str, ...]]:
            """Not in DB and contains all required features."""
            out = []
            for combo in all_combos:
                key = str(combo)
                if key in done_combos:
                    continue
                if req and not req.issubset(set(combo)):
                    continue
                out.append(combo)
            return out

        def _evaluate_block(threshold: float) -> pd.DataFrame:
            """
            Evaluate only combos not yet done & satisfying required features.
            If parallel, assume each call writes to DB; then reload.
            """
            results_df = existing_results.copy()

            combos_to_run = _filter_new_combos()
            print(f"Combos to run: {len(combos_to_run)}, done_combos: {len(done_combos)}")
            if not combos_to_run:
                print(f"No new combinations to evaluate at threshold {threshold:.3f}.")
                return results_df

            print(f"Evaluating {len(combos_to_run)} new combos with R2 >= {threshold:.3f}...")
            if effective_jobs == 1:
                new_results = []
                for combo in tqdm(combos_to_run, desc=f"Threshold {threshold:.3f} (single-core)"):
                    try:
                        res = fit_and_evaluate_single_combination_regression(self, combo, threshold)
                        new_results.append(res)
                    except Exception:
                        # swallow or log; keep going
                        pass
                new_df = _to_df(new_results)
                results_df = pd.concat([results_df, new_df], ignore_index=True) if not new_df.empty else results_df
               
            else:
                Parallel(n_jobs=effective_jobs)(
                    delayed(fit_and_evaluate_single_combination_regression)(self, combo, threshold)
                    for combo in tqdm(combos_to_run, desc=f"Threshold {threshold:.3f} (parallel)")
                )
                # reload DB to include new results
                results_df = _to_df(load_results_from_db(self.db_path))
          
            return results_df

        # ---- initial pass -------------------------------------------------------
        results = _evaluate_block(initial_r2_threshold)
        results = _sort_results(results)
      

        # If everything's -inf on Q2, relax threshold and retry (bounded loop)
        attempts = 0
        threshold = float(initial_r2_threshold)
        while _is_all_q2_neg_inf(results) and threshold > min_threshold:
            print("All Q2 values are -inf, lowering R2 threshold and retrying...")
            best = _best_r2(results)
            # choose the next threshold: either step down from best or by a fixed step
            if best is not None:
                threshold = max(min_threshold, best - 0.15)
            else:
                threshold = max(min_threshold, threshold - threshold_step)
            print(f"New threshold: {threshold:.3f}")
            results = _evaluate_block(threshold)
            results = _sort_results(results)
            attempts += 1
            if attempts > 5:
                # avoid infinite loops
                break

        # keep at least min_models_to_keep even if top_n is small
        results = results.head(max(top_n, min_models_to_keep))

        # Show table
        if not results.empty:
            print_models_regression_table(results, app, self)


        return results



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
    def __init__(self, csv_filepaths, process_method='one csv', output_name='class', names_column=None, leave_out=None, min_features_num=2, max_features_num=None, n_splits=5, metrics=None, return_coefficients=False, ordinal=False, exclude_columns=None, app=None,db_path='results'):
        self.csv_filepaths = csv_filepaths
        self.process_method = process_method
        self.output_name = output_name
        self.leave_out = leave_out
        self.names_column = names_column
        self.min_features_num = min_features_num
        self.max_features_num = max_features_num
        self.metrics = metrics if metrics is not None else ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcfadden_r2']
        self.return_coefficients = return_coefficients
        self.n_splits = n_splits
        self.ordinal = ordinal
        self.app=app
        name=name = os.path.splitext(os.path.basename(self.csv_filepaths.get('features_csv_filepath')))[0]
        self.db_path = db_path + f'_{name}.db'
        create_results_table_classification(self.db_path)
        if csv_filepaths:
      
            if process_method == 'one csv':
                self.process_features_csv()
            elif process_method == 'two csvs':
                self.process_features_csv()
                self.process_target_csv(csv_filepaths.get('target_csv_filepath'))
            self.compute_correlation()
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
            self.leave_out_samples(self.leave_out, keep_only=False)
            self.determine_number_of_features()
            # self.get_feature_combinations()
            
            

        if ordinal:
            self.model = OrdinalLogisticRegression()
        else:    
            self.model = LogisticRegression(solver='lbfgs', random_state=42)




    def compute_correlation(self, correlation_threshold=0.8, vif_threshold=5.0):
        app = self.app

        self.corr_matrix = self.features_df.corr()

        # Identify highly-correlated features above correlation_threshold
        high_corr_features = self._get_highly_correlated_features(
            self.corr_matrix, threshold=correlation_threshold
        )

        if high_corr_features:
            # Report findings
            msg = (
                f"\n--- Correlation Report ---\n"
                f"Features with correlation above {correlation_threshold}:\n"
                f"{list(high_corr_features)}\n"
            )
            if app:
                app.show_result(msg)
            print(msg)

            # === Always visualize if no app ===
            visualize = True
            if app:
                visualize = messagebox.askyesno(
                    title="Visualize Correlated Features?",
                    message="Would you like to see a heatmap of the correlation among these features?"
                )

            if visualize:
                sub_corr = self.corr_matrix.loc[
                    list(high_corr_features), list(high_corr_features)
                ]
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(sub_corr, annot=False, cmap='coolwarm', square=True, ax=ax)
                ax.set_title(f"Correlation Heatmap (>{correlation_threshold})")
                plt.tight_layout()
                plt.show()

            # === Ask to drop correlated features ===
            drop_corr = False
            if app:
                drop_corr = messagebox.askyesno(
                    title="Drop Correlated Features?",
                    message=(
                        f"Features above correlation {correlation_threshold}:\n"
                        f"{list(high_corr_features)}\n\n"
                        "Do you want to randomly drop some of these correlated features?"
                    )
                )

            if drop_corr:
                count_to_drop = len(high_corr_features) // 2
                features_to_drop = random.sample(list(high_corr_features), k=count_to_drop)

                drop_msg = (
                    f"\nRandomly selected {count_to_drop} features to drop:\n"
                    f"{features_to_drop}\n"
                )
                if app:
                    app.show_result(drop_msg)
                print(drop_msg)

                self.features_df.drop(columns=features_to_drop, inplace=True)
                self.features_list = self.features_df.columns.tolist()

                remaining_msg = f"Remaining features: {self.features_list}\n"
                if app:
                    app.show_result(remaining_msg)
                print(remaining_msg)
            else:
                no_drop_msg = "\nCorrelated features were not dropped.\n"
                if app:
                    app.show_result(no_drop_msg)
                print(no_drop_msg)

        else:
            msg = "\nNo features exceeded the correlation threshold.\n"
            if app:
                app.show_result(msg)
            print(msg)


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
    

    def leave_out_samples(self, leave_out=None, keep_only=False):
        """
        Remove or retain specific rows from features_df/target_vector based on indices or molecule names.
        Stores the removed (or retained) samples into:
            - self.leftout_samples
            - self.leftout_target_vector
            - self.molecule_names_predict

        Parameters:
            leave_out (int, str, list[int|str]): Indices or molecule names to leave out (or to keep if keep_only=True).
            keep_only (bool): If True, only the given indices/names are kept, all others are left out.
        """
        if leave_out is None:
            return

        # Normalize input to a list
        if isinstance(leave_out, (int, np.integer, str)):
            leave_out = [leave_out]
        else:
            leave_out = list(leave_out)

        # Determine if the input is indices or molecule names
        if isinstance(leave_out[0], (int, np.integer)):
            indices = [int(i) for i in leave_out]
        elif isinstance(leave_out[0], str):
            name_to_index = {name: idx for idx, name in enumerate(self.molecule_names)}
            try:
                indices = [name_to_index[name] for name in leave_out]
            except KeyError as e:
                raise ValueError(f"Molecule name {e} not found in molecule_names.") from e
        else:
            raise ValueError("leave_out must be a list of integers or strings.")

        # --- stash the relevant data ---
        selected_features = self.features_df.iloc[indices].copy()
        selected_target   = self.target_vector.iloc[indices].copy()
        selected_names    = [self.molecule_names[i] for i in indices]

        # — ensure we have the full 106 columns in the right order —
        if hasattr(self, 'feature_names'):
            selected_features = selected_features.reindex(
                columns=self.feature_names,
                fill_value=0
            )

        # locate index labels for dropping
        idx_labels = self.features_df.index[indices]

        if keep_only:
            # Everything *not* in indices becomes the "left out" set
            self.leftout_samples       = self.features_df.drop(index=idx_labels).copy()
            self.leftout_target_vector = self.target_vector.drop(index=idx_labels).copy()
            self.molecule_names_predict = [
                n for i, n in enumerate(self.molecule_names) if i not in indices
            ]

            # The new main set is just our selected rows (already reindexed)
            self.features_df     = selected_features
            self.target_vector   = selected_target
            self.molecule_names  = selected_names
        else:
            # The selected rows become the "left out" set (already reindexed)
            self.leftout_samples       = selected_features
            self.leftout_target_vector = selected_target
            self.molecule_names_predict = selected_names

            # Drop them from the main set
            self.features_df    = self.features_df.drop(index=idx_labels)
            self.target_vector  = self.target_vector.drop(index=idx_labels)
            self.molecule_names = [
                n for i, n in enumerate(self.molecule_names) if i not in indices
            ]

        # debug prints
        print(f"Left out samples:   {self.molecule_names_predict}")
        print(f"Remaining samples:  {self.molecule_names}")
        
    def process_features_csv(self):
        csv_filepath=self.csv_filepaths.get('features_csv_filepath')
        output_name=self.output_name
        names_column=self.names_column if hasattr(self, 'names_column') else None
        df = pd.read_csv(csv_filepath)
        if names_column is None:
            self.molecule_names = df.iloc[:, 0].tolist()
        else:
            self.molecule_names = df[names_column].tolist()
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

        self.model.fit(X, y)

    def predict(self, X):

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
            'mcfadden_r2': mcfadden_r2_var
            #'auc': auc
        }
        return results , y_pred



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


    def search_models(self, top_n=50, n_jobs=-1, mcfadden_threshold=0.5, bool_parallel=False):
        app = self.app
        self.get_feature_combinations()
        existing_results = load_results_from_db(self.db_path, table='classification_results')
        print(f"Loaded {len(existing_results)} existing results from the database.")
        done_combos=existing_results['combination'].tolist()
        done_combos = set(done_combos)
        print(f"Skipping {len(done_combos)} combinations already in the database.")
        combos_to_run = [
            combo for combo in self.features_combinations
            if str(combo) not in done_combos
        ]
        print(f'Combos to run: {len(combos_to_run)}, done_combos: {len(done_combos)}')
        def process_and_insert(combo):
            # run the single-combo evaluation
            result = fit_and_evaluate_single_combination_classification(
                self, combo, threshold=mcfadden_threshold
            )
            print(f"Evaluated combination: {combo}, scores: {result}")
            # insert into DB & CSV
            insert_result_into_db_classification(
                self.db_path,
                combo,
                result['scores'],   # expects keys: accuracy, precision, recall, f1_score, mcfadden_r2
                mcfadden_threshold,
                csv_path='classification_results.csv'
            )

        # --- Execute evaluations ---
        if bool_parallel and n_jobs > 1 and multiprocessing.cpu_count() > 1:
            Parallel(n_jobs=n_jobs)(
                delayed(process_and_insert)(combo)
                for combo in tqdm(combos_to_run, desc='Parallel evaluation')
            )
        else:
            for combo in tqdm(combos_to_run, desc='evaluation'):
                process_and_insert(combo)

        # --- Reload the full, up-to-date results from the database ---
        all_results = load_results_from_db(self.db_path, table='classification_results')
       
        # --- Sort by McFadden R² and take top N ---
        # take top n results based on McFadden R²
        sorted_results = all_results.sort_values(by='mcfadden_r2', ascending=False).head(top_n)
        # --- Display and store the best models/combinations ---
        print_models_classification_table(sorted_results, app, self)
        
        self.combinations_list = sorted_results['combination'].tolist()

        # --- Optionally predict on left-out set ---
        if self.leave_out:
            X = self.predict_features_df.to_numpy()
            y = self.predict_target_vector.to_numpy()
            self.fit(X, y)
            preds = self.predict(X)
            df_lo = pd.DataFrame({
                'sample_name': self.molecule_names_predict,
                'true': y.ravel(),
                'predicted': np.array(preds).ravel()
            })
            if app:
                app.show_result("\n\nPredictions on left-out samples\n\n")
                app.show_result(df_lo.to_markdown(tablefmt="pipe", index=False))
            else:
                print(df_lo.to_markdown(tablefmt="pipe", index=False))

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







