import time
from tqdm import tqdm
from itertools import combinations
from joblib import Parallel, delayed
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.preprocessing import StandardScaler
import numpy as np
import cProfile
import pstats

def set_max_features_limit(total_features_num, max_features_num=None):
    if max_features_num is None:
        max_features_num = int(total_features_num / 5.0)
    return max_features_num

def get_feature_combinations(features, min_features_num=2, max_features_num=None):
    max_features_num = set_max_features_limit(len(features), max_features_num)
    print(f"len features: {len(features)}")  # Print the number of features
    print(f"Min features: {min_features_num}")
    print(f"Max features: {max_features_num}")
    
    total_combinations = 0  # Counter to track the total number of combinations generated
    
    for current_features_num in range(min_features_num, max_features_num + 1):
        count_combinations = 0  # Counter for each specific number of features
        for combo in combinations(features, current_features_num):
            count_combinations += 1
            total_combinations += 1
            yield combo
        
        # Print the count of combinations for each number of features
        print(f"Combinations with {current_features_num} features: {count_combinations}")
    
    # Print the total count of combinations after all have been generated
    print(f"Total combinations generated: {total_combinations}")
    print(combo)

def fit_and_evaluate_single_combination(model, combination, r2_threshold=0.5):
    selected_features = model.features_df[list(combination)]
    X = selected_features.to_numpy()
    y = model.target_vector.to_numpy()

    # Fit the model
    t0 = time.time()
    model.fit(X, y)
    # print(time.time() - t0)

    # Evaluate the model
    evaluation_results = model.evaluate(X, y)
    intercept, coefficients = model.get_coefficients_from_trained_estimator()

    # Check if R-squared is above the threshold
    if evaluation_results['r2'] > r2_threshold:
        q2, mae = model.calculate_q2_and_mae(X, y)
        evaluation_results['Q2'] = q2
        evaluation_results['MAE'] = mae

    # arrange the results based on highest q2
    # sorted_evaluation_results = sorted(evaluation_results, key=lambda x: x['Q2'], reverse=True)

    # Store results
    result = {
        'combination': combination,
        'scores': evaluation_results,
        'intercept': intercept,
        'coefficients': coefficients
    }
    
    # result.to_csv('linear_models.csv', mode='a')

    return result




class LinearRegressionModel:

    def __init__(self, csv_filepaths, process_method='one csv', output_name='output', leave_out=None, min_features_num=2, max_features_num=None, n_splits=5, metrics=None, return_coefficients=False):
        self.csv_filepaths = csv_filepaths
        self.process_method = process_method
        self.output_name = output_name
        self.leave_out = leave_out
        self.min_features_num = min_features_num
        self.max_features_num = max_features_num
        self.n_splits = n_splits
        self.metrics = metrics if metrics is not None else ['r2', 'neg_mean_absolute_error']
        self.return_coefficients = return_coefficients
        self.model = LinearRegression()

        if csv_filepaths:
            self.set_mode('linear_regression')
            self.set_predict_method('linear_regression')
            if process_method == 'one csv':
                self.process_features_csv(csv_filepaths.get('features_csv_filepath'), process_method=process_method, output_name=output_name)
            elif process_method == 'two csvs':
                self.process_features_csv(csv_filepaths.get('features_csv_filepath'), process_method=process_method)
                self.process_target_csv(csv_filepaths.get('target_csv_filepath'))
            self.leave_out_samples(leave_out)
            self.determine_number_of_features(min_features_num, max_features_num)
            self.get_feature_combinations()
            self.scaler = StandardScaler()
            
            self.features_df = pd.DataFrame(self.scaler.fit_transform(self.features_df), columns=self.features_df.columns)

    def set_mode(self, mode):
        self.mode = mode

    def set_predict_method(self, mode):
        self.predict_method = 'predict' if mode == 'linear_regression' else 'predict_proba'

    def process_features_csv(self, csv_filepath, process_method, output_name):
        df = pd.read_csv(csv_filepath)
        self.features_df = df.drop(columns=['Unnamed: 0'])
        self.target_vector = df[output_name]
        self.features_df= self.features_df.drop(columns=[output_name])
       
        self.features_list = self.features_df.columns.tolist()
        self.molecule_names = df.index.tolist()

    def process_target_csv(self, csv_filepath):
        target_vector_unordered = pd.read_csv(csv_filepath)[self.output_name]
        self.target_vector = target_vector_unordered.loc[self.molecule_names]

    def leave_out_samples(self, leave_out=None):
        self.features_df = self.features_df.drop(index=leave_out) if leave_out else self.features_df

    def determine_number_of_features(self, min_features_num=2, max_features_num=4):
        total_features_num = len(self.features_list)
        self.min_features_num = min_features_num
        self.max_features_num = set_max_features_limit(total_features_num, max_features_num)

    def get_feature_combinations(self):
        self.features_combinations = list(get_feature_combinations(self.features_list, self.min_features_num, self.max_features_num))
   

    def calculate_q2_and_mae(self, X, y):
        """
        Calculate Q² cross-validation and MAE for the model.

        Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

        Returns:
        tuple: Q² cross-validation score and MAE.
        """
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        y_pred = cross_val_predict(self.model, X, y, cv=kf)
        
        q2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        return q2, mae

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

    def predict(self, X):
        """
        Make predictions using the trained linear regression model.

        Args:
        X (np.ndarray): Feature matrix.

        Returns:
        np.ndarray: Predicted values.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add a bias term
        
        return X_b.dot(self.theta)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        results = {}
        if 'r2' in self.metrics:
            results['r2'] = r2_score(y, y_pred)
        if 'neg_mean_absolute_error' in self.metrics:
            results['neg_mean_absolute_error'] = -mean_absolute_error(y, y_pred)
        return results

    def cross_validate(self, X, y, n_splits=5):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred = cross_val_predict(self.model, X, y, cv=cv)
        scores = cross_validate(self.model, X, y, cv=cv, scoring=self.metrics)
        return y_pred, scores
    
    def get_coefficients_from_trained_estimator(self):
        intercept = self.theta[0]
        coefficients = self.theta[1:]  
        return intercept, coefficients

    def fit_and_evaluate_combinations(self, n_jobs=-1):
        print(f"Number of calculated iterations: {len(self.features_combinations)}")
        print(self.features_combinations)
        ## check that the len of combination is between min and max features
        
        # results = Parallel(n_jobs=n_jobs)(delayed(fit_and_evaluate_single_combination)(self, combination) for combination in self.features_combinations)
        results=[fit_and_evaluate_single_combination(self, combination) for combination in tqdm(self.features_combinations, desc='Calculating combinations')]
        sorted_results = sorted(results, key=lambda x: x['scores'].get('Q2', float('-inf')), reverse=True)
        return sorted_results

import os
# Usage
os.chdir(r'C:\Users\edens\Documents\GitHub\LabCode\MolFeatures\feather_example')
csv_filepaths = {
'features_csv_filepath': 'output_1.csv',
'target_csv_filepath': ''
}

model = LinearRegressionModel(
    csv_filepaths=csv_filepaths,
    process_method='one csv',
    output_name='output',
    leave_out=None,
    min_features_num=2,
    max_features_num=4,
    n_splits=5,
    return_coefficients=True
)


if __name__ == "__main__":

    results = model.fit_and_evaluate_combinations(n_jobs=-1)

    for result in results[0:50]:
        print(f"Combination: {result['combination']}")
        print(f"Scores: {result['scores']}")
        print(f"Intercept: {result['intercept']}")
        print(f"Coefficients: {result['coefficients']}\n")







