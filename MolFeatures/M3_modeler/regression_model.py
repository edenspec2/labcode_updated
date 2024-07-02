import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from tqdm import tqdm
from sklearn.model_selection import cross_val_predict, cross_validate
from itertools import combinations
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
def process_features_csv(csv_folderpath, process_method='one csv', output_name='output'):
    # input:
    # csv_folderpath (str): dataset path (from model_info)
    # leave_out (List of str): samples to use as an external validation ## molecule names!! 
    # output:
    # features_df (pandas df): dataset table  
    # features (List): All the molecular features in the dataset 
    # molecule_names (List): All the molecules in the dataset 
    # data_to_model (pandas df): dataset table for model after leave_out was 
    # filtered out (if applied)  
    features_df=pd.read_csv(csv_folderpath, sep=',', index_col=[0])
    if process_method=='one csv':
        try:
            target_vector=features_df[output_name]
        except KeyError:
            target_vector=features_df['class'] # quick fix for classification shahar file
    else:
        target_vector=None
    features_df=features_df.drop(output_name, axis=1)
    features_list=list(features_df.columns)
    molecule_names=list(features_df.index)
    return features_df, target_vector, features_list, molecule_names

def process_target_csv(csv_folderpath):
    target_vector=pd.read_csv(csv_folderpath, sep=',', index_col=[0]).squeeze()
    return target_vector

# def get_difference_values(iterable_1, iterable_2):
#     difference_values=set(iterable_1)-set(iterable_2)
#     return list(difference_values)

def leave_out_samples(features_df, molecule_names, leave_out=None):
    if leave_out is None:
        leave_out = []
    # not_removed_molecules=get_difference_values(molecule_names, leave_out)
    # if not_removed_molecules:
    #     for molecule_name in not_removed_molecules:
    #        print (f'Warning: {molecule_name} is in "leave_out" but not in the dataset')
    samples_to_keep=[molecule_name for molecule_name in molecule_names if molecule_name not in leave_out]
    data_to_model=features_df.loc[samples_to_keep]
    return data_to_model

def set_max_features_limit(total_features_num, max_features_num=None):
        if max_features_num==None:
            max_features_num=int(total_features_num/5.0)
        return max_features_num

def determine_number_of_features(total_features_num, min_features_num=2, max_features_num=4):
        max_features_num=set_max_features_limit(total_features_num, max_features_num)
        min_features_num=min_features_num
        return min_features_num, max_features_num

def get_feature_combinations(features, min_features_num=2, max_features_num=None):
    max_features_num = set_max_features_limit(len(features), max_features_num)
    def feature_combinations():
        for current_features_num in range(min_features_num, max_features_num + 1):
            for combo in combinations(features, current_features_num):
                yield combo
    return feature_combinations()

from sklearn.linear_model import LinearRegression

class LinearRegressionModel(BaseEstimator, RegressorMixin):

    def __init__(self, csv_filepaths, process_method='one csv', output_name='output', leave_out=None, min_features_num=2, max_features_num=4, mode='linear_regression', n_splits=5, metrics=None, return_coefficients=False):
        self.csv_filepaths = csv_filepaths
        self.process_method = process_method
        self.output_name = output_name
        self.leave_out = leave_out
        self.min_features_num = min_features_num
        self.max_features_num = max_features_num
        self.mode = mode


        self.set_mode(mode)
        self.set_predict_method(mode)
        if process_method == 'one csv':
            self.process_features_csv(csv_filepaths.get('features_csv_filepath'), process_method=process_method, output_name=output_name)
        elif process_method == 'two csvs':
            self.process_features_csv(csv_filepaths.get('features_csv_filepath'), process_method=process_method)
            self.process_target_csv(csv_filepaths.get('target_csv_filepath'))

        self.leave_out_samples(leave_out)
        self.determine_number_of_features(min_features_num, max_features_num)
        self.get_feature_combinations()

        self.n_splits = n_splits
        self.return_coefficients = return_coefficients
        self.metrics = metrics if metrics is not None else ['r2', 'neg_mean_absolute_error']
        self.scaler = StandardScaler()
        self.features_df = pd.DataFrame(self.scaler.fit_transform(self.features_df), columns=self.features_df.columns)



    def set_mode(self, mode):
        if mode in ['linear_regression', 'classification']:
            self.mode = mode
        else:
            raise ValueError("Mode must be 'linear_regression' or 'classification'")

    def set_predict_method(self, mode):
        if mode == 'classification':
            self.predict_method = 'predict_proba'
        elif mode == 'linear_regression':
            self.predict_method = 'predict'
        else:
            raise ValueError("Mode must be 'linear_regression' or 'classification'")
        
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Returns:
        dict: Parameter names mapped to their values.
        """
        return {"csv_filepaths": self.csv_filepaths, "process_method": self.process_method, "output_name": self.output_name, 
                "leave_out": self.leave_out, "min_features_num": self.min_features_num, "max_features_num": self.max_features_num, 
                "mode": self.mode, "n_splits": self.n_splits, "metrics": self.metrics, "return_coefficients": self.return_coefficients}

    def set_params(self, **params):
        """
        Set parameters for this estimator.

        Args:
        **params: Estimator parameters.

        Returns:
        self: Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    # def fit(self, X, y):
    #     """
    #     Train the linear regression model using the normal equation.

    #     Args:
    #     X (np.ndarray): Feature matrix.
    #     y (np.ndarray): Target vector.

    #     Returns:
    #     self: Fitted model.
    #     """
    #     X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add a bias term
    #     self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    #     return self

    def fit(self, X, y):
        """
        Fit the model to the data using scikit-learn's LinearRegression.

        Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        """
        self.model = LinearRegression()
        self.model.fit(X, y)
        return self

    
    
    def process_features_csv(self, csv_filepath, process_method, output_name):
        self.features_df, self.target_vector, self.features_list, self.molecule_names = process_features_csv(csv_filepath, process_method, output_name)

    def process_target_csv(self, csv_filepath, output_name):
        target_vector_unordered = process_target_csv(csv_filepath, output_name)
        self.target_vector = target_vector_unordered.loc[self.molecule_names]

    def leave_out_samples(self, leave_out=None):
        self.data_to_model = leave_out_samples(self.features_df, self.molecule_names, leave_out)

    def determine_number_of_features(self, min_features_num=2, max_features_num=4):
        self.min_features_num, self.max_features_num = determine_number_of_features(
            total_features_num=len(self.features_list),
            min_features_num=min_features_num,
            max_features_num=max_features_num
        )

    def get_feature_combinations(self):
        self.features_combinations = get_feature_combinations(
            features=self.features_list,
            min_features_num=self.min_features_num,
            max_features_num=self.max_features_num
        )
    

    def fit_and_evaluate_combinations(self):
        """
        Fit the model on each combination of features and evaluate their performance.

        Returns:
        list: A list of dictionaries containing combination, scores, and coefficients.
        """
        results = []
        for combination in tqdm(self.features_combinations): # , total=len(list(self.features_combinations)), desc="Evaluating Combinations"
            selected_features = self.features_df[list(combination)]
            X = selected_features.to_numpy()
            y = self.target_vector.to_numpy()

            # Fit the model
            self.fit(X, y)

            # Evaluate the model
            evaluation_results = self.evaluate(X, y)
            intercept, coefficients = self.get_coefficients_from_trained_estimator()

            # Store results
            result = {
                'combination': combination,
                'scores': evaluation_results,
                'intercept': intercept,
                'coefficients': coefficients
            }
            results.append(result)
            # print(f"Finished evaluating combination: {combination}")

        return results
    
    def train(self, X, y):
        """
        Train the linear regression model using the normal equation.

        Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        """
        # Add a column of ones to the feature matrix for the intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Compute the normal equation: theta = (X.T * X)^(-1) * X.T * y
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        """
        Make predictions using the trained linear regression model.

        Args:
        X (np.ndarray): Feature matrix.

        Returns:
        np.ndarray: Predicted values.
        """
        # Add a column of ones to the feature matrix for the intercept term
        # X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # # Compute predictions: y_hat = X * theta
        # return X_b.dot(self.theta)
        return self.model.predict(X)

    
    def evaluate(self, X, y):
        """
        Evaluate the model using specified metrics.

        Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

        Returns:
        dict: A dictionary with metric names and their corresponding values.
        """
        y_pred = self.predict(X)
        results = {}
        if 'r2' in self.metrics:
            results['r2'] = r2_score(y, y_pred)
        if 'neg_mean_absolute_error' in self.metrics:
            results['neg_mean_absolute_error'] = -mean_absolute_error(y, y_pred)
        return results
    
    def get_predictions_cross_validation(self):
        """
        Perform cross-validation and return predictions.

        Returns:
        np.ndarray: Cross-validated predictions.
        """
        if isinstance(self.n_splits, int):
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        else:
            kf = LeaveOneOut()

        predictions = np.zeros(len(self.y))
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train = self.y[train_index]
            self.train(X_train, y_train)
            predictions[test_index] = self.predict(X_test)
        return predictions

    def get_coefficients_from_trained_estimator(self):
        """
        Extract coefficients from a trained estimator.

        Args:
        estimator: A trained estimator with 'coef_' and 'intercept_' attributes.

        Returns:
        np.ndarray: Model coefficients.
        """
        
        coefficients = self.model.coef_ # [:-1]
        intercept = self.model.intercept_      # [-1]

        return coefficients, intercept
    
    def get_scores_and_coefficients_cross_validation(self):
        """
        Perform cross-validation, calculate metrics, and optionally return coefficients.

        Returns:
        dict: A dictionary with metric names and their corresponding values.
        np.ndarray: Model coefficients if return_coefficients is True.
        """
        if isinstance(self.n_splits, int):
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        else:
            kf = LeaveOneOut()

        scores = cross_validate(self, self.features_df, self.target_vector, cv=kf, scoring=self.metrics, return_estimator=self.return_coefficients)

        actual_scores = {metric: np.mean(scores['test_' + metric]) for metric in self.metrics}
        coefficients = None
        
        if self.return_coefficients:
            intercept, coefficients = self.get_coefficients_from_trained_estimator()

        return actual_scores, (intercept, coefficients)


    def return_scores_and_predictions_cross_validation(self):
        """
        Perform cross-validation and return scores and predictions.

        Returns:
        np.ndarray: Cross-validated predictions.
        dict: A dictionary with metric names and their corresponding values.
        np.ndarray: Model coefficients if return_coefficients is True.
        """
        predictions = cross_val_predict(self, self.features_df, self.target_vector, cv=self.n_splits, method=self.predict_method)
        scores, coefficients = self.get_scores_and_coefficients_cross_validation()
        return predictions, scores, coefficients

import os
from sklearn.linear_model import LinearRegression 

if __name__=='__main__':
    os.chdir(r'C:\Users\edens\Documents\GitHub\LabCode\MolFeatures\feather_example')
    csv_filepaths = {
    'features_csv_filepath': 'output_1.csv',
    'target_csv_filepath': ''
    }

    # Initialize the LinearRegressionModel
    model = LinearRegressionModel(csv_filepaths, process_method='one csv', output_name='output', leave_out=None, min_features_num=2, max_features_num=None, n_splits=5, return_coefficients=True)
    # Generate feature combinations
    model.get_feature_combinations()

    # Fit and evaluate combinations
    results = model.fit_and_evaluate_combinations()

    # Print or analyze results
    for result in results:
        print(f"Combination: {result['combination']}")
        print(f"Scores: {result['scores']}")
        print(f"Intercept: {result['intercept']}")
        print(f"Coefficients: {result['coefficients']}\n")
        
    # # Train the model
    # X = model.features_df.to_numpy()
    # y = model.target_vector.to_numpy()
    # model.fit(X, y)

    # # Evaluate the model
    # evaluation_results = model.evaluate(X, y)
    # print("Evaluation Results:", evaluation_results)

    # # Perform cross-validation and get predictions and scores
    # predictions, cross_val_scores, coefficients = model.return_scores_and_predictions_cross_validation()
    # print("Cross-Validation Scores:", cross_val_scores)
    # print("Intercept:", coefficients[0])
    # print("Coefficients:", coefficients[1])
    # # Evaluate the model
    # evaluation_results = model.evaluate(X, y)
    # print("Evaluation Results:", evaluation_results)

    # # Perform cross-validation and get predictions and scores
    # predictions, cross_val_scores, coefficients = model.return_scores_and_predictions_cross_validation()
    # print("Cross-Validation Scores:", cross_val_scores)
    # print("Coefficients:", coefficients)

