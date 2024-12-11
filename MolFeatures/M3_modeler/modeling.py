import time
from tqdm import tqdm
from itertools import combinations
from joblib import Parallel, delayed
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer, accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, cross_val_score
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

from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from plot import *
    from modeling_utils import simi_sampler, stratified_sampling_with_plots
    from modeling_utils import *
except:
    from M3_modeler.plot import *
    from M3_modeler.modeling_utils import simi_sampler, stratified_sampling_with_plots
    from M3_modeler.modeling_utils import *

    



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
        q2, mae = model.calculate_q2_and_mae(X, y,n_splits=X.shape[0])
        evaluation_results['Q2'] = q2
        evaluation_results['MAE'] = mae

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
    

    return result,fit_time,eval_time,q2_time




class PlotModel:
    def __init__(self, model):
        self.model = model
    
    


class LinearRegressionModel:

    def __init__(self, csv_filepaths, process_method='one csv', output_name='output', leave_out=None, min_features_num=2, max_features_num=None, n_splits=5, metrics=None, return_coefficients=False):
        self.csv_filepaths = csv_filepaths
        self.process_method = process_method
        self.output_name = output_name
        self.leave_out = leave_out
        self.min_features_num = min_features_num
        self.max_features_num = max_features_num
        self.metrics = metrics if metrics is not None else ['r2', 'neg_mean_absolute_error']
        self.return_coefficients = return_coefficients
        self.model = LinearRegression()
        self.n_splits = n_splits

        if csv_filepaths:
            if process_method == 'one csv':
                self.process_features_csv(csv_filepaths.get('features_csv_filepath'), output_name=output_name)
            elif process_method == 'two csvs':
                self.process_features_csv(csv_filepaths.get('features_csv_filepath'))
                self.process_target_csv(csv_filepaths.get('target_csv_filepath'))
           
            self.leave_out_samples(leave_out)
            self.determine_number_of_features()
            self.get_feature_combinations()
            self.scaler = StandardScaler()
            
            self.features_df = pd.DataFrame(self.scaler.fit_transform(self.features_df), columns=self.features_df.columns)


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

            

        

        self.features_df = self.features_df.drop(index=leave_out) if leave_out else self.features_df
        self.target_vector = self.target_vector.drop(index=leave_out) if leave_out else self.target_vector

        

    def determine_number_of_features(self):
        total_features_num = self.features_df.shape[0]
        self.max_features_num = set_max_features_limit(total_features_num, self.max_features_num)
  


    def get_feature_combinations(self):
       
        self.features_combinations = list(get_feature_combinations(self.features_list, self.min_features_num, self.max_features_num))
   

    def calculate_q2_and_mae(self, X, y):
        """
        Calculate Q² cross-validation and MAE for the model using manual splitting.

        Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        n_splits (int): Number of splits for cross-validation.

        Returns:
        tuple: Q² cross-validation score and MAE.
        """
        n_splits=self.n_splits
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        fold_size = n_samples // n_splits
        y_pred = np.zeros(n_samples)

        # Manual cross-validation
        for i in range(n_splits):
            # Create train and test indices
            start = i * fold_size
            end = start + fold_size if i != n_splits - 1 else n_samples
            
            test_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])

            # Split the data
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            # Train the model on the training data
            self.fit(X_train, y_train)

            # Predict on the test data
            y_pred[test_indices] = self.predict(X_test)

        # Calculate Q² and MAE
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
       
        # Make predictions
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



    def fit_and_evaluate_combinations(self, top_n=50, n_jobs=-1, initial_r2_threshold=0.85, app=False):
        def is_all_inf(results):
            return all(x['scores'].get('Q2', float('-inf')) == float('-inf') for x in results)

        def evaluate_with_threshold(threshold):
            # Perform evaluation with the specified R2 threshold
            # results = [fit_and_evaluate_single_combination_regression(self, combination, r2_threshold=threshold)
            #         for combination in tqdm(self.features_combinations, desc=f'Calculating combinations with threshold {threshold}')]
            results = Parallel(n_jobs=n_jobs)(
            delayed(self.fit_and_evaluate_single_combination_with_prints)(combination, threshold)
            for combination in tqdm(self.features_combinations, desc=f'Calculating combinations with threshold {threshold}')
        )

            results_eval = [result[0] for result in results if result]
            return results_eval

        def get_highest_r2(results):
            # Extract the highest R2 value from the results that are not -inf
            r2_values = [x['scores'].get('r2') for x in results ]
            if r2_values:
                print(max(r2_values))
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
    def __init__(self, csv_filepaths, process_method='one csv', output_name='class', leave_out=None, min_features_num=2, max_features_num=None,n_splits=5, metrics=None, return_coefficients=False,ordinal=False, exclude_columns=None):
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
        if csv_filepaths:
      
            if process_method == 'one csv':
                self.process_features_csv(csv_filepaths.get('features_csv_filepath'),  output_name=output_name)
            elif process_method == 'two csvs':
                self.process_features_csv(csv_filepaths.get('features_csv_filepath'))
                self.process_target_csv(csv_filepaths.get('target_csv_filepath'))
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
            
            print(self.calculate_vif())

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


    def fit_and_evaluate_combinations(self,top_n, n_jobs=-1,threshold=0.5, app=False):

        results=[fit_and_evaluate_single_combination_classification(self,combination, threshold=threshold) for combination in tqdm(self.features_combinations, desc='Calculating combinations')]
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




