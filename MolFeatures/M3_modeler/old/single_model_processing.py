import os
import pandas as pd
import numpy as np

from itertools import combinations
# from tqdm import tqdm # visual option - is it important? 
from enum import Enum

from sklearn.model_selection import LeaveOneOut # CHANGE
from sklearn.metrics import r2_score # CHANGE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

class StrConstants(Enum):
    NAME_JOINER=' + '
    MODELS_SUMMARY='Summary.xls'

class OutputColumnNames(Enum):
    R2='R2'
    Q2='Q2'
    MAE='MAE'
    ACCURACY='accuracy'
    COEFF='Coefficients'
    PRECISION='precision'
    RECALL='recall'
    f1='f1'
    roc_auc='roc_auc'
    FEATURES='model_name'

class DfColumns(Enum):
    TRAIN_PREDICT=['True values', 'Predictions']

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

def determine_number_of_features(total_features_num, min_features_num=2, max_features_num=None):
        max_features_num=set_max_features_limit(total_features_num, max_features_num)
        min_features_num=min_features_num
        return min_features_num, max_features_num

def get_feature_combinations(features, min_features_num=2, max_features_num=None):
    # return:
    # model_combinations (list of lists): all the features combinations to 
    # model
    max_features_num=set_max_features_limit(len(features), max_features_num)
    features_combinations = []
    for current_features_num in range(min_features_num, max_features_num+1):
        features_combinations+=combinations(features, r = current_features_num)
    return features_combinations


def get_predictions_cross_validation(estimator, X, y, n_splits, predict_method='predict'): #'predict_proba'
    predictions=cross_val_predict(estimator,X, y, cv=n_splits, method=predict_method)
    return predictions

# def get_coefficients_from_trained_estimnator(estimator):
#     coefficients=estimator.coef_
#     intercept=estimator.intercept_
#     return np.append(coefficients, intercept)






def get_metric_name(metric):
    mapping = {
        'r2': OutputColumnNames.Q2.value,
        'neg_mean_absolute_error': OutputColumnNames.MAE.value,
        'accuracy': OutputColumnNames.ACCURACY.value,  # Assuming you have ACCURACY in OutputColumnNames
        'precision': OutputColumnNames.PRECISION.value,  # Assuming you have PRECISION in OutputColumnNames
        'recall': OutputColumnNames.RECALL.value,  # Assuming you have RECALL in OutputColumnNames
        'f1': OutputColumnNames.f1.value,  # Assuming you have f1 in OutputColumnNames
        'roc_auc': OutputColumnNames.roc_auc.value  # Assuming you have roc_auc in OutputColumnNames
        # Add other metrics as needed
    }
    return mapping.get(metric, metric)


def get_coefficients_from_trained_estimator(estimator):
    coefficients=estimator.coef_
    intercept=estimator.intercept_
    return np.append(coefficients, intercept)

def get_coefficients_from_trained_estimator(estimator):
    try:
        # Check if the estimator has 'coef_' and 'intercept_' attributes
        if hasattr(estimator, 'coef_') and hasattr(estimator, 'intercept_'):
            coefficients = estimator.coef_
            intercept = estimator.intercept_

            # Handle the case for multi-class classification
            if len(coefficients.shape) > 1:
                coefficients = coefficients.flatten()
            
            return np.append(coefficients, intercept)
        else:
            # Creating a dictionary with coefficients for each class
            coefficients = {}
            dm=0
            for index, class_coefficients in enumerate(estimator.coef_):
                coefficients[f'class_{index}'] = class_coefficients
            return np.append(coefficients, dm)

    except AttributeError as e:
        # Handle the attribute error
        print(f"Error: {e}")
        return None
    
def get_scores_and_coefficients_cross_validation(estimator, X, y, n_splits, 
                                                 metrics=None,
                                                 task='classification',
                                                 return_coefficients=False):
    coefficients = None
    if metrics is None:
        if task == 'classification':
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        else:
            metrics = ['r2', 'neg_mean_absolute_error']

    if task == 'regression' and 'r2' in metrics and isinstance(n_splits, LeaveOneOut):
        metrics = [metric for metric in metrics if metric != 'r2']
    # print(metrics)
    scores = cross_validate(estimator, X, y, cv=n_splits, scoring=metrics, return_estimator=return_coefficients)
    actual_scores = dict()

    for metric in metrics:
        metric_name = get_metric_name(metric)
        possible_score = scores.get('test_' + metric)
        try:
            actual_score = np.mean(possible_score)
            if metric_name==OutputColumnNames.MAE.value:
                actual_score = -1 * actual_score
            actual_scores.update({metric_name: actual_score})
        except TypeError:
            pass
    
    if return_coefficients:
        coefficients=get_coefficients_from_trained_estimator(scores.get('estimator')[0]) 


    return actual_scores, coefficients


def get_metric_name(metric):
    if metric == 'r2':
        metric_name = OutputColumnNames.Q2.value
    elif metric == 'neg_mean_absolute_error':
        metric_name = OutputColumnNames.MAE.value
    elif metric == 'accuracy':  # New metric for classification
        metric_name = OutputColumnNames.ACCURACY.value
    else:
        metric_name = metric  # Default to using the metric as its own name
    return metric_name


class ModelEvalCrossValidation():
    def __init__(self, X, y, n_splits, model_type='linear_regression' ,predict_method='predict', 
                 metrics=None,
                 return_coefficients=False):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.predict_method = predict_method
        self.task = model_type
        self.return_coefficients = return_coefficients

        if metrics is None:
            if self.task == 'classification':
                self.metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            else:  # default to regression metrics
                self.metrics = ['r2', 'neg_mean_absolute_error']
        else:
            self.metrics = metrics

        
    def get_predictions_cross_validation(self, estimator): 
        self.predictions=get_predictions_cross_validation(estimator, self.X, self.y, self.n_splits, self.predict_method)

    # def get_scores_and_coefficients_cross_validation(self, estimator):
    #     self.scores, self.coefficients=get_scores_and_coefficients_cross_validation(estimator, self.X, self.y, self.n_splits, self.metrics, self.return_coefficients)
    #     if ('r2' in self.metrics) and (get_metric_name('r2') not in self.scores.keys()): # CHANGE
    #         q_2_score = (-1) * r2_score(self.y, self.predictions)
    #         self.scores.update({OutputColumnNames.Q2.value: q_2_score})
    def get_scores_and_coefficients_cross_validation(self, estimator):
        self.scores, self.coefficients = get_scores_and_coefficients_cross_validation(
            estimator, self.X, self.y, self.n_splits, 
            metrics=self.metrics, 
            task=self.task, 
            return_coefficients=self.return_coefficients
        )
        # print(f"Scores: {self.scores}")
        # Assuming self.mode is an attribute that specifies 'regression' or 'classification'
        if self.task == 'linear_regression':
            # Handle regression-specific post-processing, e.g., calculating Q² if not present
            # print('metric:', self.metrics)
            if ('r2' in self.metrics) and (isinstance(self.n_splits, LeaveOneOut)):
                # Make sure self.predictions is computed for your regression model
                q_2_score = r2_score(self.y, self.predictions)
                # print('q_2:', q_2_score)
                self.scores.update({OutputColumnNames.Q2.value: q_2_score})
        elif self.task == 'classification':
            if 'accuracy' not in self.scores:
                accuracy = accuracy_score(self.y, self.predictions)
                self.scores.update({'accuracy': accuracy})
            # Add other classification metrics if not already included
            for metric in ['precision', 'recall', 'f1']:
                if metric not in self.scores:
                    score = globals()[f'{metric}_score'](self.y, self.predictions, average='weighted')
                    self.scores.update({metric: score})
            if 'roc_auc' not in self.scores and self.predictions.shape[1] == 2:
                roc_auc = roc_auc_score(self.y, self.predictions[:, 1])
                self.scores.update({'roc_auc': roc_auc})


    def return_scores_and_predictions_cross_validation(self, estimator):
        self.get_predictions_cross_validation(estimator)
        self.get_scores_and_coefficients_cross_validation(estimator)    
        return self.predictions, self.scores, self.coefficients

def get_cross_validation_results(training_set, target, n_splits, model_type='linear_regression',
                                 predict_method='predict', metrics=None, return_coefficients=False):

    # input:
    # training_set (numpy_arr): 2d array in which rows represent molecule and
    # columns represent features
    # target (numpy_arr): 2d array with 1 column. Each row contains a label.
    # folds (int): number of folds for Kfolds cross-validation 
    
    # output:
    # Q2 (float): Q2 score of the model on the training set 
    # MAE (float): MAE score of the model on the training set
    if model_type == 'classification':
        estimator = LogisticRegression(multi_class='multinomial')  # replace with your chosen classifier
        # Set default classification metrics if none are provided
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    elif model_type == 'linear_regression':
        estimator = LinearRegression()
        # Set default regression metrics if none are provided
        if metrics is None:
            metrics = ['r2', 'neg_mean_absolute_error']

    model_evaluator=ModelEvalCrossValidation(X=training_set,
                                             y=target,
                                             n_splits=n_splits,
                                             model_type=model_type,
                                             predict_method=predict_method, 
                                             metrics=metrics,
                                             return_coefficients=return_coefficients)
    predictions, scores, coefficients=model_evaluator.return_scores_and_predictions_cross_validation(estimator=estimator)
    # print(f'scores croess: {scores}')  
    # print(f'coefficients: {coefficients}')
    # print(f"Scores: {scores}")
    # print(f"Coefficients: {coefficients}")         
    return predictions, scores, coefficients



def get_r2_linear_regression(X,y):
    estimator=LinearRegression()
    _, r2_score=train_model(estimator, X, y, return_score=True)
    return r2_score



def get_accuracy_classification(X, y):
    estimator = LogisticRegression(multi_class='multinomial')  # Or any other classifier you prefer
    
    # Assuming train_model is a function that trains the model and returns predictions
    predictions, _ = train_model(estimator, X, y, return_predictions=True)
    probabilities, _ = train_model(estimator, X, y, return_predictions_proba=True)

    num_classes = y.nunique()
    class_columns = [f'Prob_Class_{i}' for i in range(num_classes)]
    probabilities_df = pd.DataFrame(probabilities, columns=class_columns)

    # Calculate metrics
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, average='weighted')
    recall = recall_score(y, predictions, average='weighted')
    f1 = f1_score(y, predictions, average='weighted')
    
    # ROC AUC calculation (only for binary classification or if y is binarized)
    if num_classes == 2 or (probabilities.shape[1] == 2):
        roc_auc = roc_auc_score(y, probabilities[:, 1])  # For binary classification
    else:
        roc_auc = "ROC AUC unavailable for multiclass"

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}, probabilities_df, predictions, estimator


def get_train_predictions_linear_regression(X, y, labels):
    print(f"Shapes in get_train_predictions_linear_regression - X: {X.shape}, y: {len(y)}")
    estimator = LinearRegression()
    train_predictions, *_ = train_model(estimator, X, y, return_predictions=True)
    print(f"Length of train_predictions: {len(train_predictions)}")
    return train_predictions, labels, estimator


 
def train_model(estimator, X, y, return_score=False, return_predictions=False, return_predictions_proba=False):
    # Fit the estimator
    estimator.fit(X, y)

    # Initialize variables for outputs
    score, predictions, probabilities = None, None, None
    
    # Calculate score if required
    if return_score:
        score = estimator.score(X, y)

    # Get predictions if required
    if return_predictions:
        predictions = estimator.predict(X)
        
    # Get prediction probabilities if required
    if return_predictions_proba:
        probabilities = estimator.predict_proba(X)

    # Return based on required outputs
    if return_predictions_proba:
        return probabilities, score
    return predictions, score




def test_model_vs_cutoff(X, y, n_splits, metric_cutoff=0.4, model_type='linear_regression',  predict_method='predict', metrics=None,
                         return_coefficients=False):
    # input:
    # feat_comb (list of str): list of features to include in the model
    # folds (int): number of folds for Kfolds cross-validation 
    # cutoff (float): cutoff to filter training sets with low r2. If r2 lower
    # than cutoff, no cross-validation is made.
    # coef (bool): indicate weather to return also the model coeeficients
    # output:
    # list of r2, q2 and MAE metric values. If coef is True, coefficient 
    # returned as well as a np.array. If r2 < cutoff, None is returned.
    
    final_results=dict()
    if model_type=='linear_regression':
        if metrics is None:
            metrics=['r2', 'neg_mean_absolute_error']
        r2_score=get_r2_linear_regression(X, y)
        if r2_score > metric_cutoff:
            final_results.update({OutputColumnNames.R2.value: r2_score})
            _, scores, coefficients=get_cross_validation_results(X, y, n_splits, model_type, predict_method, metrics, return_coefficients)
            final_results.update(scores) # CHANGE   
            if return_coefficients:
                final_results.update({OutputColumnNames.COEFF.value: coefficients})
    
    elif model_type == 'classification':
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        metric_results, probabilities, predictions, estimator = get_accuracy_classification(X, y)
        accuracy = metric_results['accuracy']
        if accuracy > metric_cutoff:
            for metric in metrics:
                if metric in metric_results:
                    final_results.update({metric.upper(): metric_results[metric]})
            if metrics is None:
                metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            _, scores, coefficients = get_cross_validation_results(X, y, n_splits, model_type, predict_method, metrics, return_coefficients)
            print(f"Scores: {scores}, coefficients: {coefficients}")
            if return_coefficients:
                final_results.update({OutputColumnNames.COEFF.value: coefficients})
                  
    return final_results

    
    

def rank_models(models_df, rank_by='MAE', top_n=10):
    """
    Ranks models based on a specified metric and returns the top n models.

    Parameters:
    - models_df (DataFrame): Table of different models (rows) and their metrics (columns).
    - rank_by (str): Metric to rank models by ('Q2', 'R2', 'MAE').
    - top_n (int): Number of top models to include in the results.

    Returns:
    - DataFrame: Ranked models table.
    """
    
    # Ensure the rank_by column exists in models_df
    
    assert rank_by in models_df.columns, f'"rank_by" value "{rank_by}" is not a valid column name in models_df'

    # Determine sorting order (False for metrics where higher is better, True otherwise)
    ascending = True if rank_by == 'MAE' else False

    # Sort and select top n models
    ranked_models_df = models_df.sort_values(by=[rank_by], ascending=ascending).head(top_n)

    return ranked_models_df

# def test_and_rank_subset_models(features_df, target_vector, features_combinations='all', n_splits=None, min_features_num=2, max_features_num=None, 
#                metric_cutoff=0.4, model_type='linear_regression' ,predict_method='predict', metrics=None, return_coefficients=False,
#                rank_by='MAE', top_n=10):
#     # input:
#     # folds (int or None): number of folds for Kfolds cross-validation. If 
#     # None, folds will be set as the total number of samples (as in LOO CV)
#     # cutoff (float): cutoff to filter training sets with low r2. If r2 lower
#     # than cutoff, no cross-validation is made.
#     # rank_by (str): Metric value to rank models by ('Q2', 'R2' or 'MAE')
#     # top_n (int): Number of top models (according to ranking) to include in
#     # the results
#     # output:
#     # Ranked models table (dataframe)
#     if model_type=='linear_regression':
#         metrics=['r2', 'neg_mean_absolute_error']
#         rank_by='MAE'
#     elif model_type=='classification':
#         metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
#         rank_by='ACCURACY'

#     final_results=[]               
#     n_splits=n_splits if n_splits else LeaveOneOut() # CHANGE
#     print(f'n splits: {n_splits}')
#     if features_combinations=='all':
#         features_combinations=get_feature_combinations(features_df.columns, min_features_num, max_features_num)
#     print (f"number of calculated iterations: {len(features_combinations)}")
#     if n_splits is None:
#         n_splits=features_df.shape[0]
#     total_combinations = len(features_combinations)
#     print('varify')
#     for feature_combination in tqdm(features_combinations,total=total_combinations, desc="Testing Models"): #tqdm?
#         model_results=test_model_vs_cutoff(X=features_df.filter(items=feature_combination, axis=1), 
#                                             y=target_vector, 
#                                             n_splits=n_splits, 
#                                             metric_cutoff=metric_cutoff,
#                                             model_type=model_type,
#                                             return_coefficients=return_coefficients,
#                                             predict_method=predict_method, 
#                                             metrics=metrics)
#         if model_results:
#             model_name=StrConstants.NAME_JOINER.value.join(feature_combination)
#             model_results.update({OutputColumnNames.FEATURES.value: model_name})
#             final_results.append(model_results)  
               
#     models_df=pd.DataFrame(final_results)
   
#     # print(f'Models df: {models_df}')
#     ranked_models_df=rank_models(models_df, rank_by, top_n)        
#     return ranked_models_df

from sklearn.model_selection import LeaveOneOut
from joblib import Parallel, delayed

def test_and_rank_subset_models(features_df, target_vector, features_combinations='all', n_splits=None, min_features_num=2, max_features_num=None, 
               metric_cutoff=0.4, model_type='linear_regression' ,predict_method='predict', metrics=None, return_coefficients=False,
               rank_by='MAE', top_n=10, n_jobs=-1):
    """
    Test and rank subset models using cross-validation.

    Args:
    - features_df (pd.DataFrame): DataFrame containing the feature set.
    - target_vector (pd.Series or np.ndarray): Target values.
    - features_combinations (str or list): List of feature combinations to test, or 'all' to test all combinations.
    - n_splits (int or None): Number of folds for cross-validation. If None, use Leave-One-Out.
    - min_features_num (int): Minimum number of features in a combination.
    - max_features_num (int or None): Maximum number of features in a combination. If None, use all.
    - metric_cutoff (float): Cutoff value for metrics to filter out low-performing models.
    - model_type (str): Type of model ('linear_regression' or 'classification').
    - predict_method (str): Prediction method ('predict' or 'predict_proba').
    - metrics (list or None): List of metrics to evaluate.
    - return_coefficients (bool): Whether to return model coefficients.
    - rank_by (str): Metric to rank models by ('Q2', 'R2', 'MAE', 'ACCURACY', etc.).
    - top_n (int): Number of top models to return.
    - n_jobs (int): Number of jobs to run in parallel. -1 means using all processors.

    Returns:
    - pd.DataFrame: Ranked models table.
    """
    if model_type == 'linear_regression':
        metrics = ['r2', 'neg_mean_absolute_error']
        rank_by = 'MAE'
    elif model_type == 'classification':
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        rank_by = 'ACCURACY'

    final_results = []
    n_splits = n_splits if n_splits else LeaveOneOut()
    
    if features_combinations == 'all':
        features_combinations = get_feature_combinations(features_df.columns, min_features_num, max_features_num)
    
    print(f"Number of calculated iterations: {len(features_combinations)}")

    if n_splits is None:
        n_splits = features_df.shape[0]

    total_combinations = len(features_combinations)
    
    # Define a helper function to be called in parallel
    def evaluate_combination(feature_combination):
        model_results = test_model_vs_cutoff(
            X=features_df.filter(items=feature_combination, axis=1), 
            y=target_vector, 
            n_splits=n_splits, 
            metric_cutoff=metric_cutoff,
            model_type=model_type,
            return_coefficients=return_coefficients,
            predict_method=predict_method, 
            metrics=metrics
        )
        if model_results:
            model_name = StrConstants.NAME_JOINER.value.join(feature_combination)
            model_results.update({OutputColumnNames.FEATURES.value: model_name})
            return model_results
        return None
    
    # Use joblib.Parallel to run the evaluations in parallel
    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_combination)(feature_combination)
        for feature_combination in tqdm(features_combinations, total=total_combinations, desc="Testing Models")
    )

    # Filter out None results
    final_results = [result for result in parallel_results if result is not None]
    
    models_df = pd.DataFrame(final_results)
    
    ranked_models_df = rank_models(models_df, rank_by, top_n)
    return ranked_models_df

def get_features_combination_name(models_df, model_num):

    # print(f"Model num: {model_num}, model shape: {models_df.shape} ")
    features_combination=models_df.iloc[model_num][OutputColumnNames.FEATURES.value].split(StrConstants.NAME_JOINER.value)
    return features_combination

def generate_df(data, df_index=None, df_columns=None):
    data_df=pd.DataFrame(data, index=df_index, columns=df_columns)
    return data_df

def generate_train_predictions_df(y, y_hat, labels):
    """
    Generates a DataFrame containing the true values, predicted values, and labels.

    Parameters:
    - y (array-like): The true values.
    - y_hat (array-like): The predicted values.
    - labels (array-like): The labels corresponding to each data point.

    Returns:
    - DataFrame: A DataFrame with columns for true values, predictions, and labels.
    """
    train_predictions_df = pd.DataFrame({
        'True values': y,
        'Predictions': y_hat,
        'Labels': labels
    })
    return train_predictions_df

class Model():
    
    # settings:
    # csv_folderpath (str): dataset path (.csv extention only). Each row in the dataset
    # represents a molecule (first column: molecules names). Each column represents
    # a property (first row: property name). last column should contain the output/target
    # values and its header (first row last column) should be "output".
    # leave_out (List of str): samples to use as external validation. If [] is 
    # inserted, no samples are used for validation. default is [].
    # min_feat (int): minimum features that a possible model can consider. Default
    # is 2
    # max_feat (int or None): maximum features that a possible model can consider
    # Default is None (in that case max_feat will be set as 1/5 of the dataset size
    # after the extraction of the external validation molecules) 
    
    def __init__(self, csv_filepaths, process_method='one csv', output_name='output', leave_out=None, min_features_num=2, max_features_num=None, mode='linear_regression'):
        self.set_mode(mode)
        self.set_predict_method(mode)
        if process_method=='one csv':
            self.process_features_csv(csv_filepaths.get('features_csv_filepath'), process_method=process_method, output_name=output_name)
        elif process_method=='two csvs':
            self.process_features_csv(csv_filepaths.get('features_csv_filepath'), process_method=process_method)
            self.process_target_csv(csv_filepaths.get('target_csv_filepath'))
        self.leave_out_samples(leave_out)
        self.determine_number_of_features(min_features_num, max_features_num)
        self.get_feature_combinations()

    
    def set_mode(self, mode):
        """
        Set the mode of the model to either 'regression' or 'classification'.

        Args:
        mode (str): The mode to set, either 'regression' or 'classification'.
        """
        if mode in ['linear_regression', 'classification']:
            self.mode = mode
        else:
            raise ValueError("Mode must be 'regression' or 'classification'")
        
    
    
    def set_predict_method(self, mode):
        """
        Set the mode of the model to either 'regression' or 'classification'.

        Args:
        mode (str): The mode to set, either 'regression' or 'classification'.
        """
        if mode == 'classification':
            self.predict_method='predict_proba'
        elif mode == 'linear_regression':
            self.predict_method='predict'
        else:
            raise ValueError("Mode must be 'regression' or 'classification'")

    def process_features_csv(self, csv_filepath, process_method, output_name):
        self.features_df, self.target_vector, self.features_list, self.molecule_names=process_features_csv(csv_filepath, process_method, output_name)

    def process_target_csv(self, csv_filepath, output_name):
        target_vector_unordered=process_target_csv(csv_filepath, output_name)
        self.target_vector=target_vector_unordered.loc[self.molecule_names]

    def leave_out_samples(self, leave_out=None):
        self.data_to_model=leave_out_samples(self.features_df, self.molecule_names, leave_out)
        
    def determine_number_of_features(self, min_features_num=2, max_features_num=None):
        self.min_features_num, self.max_features_num=determine_number_of_features(total_features_num=len(self.features_list),
                                                                                  min_features_num=min_features_num,
                                                                                  max_features_num=max_features_num,)        

    def get_feature_combinations(self): #examine if really needed
        self.features_combinations=get_feature_combinations(features=self.features_list,
                                                            min_features_num=self.min_features_num, 
                                                            max_features_num=self.max_features_num)

    # used for single-testing/semi-automatic - not automation   
    def test_model_vs_cutoff(self, feature_combination, n_splits=2, metric_cutoff=0.45,
                             metrics=None, return_coefficients=False):        
        # input:
        # feat_comb (list of str): list of features to include in the model
        # folds (int): number of folds for Kfolds cross-validation 
        # cutoff (float): cutoff to filter training sets with low r2. If r2 lower
        # than cutoff, no cross-validation is made.
        # coef (bool): indicate weather to return also the model coeeficients
        # output:
        # list of r2, q2 and MAE metric values. If coef is True, coefficient 
        # returned as well as a np.array. If r2 < cutoff, None is returned.
        if self.mode=='linear_regression':
            metrics=['r2', 'neg_mean_absolute_error']
        elif self.mode=='classification':
            metrics=['accuracy' , 'precision', 'recall', 'f1', 'roc_auc']

        training_set=self.features_df.filter(items=feature_combination).to_numpy()
        model_results=test_model_vs_cutoff(X=training_set, y=self.target_vector, 
                                           n_splits=n_splits,
                                           metric_cutoff=metric_cutoff,
                                           model_type=self.mode, 
                                           return_coefficients=return_coefficients,
                                           predict_method=self.predict_method, 
                                           metrics=metrics)
        return model_results

    

    def get_train_predictions(self, feature_combination, return_df=False):
    # input:
    # feat_comb (list of str): list of features to include in the model
    # df (bool): indicate whether to return values as list or as dataframe
    # output:
    # For regression: Tuple of target and predictions as 2D numpy arrays with one column.
    # For classification: Predicted classes.
    # If df is True, values are returned as one dataframe.
        training_set = self.features_df.filter(items=feature_combination).to_numpy()

        # Check the current mode and process accordingly
        if self.mode == 'linear_regression':
            train_predictions, labels, self.model= get_train_predictions_linear_regression(X=training_set, y=self.target_vector, labels=self.molecule_names)
            # print(f' estimator coef: {self.model.coef_}, intercept: {self.model.intercept_}')
        elif self.mode == 'classification':
            labels=self.molecule_names
            accuracy, probabilities, train_predictions, self.model =get_accuracy_classification(training_set, self.target_vector)
            # print(f' estimator coef: {self.model.coef_}, intercept: {self.model.intercept_}, classes: {self.model.classes_}')
            return self.target_vector,train_predictions, probabilities, labels, accuracy
        else:
            raise ValueError("Invalid mode. Mode should be either 'linear_regression' or 'classification'")

        if return_df:
            train_predictions_df = generate_train_predictions_df(y=self.target_vector, y_hat=train_predictions, labels=self.molecule_names)
            return train_predictions_df, labels
        else:
            return self.target_vector, train_predictions, labels


    def test_and_rank_subset_models(self, features_combinations, **subset_model_kwargs):
        self.models_df=test_and_rank_subset_models(features_df=self.features_df, 
                                                   target_vector=self.target_vector,
                                                   features_combinations=features_combinations,
                                                   model_type=self.mode,
                                                   predict_method=self.predict_method,
                                                   **subset_model_kwargs)

    def test_and_rank_all_subset_models(self, **subset_model_kwargs):
        self.models_df=test_and_rank_subset_models(features_df=self.features_df, 
                                                   target_vector=self.target_vector,
                                                   features_combinations=self.features_combinations,
                                                   min_features_num=self.min_features_num,
                                                   max_features_num=self.max_features_num,
                                                   model_type=self.mode,
                                                   predict_method=self.predict_method,
                                                   **subset_model_kwargs)

    # for the reports, should be better
    def get_cross_validation_results(self, features_combinations, n_splits=3, **cross_validation_kwargs):
        cross_validation_results=get_cross_validation_results(training_set=self.features_df.filter(items=features_combinations),
                                                              target=self.target_vector, 
                                                              n_splits=n_splits,
                                                              model_type=self.mode,
                                                              predict_method=self.predict_method,
                                                              **cross_validation_kwargs)
        return cross_validation_results