import os
import pandas as pd
import numpy as np

from itertools import combinations
# from tqdm import tqdm # visual option - is it important? 
from enum import Enum

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, cross_validate

class StrConstants(Enum):
    NAME_JOINER=' + '
    MODELS_SUMMARY='Summary.xls'

class OutputColumnNames(Enum):
    R2='R2'
    Q2='Q2'
    MAE='MAE'
    COEFF='Coefficients'
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
        target_vector=features_df[output_name] # Add assert statement
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

def get_coefficients_from_trained_estimnator(estimator):
    coefficients=estimator.coef_
    intercept=estimator.intercept_
    return np.append(coefficients, intercept)

def get_scores_and_coefficients_cross_validation(estimator, X, y, n_splits, metrics=['r2', 'neg_mean_absolute_error'], #'explained_variance'
                                                 return_coefficients=False):
    coefficients=None
    n_splits = 4 # Q_2 isn't calculated well when doing leave-one-out cross-validation
    scores=cross_validate(estimator, X, y, cv=n_splits, scoring=metrics, return_estimator=return_coefficients)
    # for metric in metrics:
    actual_scores={'Q2': scores.get('test_r2'), 'MAE': scores.get('test_neg_mean_absolute_error')}
    # default_keys test_r2, test_neg_mean_absolute_error
    # for train metrics - set return_train_score=True
    if return_coefficients:
        coefficients=get_coefficients_from_trained_estimnator(scores.get('estimator')[0]) # Choose the right estimator, not just 0
    return actual_scores, coefficients

class ModelEvalCrossValidation():
    def __init__(self, X, y, n_splits, predict_method='predict', metrics=['r2', 'neg_mean_absolute_error'], return_coefficients=False):  #'explained_variance'
        self.X=X
        self.y=y
        self.n_splits=n_splits
        self.predict_method=predict_method
        self.metrics=metrics
        self.return_coefficients=return_coefficients
        
    def get_predictions_cross_validation(self, estimator): 
        self.predictions=get_predictions_cross_validation(estimator, self.X, self.y, self.n_splits, self.predict_method)

    def get_scores_and_coefficients_cross_validation(self, estimator):
        self.scores, self.coefficients=get_scores_and_coefficients_cross_validation(estimator, self.X, self.y, self.n_splits, self.metrics, self.return_coefficients)

    def return_scores_and_predictions_cross_validation(self, estimator):
        self.get_predictions_cross_validation(estimator)
        self.get_scores_and_coefficients_cross_validation(estimator)    
        return self.predictions, self.scores, self.coefficients

def get_cross_validation_results(training_set, target, n_splits, model_type='linear regression',
                                 predict_method='predict', metrics=['r2', 'neg_mean_absolute_error'], return_coefficients=False):
    # input:
    # training_set (numpy_arr): 2d array in which rows represent molecule and
    # columns represent features
    # target (numpy_arr): 2d array with 1 column. Each row contains a label.
    # folds (int): number of folds for Kfolds cross-validation 
    
    # output:
    # Q2 (float): Q2 score of the model on the training set 
    # MAE (float): MAE score of the model on the training set
    if model_type=='linear regression':
        estimator = LinearRegression()
    model_evaluator=ModelEvalCrossValidation(X=training_set,
                                             y=target,
                                             n_splits=n_splits, 
                                             predict_method=predict_method, 
                                             metrics=metrics,
                                             return_coefficients=return_coefficients)
    predictions, scores, coefficients=model_evaluator.return_scores_and_predictions_cross_validation(estimator=estimator)  
    # print(f"Scores: {scores}")
    # print(f"Coefficients: {coefficients}")         
    return predictions, scores, coefficients

# def train_model(estimator, X, y, return_score=False, return_predictions=False):
#     score, predictions= None, None
#     estimator.fit(X,y)
#     if return_score:
#         score=estimator.score(X,y)
#     if return_predictions:
#         predictions=estimator.predict(X)
#     return predictions, score

def get_r2_linear_regression(X,y):
    estimator=LinearRegression()
    _, r2_score=train_model(estimator, X, y, return_score=True)
    return r2_score

# def get_train_predictions_linear_regression(X,y):
#     estimator=LinearRegression()
#     train_predictions, *_=train_model(estimator, X, y, return_predictions=True)
#     return train_predictions

def get_train_predictions_linear_regression(X, y, labels):
    print(f"Shapes in get_train_predictions_linear_regression - X: {X.shape}, y: {len(y)}")

    estimator = LinearRegression()
    train_predictions, *_ = train_model(estimator, X, y, return_predictions=True)
    print(f"Length of train_predictions: {len(train_predictions)}")

    return train_predictions, labels

def train_model(estimator, X, y, return_score=False, return_predictions=False):
    estimator.fit(X, y)

    score, predictions = None, None
    if return_score:
        score = estimator.score(X, y)

    if return_predictions:
        predictions = estimator.predict(X)
        print(f"Length of predictions in train_model: {len(predictions)}")

    return predictions, score



def test_model_vs_cutoff(X, y, n_splits, metric_cutoff=0.4, model_type='linear regression',  predict_method='predict', metrics=['r2', 'neg_mean_absolute_error'],
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
    r2_score=get_r2_linear_regression(X, y)
    
    if r2_score > metric_cutoff:
        final_results.update({OutputColumnNames.R2.value: r2_score})
        _, scores, coefficients=get_cross_validation_results(X, y, n_splits, model_type, predict_method, metrics, return_coefficients)
        
        try:
            # final_results.update({OutputColumnNames.Q2.value: np.mean(scores.get('Q2'))})
            final_results[OutputColumnNames.Q2.value] = np.mean(scores.get('Q2'))
            
        except Exception as e:
            print(f"Error calculating Q2: {e}")
            final_results[OutputColumnNames.Q2.value] = np.nan
        final_results.update({OutputColumnNames.MAE.value: np.mean(scores.get('MAE'))})         
        if return_coefficients:
            final_results.update({OutputColumnNames.COEFF.value: coefficients})

    return final_results

# def rank_models(models_df, rank_by='MAE', top_n=10):
#     # input:
#     # models_table (dataframe): table of diffrent models (rows) and their 
#     # metrics (columns)
#     # rank_by (str): Metric value to rank models by ('Q2', 'R2' or 'MAE')
#     # top_n (int): Number of top models (according to ranking) to include in
#     # the results
#     # output:
#     # Ranked models table (dataframe)
#     assert rank_by in models_df.columns, '"rank_by" ({rank_by}) input value is not valid'
#     # ranked_models_df=models_df
#     ranked_models_df=models_df.sort_values(by=[rank_by], ascending=False).head(top_n)
#     return ranked_models_df

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

def test_and_rank_subset_models(features_df, target_vector, features_combinations='all', n_splits=None, min_features_num=2, max_features_num=None, 
               metric_cutoff=0.4, predict_method='predict', metrics=['r2', 'neg_mean_absolute_error'], return_coefficients=False,
               rank_by='MAE', top_n=10):
    # input:
    # folds (int or None): number of folds for Kfolds cross-validation. If 
    # None, folds will be set as the total number of samples (as in LOO CV)
    # cutoff (float): cutoff to filter training sets with low r2. If r2 lower
    # than cutoff, no cross-validation is made.
    # rank_by (str): Metric value to rank models by ('Q2', 'R2' or 'MAE')
    # top_n (int): Number of top models (according to ranking) to include in
    # the results
    # output:
    # Ranked models table (dataframe)
    final_results=[]               
    n_splits=n_splits if n_splits else len(features_df.index) # if n_splits is None, set as len(features_df.index) = LOO CV
    if features_combinations=='all':
        features_combinations=get_feature_combinations(features_df.columns, min_features_num, max_features_num)
    print (f"number of calculated iterations: {len(features_combinations)}")
    if n_splits is None:
        n_splits=features_df.shape[0]
    
    for feature_combination in features_combinations: #tqdm?
        model_results=test_model_vs_cutoff(X=features_df.filter(items=feature_combination, axis=1), 
                                           y=target_vector, 
                                           n_splits=n_splits, 
                                           metric_cutoff=metric_cutoff, 
                                           return_coefficients=return_coefficients,
                                           predict_method=predict_method, 
                                           metrics=metrics)
        if model_results:
            model_name=StrConstants.NAME_JOINER.value.join(feature_combination)
            model_results.update({OutputColumnNames.FEATURES.value: model_name})
            final_results.append(model_results)  
               
    models_df=pd.DataFrame(final_results)
    ranked_models_df=rank_models(models_df, rank_by, top_n)        
    return ranked_models_df

def get_features_combination_name(models_df, model_num):
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
    
    def __init__(self, csv_filepaths, process_method='one csv', output_name='output', leave_out=None, min_features_num=2, max_features_num=None):
        if process_method=='one csv':
            self.process_features_csv(csv_filepaths.get('features_csv_filepath'), process_method=process_method, output_name=output_name)
        elif process_method=='two csvs':
            self.process_features_csv(csv_filepaths.get('features_csv_filepath'), process_method=process_method)
            self.process_target_csv(csv_filepaths.get('target_csv_filepath'))
        self.leave_out_samples(leave_out)
        self.determine_number_of_features(min_features_num, max_features_num)
        self.get_feature_combinations()

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
    def test_model_vs_cutoff(self, feature_combination, n_splits=2, metric_cutoff=0.85,  predict_method='predict',
                             metrics=['r2', 'neg_mean_absolute_error'], return_coefficients=False):        
        # input:
        # feat_comb (list of str): list of features to include in the model
        # folds (int): number of folds for Kfolds cross-validation 
        # cutoff (float): cutoff to filter training sets with low r2. If r2 lower
        # than cutoff, no cross-validation is made.
        # coef (bool): indicate weather to return also the model coeeficients
        # output:
        # list of r2, q2 and MAE metric values. If coef is True, coefficient 
        # returned as well as a np.array. If r2 < cutoff, None is returned.
        training_set=self.features_df.filter(items=feature_combination).to_numpy()
        model_results=test_model_vs_cutoff(X=training_set, y=self.target_vector, 
                                           n_splits=n_splits,
                                           metric_cutoff=metric_cutoff, 
                                           return_coefficients=return_coefficients,
                                           predict_method=predict_method, 
                                           metrics=metrics)
        return model_results

    # used for single-testing/semi-automatic - not automation
    def get_train_predictions(self, feature_combination, return_df=False):
        # input:
        # feat_comb (list of str): list of features to include in the model
        # df (bool): indicate whether to return values as list or as dataframe
        # output:
        # Tuple of target and predictions as 2D numpy arrays with one column. If
        # df is True, values are returned as one dataframe. Now also returns labels.
        training_set = self.features_df.filter(items=feature_combination).to_numpy()
        
        # Assuming that `molecule_names` are the labels you want to return.
        labels_list = self.molecule_names
        
        train_predictions, labels = get_train_predictions_linear_regression(X=training_set, y=self.target_vector, labels=labels_list)
        print(f"Inside get_train_predictions - Length of train_predictions: {len(train_predictions)}")
        print(f"Inside get_train_predictions - Length of labels: {len(labels)}")


        
        if return_df:
            train_predictions_df = generate_train_predictions_df(y=self.target_vector, y_hat=train_predictions, labels=labels_list)
            print(f"Length of y: {len(self.target_vector)}")
            print(f"Length of y_hat: {len(train_predictions)}")
            print(f"Length of labels: {len(labels_list)}")
            return train_predictions_df, labels
        else:
            print(f"Length of y: {len(self.target_vector)}")
            print(f"Length of y_hat: {len(train_predictions)}")
            print(f"Length of labels: {len(labels_list)}")
            return self.target_vector, train_predictions, labels

    def test_and_rank_subset_models(self, features_combinations, **subset_model_kwargs):
        self.models_df=test_and_rank_subset_models(features_df=self.features_df, 
                                                   target_vector=self.target_vector,
                                                   features_combinations=features_combinations,
                                                   **subset_model_kwargs)

    def test_and_rank_all_subset_models(self, **subset_model_kwargs):
        self.models_df=test_and_rank_subset_models(features_df=self.features_df, 
                                                   target_vector=self.target_vector,
                                                   features_combinations=self.features_combinations,
                                                   min_features_num=self.min_features_num,
                                                   max_features_num=self.max_features_num,
                                                   **subset_model_kwargs)

    # for the reports, should be better
    def get_cross_validation_results(self, features_combinations, n_splits=3, model_type='linear regression', **cross_validation_kwargs):
        cross_validation_results=get_cross_validation_results(training_set=self.features_df.filter(items=features_combinations),
                                                              target=self.target_vector, 
                                                              n_splits=n_splits,
                                                              model_type=model_type,
                                                              **cross_validation_kwargs)
        return cross_validation_results