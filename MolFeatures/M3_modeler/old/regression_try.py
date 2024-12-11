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


def print_models_classification_table(results):
    formulas=[result['combination'] for result in results]
    accuracy=[result['scores']['accuracy'] for result in results]
    precision=[result['scores']['precision'] for result in results]
    recall=[result['scores']['recall'] for result in results]
    f1=[result['scores']['f1_score'] for result in results]
    model_ids=[i for i in range(len(results))]
    models=[result['models'] for result in results]
    avg_accuracy=[result['scores'].get('avg_accuracy', float('-inf')) for result in results]
    avg_f1=[result['scores'].get('avg_f1_score', float('-inf')) for result in results]
    # avg_auc=[result['scores'].get('avg_auc', float('-inf')) for result in results]
    # Create a DataFrame from the inputs
    df = pd.DataFrame({
        'formula': formulas,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_accuracy': avg_accuracy,
        'avg_f1': avg_f1,
        #'avg_auc': avg_auc,

        'Model_id': model_ids
    })
    df.sort_values(by='avg_accuracy', ascending=False, inplace=True)
    # Set the index to range from 1 to n (1-based indexing)
    df.index = range(1, len(df) + 1)
    print(df.to_markdown(index=False, tablefmt="pipe"))
    try:
        df.to_csv('models_classification_table.csv', index=False)
    except:
        print('could not save the table')

    selected_model = int(input("Select a model number to print probabilities: "))
    model=models[selected_model]
    _,probablities_df = fit_and_evaluate_single_combination_classification(model,formulas[selected_model], return_probabilities=True)
    plot_probabilities(probablities_df)

    print(probablities_df.to_markdown(tablefmt="pipe"))

    # Print the confusion matrix
    y_pred = model.predict(model.features_df[list(formulas[selected_model])].to_numpy())
    y_true = model.target_vector.to_numpy()
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix\n")
    class_names=np.unique(y_true)
    class_names=[f'Class_{i}' for i in class_names]

    model_precision = precision_score(y_true, y_pred, average=None)
    model_recall = recall_score(y_true, y_pred, average=None)
    model_accuracy = accuracy_score(y_true, y_pred)

    print('precision',model_precision,'recall',model_recall,'accuracy',model_accuracy)
    plot_enhanced_confusion_matrix(cm, class_names, model_precision, model_recall, model_accuracy)



def print_models_regression_table(results):

    formulas=[result['combination'] for result in results]
    r_squared=[result['scores']['r2'] for result in results]
    q_squared=[result['scores'].get('Q2', float('-inf')) for result in results]
    mae=[result['scores'].get('MAE', float('-inf')) for result in results]
    model_ids=[i for i in range(len(results))]
    intercepts=[result['intercept'] for result in results]
    model_coefficients=[result['coefficients'] for result in results]
    models=[result['models'] for result in results]

    # Create a DataFrame from the inputs
    df = pd.DataFrame({
        'formula': formulas,
        'R.sq': r_squared,
        'Q.sq': q_squared,
        'MAE': mae,
        'Model_id': model_ids
    })

    # Sort the DataFrame by Q.sq (descending) for a similar order
    df = df.sort_values(by='Q.sq', ascending=False)
    
    # Set the index to range from 1 to n (1-based indexing)
    df.index = range(1, len(df) + 1)
    
    # Print the DataFrame as a markdown-like table
    print(df.to_markdown(index=False, tablefmt="pipe"))
    try:
        df.to_csv('models_table.csv', index=False)
    except:
        print('could not save the table')
    

    ## Need the actual model list to calculate CV3 CV5

    
    selected_model = int(input("Select a model number to print coefficients: "))
    model=models[selected_model]
   
    features=list(formulas[selected_model])
    
    X=model.features_df[features].to_numpy()

    y=model.target_vector.to_numpy()
    model.fit(X,y)
    pred,lwr,upr=model.predict(X,calc_covariance_matrix=True)
    coef_df=model.get_covariace_matrix(features)
    
    

    print("\nModel Coefficients\n")
    print(coef_df.to_markdown(tablefmt="pipe"))
    
    # Print 3-fold CV and 5-fold CV (mock data for now)
    print("\n3-fold CV\n")
    print(formulas[selected_model])
    
    Q2_3, MAE_3 = model.calculate_q2_and_mae(X, y, n_splits=3)
    print(pd.DataFrame({'Q2': [Q2_3], 'MAE': [MAE_3]}).to_markdown(tablefmt="pipe", index=False))
    
    Q2_5, MAE_5 = model.calculate_q2_and_mae(X, y, n_splits=5)
    print("\n5-fold CV\n")
    print(pd.DataFrame({'Q2':[Q2_5], 'MAE': [MAE_5]}).to_markdown(tablefmt="pipe", index=False))
    
    # Print Unnormalized Data Model Coefficients (mock data for now)
    # print("\nUnnormalized Data Model Coefficients\n")
    # unnormalized_coef_data = coef_data.copy()
    # unnormalized_coef_data['Estimate'] = [0.5813882, -0.5187489, 1.8009864, 1.3317469, -0.4933487]
    # unnormalized_coef_data['Std. Error'] = [0.2827138, 0.2577039, 0.1643857, 0.3477300, 0.2298582]
    # print(unnormalized_coef_data.to_markdown(tablefmt="pipe"))
    _=generate_q2_scatter_plot(y,pred,model.molecule_names,lwr,upr)
    # plt.show()

    



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
def fit_and_evaluate_single_combination_classification(model, combination, accuracy_threshold=0.7, return_probabilities=False):
        selected_features = model.features_df[list(combination)]
        X = selected_features.to_numpy()
        y = model.target_vector.to_numpy()

        # Fit the model
        model.fit(X, y)

        # Evaluate the model
        evaluation_results = model.evaluate(X, y)
      
        # Check if accuracy is above the threshold
        if evaluation_results['accuracy'] > accuracy_threshold:
            avg_accuracy, avg_f1 = model.cross_validation(X, y) ## , avg_auc
            evaluation_results['avg_accuracy'] = avg_accuracy
            evaluation_results['avg_f1_score'] = avg_f1
            # evaluation_results['avg_auc'] = avg_auc

        results={
            'combination': combination,
            'scores': evaluation_results,
            'models': model
        }

        if return_probabilities:

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
        q2, mae = model.calculate_q2_and_mae(X, y)
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

def set_q2_plot_settings(ax, lower_bound, upper_bound, fontsize=15):
    bounds_array = np.array([lower_bound, upper_bound])
    ax.plot(bounds_array, bounds_array, 'k--', linewidth=2)  # black dashed line
    ax.set_xlabel('Measured', fontsize=fontsize)  # Assuming 'Measured' is the label you want
    ax.set_ylabel('Predicted', fontsize=fontsize)
    ax.set_ylim(bounds_array)
    ax.set_xlim(bounds_array)
    ax.grid(True)  # Adding a grid

## might change in the future to plot confidence intervals as dotted lines calculated from the covariance matrix
def generate_q2_scatter_plot(y, y_pred, labels, lower_bound=None, upper_bound=None, figsize=(10, 10), fontsize=12, scatter_color='black'):
    # Create a DataFrame for seaborn usage
    data = pd.DataFrame({
        'Measured': y,
        'Predicted': y_pred,
        'Labels': labels
    })
    
    # Initialize the plot with seaborn's set style and context
    sns.set(style="whitegrid", context="notebook", rc={"figure.figsize": figsize})

    min_bound = np.min(lower_bound)
    max_bound = np.max(upper_bound)

    # Create the scatter plot with regression line using sns.lmplot
    plot = sns.lmplot(x='Measured', y='Predicted', data=data,
                      height=figsize[0]/2.54, aspect=figsize[0]/figsize[1], # height in inches and aspect ratio
                      scatter_kws={'s': 50, 'color': scatter_color},
                      line_kws={'color': 'black', 'lw': 2},
                      ci=95) # ci=None to not display the confidence interval

    # Adjusting the axes limits if bounds are provided
    
    
    # plot.ax.set_ylim(min_bound, max_bound)
    # plot.ax.set_xlim(np.min(y), np.max(y))

    # Adding annotations
    for i, row in data.iterrows():
        plot.ax.annotate(row['Labels'], (row['Measured'], row['Predicted']), 
                         textcoords="offset points", xytext=(5,5), ha='center', fontsize=fontsize)

    # Additional customization options directly with seaborn
    plot.set_axis_labels("Measured", "Predicted", fontsize=fontsize, weight='bold')
    plot.fig.suptitle('Regression Analysis with Labels', fontsize=fontsize+2, weight='bold')
    plt.show()


    return plot

import matplotlib.pyplot as plt

def plot_probabilities(probabilities_df):
    df = probabilities_df.copy()
    print(df)
    rankings = df[['Prob_Class_1', 'Prob_Class_2', 'Prob_Class_3']].rank(axis=1, ascending=False, method='min')
    df['Rank'] = df.apply(lambda row: rankings.loc[row.name, 'Prob_Class_' + str(row['Predicted_Class']).split('.')[0]], axis=1)
    color_map = {1: 'blue', 2: 'green', 3: 'red'}
    df['Color_Code'] = df['Rank'].map(color_map)
    df['Labels'] = df.apply(lambda row: f"Sample_{row.name} (Pred: {row['Predicted_Class']})", axis=1)
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(df[['Prob_Class_1', 'Prob_Class_2', 'Prob_Class_3']],
                          cmap='Blues', annot=True, fmt=".2f", cbar_kws={'label': '% probability'})
    # Set the row labels
    plt.yticks(ticks=np.arange(0.5, len(df.index), 1), labels=df['Labels'], rotation=0, fontsize=10)

    for ytick, color in zip(plt.gca().get_yticklabels(), df['Color_Code']):
        ytick.set_color(color)

    plt.title('Probability Heatmap with Prediction Classes')
    plt.xlabel('Probability Classes')
    plt.ylabel('Samples with Predictions')

    plt.show()

def plot_enhanced_confusion_matrix(cm, classes, precision, recall, accuracy, figsize=(10, 8)):
    # Prepare the confusion matrix data
    cm_percent = cm / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm).astype(str)
    
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_percent[i, j]
            if i == j:
                s = cm.sum(axis=1)[i]
                annot[i, j] = '%d\n(%.1f%%)' % (c, p)
            else:
                annot[i, j] = '%d\n(%.1f%%)' % (c, p)
    
    cm = pd.DataFrame(cm, index=classes, columns=classes)
    cm.index.name = 'True'
    cm.columns.name = 'Predicted'
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap='viridis', cbar=False, square=True)
    
    # Annotations
    for i in range(nrows):
        ax.text(nrows, i, f'{precision[i]*100:.1f}%', va='center', ha='center', backgroundcolor='yellow')
        ax.text(i, nrows, f'{recall[i]*100:.1f}%', va='center', ha='center', backgroundcolor='pink')
    
    ax.text(nrows, nrows, f'{accuracy*100:.1f}%', va='center', ha='center', backgroundcolor='aqua')
    
    # Axis adjustments
    ax.set_xticklabels(list(classes) + ['Precision'])
    ax.set_yticklabels(list(classes) + ['Recall'])
    plt.show()



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
        self.n_splits = n_splits
        self.metrics = metrics if metrics is not None else ['r2', 'neg_mean_absolute_error']
        self.return_coefficients = return_coefficients
        self.model = LinearRegression()
        
        if csv_filepaths:
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


    def process_features_csv(self, csv_filepath, output_name):
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
   

    # def calculate_q2_and_mae(self, X, y):
    #     """
    #     Calculate Q² cross-validation and MAE for the model.

    #     Args:
    #     X (np.ndarray): Feature matrix.
    #     y (np.ndarray): Target vector.

    #     Returns:
    #     tuple: Q² cross-validation score and MAE.
    #     """
    #     kf = KFold(n_splits=X.shape[0], shuffle=True, random_state=42)
    #     y_pred = cross_val_predict(self.model, X, y, cv=kf)
        
    #     q2 = r2_score(y, y_pred)
    #     mae = mean_absolute_error(y, y_pred)
        
    #     return q2, mae

    def calculate_q2_and_mae(self, X, y, n_splits=5):
        """
        Calculate Q² cross-validation and MAE for the model using manual splitting.

        Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        n_splits (int): Number of splits for cross-validation.

        Returns:
        tuple: Q² cross-validation score and MAE.
        """
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

    def cross_validate(self, X, y, n_splits=5):
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



    def fit_and_evaluate_combinations(self, n_jobs=-1):

        # results = Parallel(n_jobs=n_jobs)(delayed(fit_and_evaluate_single_combination)(self, combination) for combination in self.features_combinations)
        results=[fit_and_evaluate_single_combination_regression(self, combination) for combination in tqdm(self.features_combinations, desc='Calculating combinations')]
        # results=[fit_and_evaluate_single_combination(self, combination) for combination in self.features_combinations]
        results_eval = [result[0] for result in results]
        
        sorted_results = sorted(results_eval, key=lambda x: x['scores'].get('Q2', float('-inf')), reverse=True)
        print_models_classification_table(sorted_results)
        return sorted_results

class ClassificationModel:
    def __init__(self, csv_filepaths, process_method='one csv', output_name='class', leave_out=None, min_features_num=2, max_features_num=None, n_splits=5, metrics=None, return_coefficients=False):
        self.csv_filepaths = csv_filepaths
        self.process_method = process_method
        self.output_name = output_name
        self.leave_out = leave_out
        self.min_features_num = min_features_num
        self.max_features_num = max_features_num
        self.n_splits = n_splits
        self.metrics = metrics if metrics is not None else ['accuracy','precision','recall' ,'f1', 'roc_auc']
        self.return_coefficients = return_coefficients
        
        if csv_filepaths:
      
            if process_method == 'one csv':
                self.process_features_csv(csv_filepaths.get('features_csv_filepath'),  output_name=output_name)
            elif process_method == 'two csvs':
                self.process_features_csv(csv_filepaths.get('features_csv_filepath'))
                self.process_target_csv(csv_filepaths.get('target_csv_filepath'))
            self.leave_out_samples(leave_out)
            self.determine_number_of_features(min_features_num, max_features_num)
            self.get_feature_combinations()
            self.scaler = StandardScaler()
            
            self.features_df = pd.DataFrame(self.scaler.fit_transform(self.features_df), columns=self.features_df.columns)
            print(self.features_df)

        self.model = LogisticRegression(solver='lbfgs', random_state=42)


    def get_feature_combinations(self):
        self.features_combinations = list(get_feature_combinations(self.features_list, self.min_features_num, self.max_features_num))


    def determine_number_of_features(self, min_features_num=2, max_features_num=4):
        total_features_num = len(self.features_list)
        self.min_features_num = min_features_num
        self.max_features_num = set_max_features_limit(total_features_num, max_features_num)


    def leave_out_samples(self, leave_out=None):
        self.features_df = self.features_df.drop(index=leave_out) if leave_out else self.features_df

        
    def process_features_csv(self, csv_filepath, output_name):
        df = pd.read_csv(csv_filepath)
        self.features_df = df.drop(columns=['Unnamed: 0'])
        self.target_vector = df[output_name]
        self.features_df= self.features_df.drop(columns=[output_name])
       
        self.features_list = self.features_df.columns.tolist()
        self.molecule_names = df.index.tolist()

    def process_target_csv(self, csv_filepath):
        target_vector_unordered = pd.read_csv(csv_filepath)[self.output_name]
        self.target_vector = target_vector_unordered.loc[self.molecule_names]

    def fit(self, X, y):
        # Train the classifier
        self.model.fit(X, y)

    def predict(self, X):
        # Make predictions using the classifier
        return self.model.predict(X)

    def evaluate(self, X, y):
        # Evaluate the classifier using different metrics
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred,average='weighted', zero_division=0)
        recall = recall_score(y, y_pred,average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred,average='weighted', zero_division=0)
        # auc = roc_auc_score(y, self.model.predict_proba(X)[:, 1])
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
            #'auc': auc
        }
        return results

    def cross_validation(self, X, y, n_splits=5):
        # Perform k-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Define custom scorers
        f1_scorer = make_scorer(f1_score, average='weighted', zero_division=0)  # Choose 'macro', 'micro', or 'weighted'
        roc_auc_scorer = make_scorer(roc_auc_score, multi_class='ovr', average='weighted', zero_division=0)  # Choose 'ovr' or 'ovo', and averaging method

        # Perform cross-validation
        accuracy = cross_val_score(self.model, X, y, cv=kf, scoring='accuracy')
        f1 = cross_val_score(self.model, X, y, cv=kf, scoring=f1_scorer)
        # auc = cross_val_score(self.model, X, y, cv=kf, scoring=roc_auc_scorer)
        
        # Calculate mean values
        avg_accuracy = np.mean(accuracy)
        avg_f1 = np.mean(f1)
        # avg_auc = np.mean(auc)
        
        return avg_accuracy, avg_f1 #, avg_auc


    # def fit_and_evaluate_single_combination(self, combination, accuracy_threshold=0.7, return_probabilities=False):
    #     selected_features = self.features_df[list(combination)]
    #     X = selected_features.to_numpy()
    #     y = self.target_vector.to_numpy()

    #     # Fit the model
    #     self.fit(X, y)

    #     # Evaluate the model
    #     evaluation_results = self.evaluate(X, y)
      
    #     # Check if accuracy is above the threshold
    #     if evaluation_results['accuracy'] > accuracy_threshold:
    #         avg_accuracy, avg_f1 = self.cross_validation(X, y) ## , avg_auc
    #         evaluation_results['avg_accuracy'] = avg_accuracy
    #         evaluation_results['avg_f1_score'] = avg_f1
    #         # evaluation_results['avg_auc'] = avg_auc

    #     results={
    #         'combination': combination,
    #         'scores': evaluation_results,
    #         'models': self.model
    #     }

    #     if return_probabilities:

    #         probabilities = self.model.predict_proba(X)
    #         # Creating a DataFrame for probabilities
    #         prob_df = pd.DataFrame(probabilities, columns=[f'Prob_Class_{i}' for i in range(probabilities.shape[1])])
    #         prob_df['Predicted_Class'] = self.model.predict(X)
    #         prob_df['True_Class'] = y

    #         return results, prob_df

    #     return results
    
    def fit_and_evaluate_combinations(self, n_jobs=-1):
        # Generate combinations of features
        features_list = self.features_df.columns.tolist()
        min_features_num = 2
        max_features_num = len(features_list) // 5
        feature_combinations = list(get_feature_combinations(features_list, min_features_num, max_features_num))

        # Evaluate each combination
        results=[fit_and_evaluate_single_combination_classification(self,combination) for combination in tqdm(feature_combinations, desc='Calculating combinations')]
        # print('results',results)
       
        sorted_results = sorted(results, key=lambda x: x['scores'].get('avg_accuracy', 0), reverse=True)
        print_models_classification_table(sorted_results)

        return sorted_results
    


import os
# Usage





import time
if __name__ == "__main__":
    
    os.chdir(r'C:\Users\edens\Documents\GitHub\LabCode\MolFeatures\modeling_example')
    csv_filepaths = {
    'features_csv_filepath': 'Logistic_Dataset_Example.csv',
    'target_csv_filepath': ''
    }

    model=ClassificationModel(csv_filepaths)
    start_time = time.time()
    results = model.fit_and_evaluate_combinations(n_jobs=-1)
    end_time = time.time()
    # print(results)
    print_models_classification_table(results)


#     model = LinearRegressionModel(
#     csv_filepaths=csv_filepaths,
#     process_method='one csv',
#     output_name='output',
#     leave_out=None,
#     min_features_num=2,
#     max_features_num=4,
#     n_splits=5,
#     return_coefficients=True
# )
    
#     # cProfile.runctx(model.fit_and_evaluate_combinations(n_jobs=-1), globals(), locals(), filename='profile.prof')
#     # stats=pstats.Stats('profile.prof')
#     # stats.strip_dirs().sort_stats('cumulative').print_stats(50)
    
#     start_time = time.time()
#     results = model.fit_and_evaluate_combinations(n_jobs=-1)
#     end_time = time.time()

#     elapsed_time = end_time - start_time
#     print(f"Elapsed time: {elapsed_time:.2f} seconds")
    


#     print_models_table(results)




