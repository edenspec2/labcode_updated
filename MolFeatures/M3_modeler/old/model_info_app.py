import tkinter as tk
from tkinter import simpledialog, messagebox
from .single_model_processing import StrConstants, DfColumns, get_features_combination_name, generate_train_predictions_df
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def show_results(message):
    # Method to display results in the Tkinter application
    tk.messagebox.showinfo("Results", message)       


# def generate_metrics_data_arrays(features_combination, model_obj):
#     _, fold3, fold3_coefficients=model_obj.get_cross_validation_results(features_combination, n_splits=3, return_coefficients=True)
#     _, fold5, fold5_coefficients=model_obj.get_cross_validation_results(features_combination, n_splits=5, return_coefficients=True)
#     if model_obj.mode=='linear_regression':
#         _, loo, loo_coefficients=model_obj.get_cross_validation_results(features_combination, n_splits=len(model_obj.features_df.index), return_coefficients=True)
#     elif model_obj.mode=='classification':
#         num_classes=len(model_obj.target_vector.unique())
#         n_splits=math.floor(len(model_obj.features_df.index)/num_classes)
#         _, loo, loo_coefficients=model_obj.get_cross_validation_results(features_combination, n_splits=n_splits, return_coefficients=True)
#     metrics_data_array={'fold3': fold3, 'fold5': fold5, 'loo': loo}
#     coefficients_data_array={'fold3': fold3_coefficients, 'fold5': fold5_coefficients, 'loo': loo_coefficients}
#     return metrics_data_array, coefficients_data_array

# def generate_metrics_data_arrays(features_combination, model_obj):
#     """
#     Generate arrays of metrics and coefficients for different cross-validation splits.

#     Args:
#         features_combination (list): List of features to include in the model.
#         model_obj (object): Model object with necessary methods and attributes.

#     Returns:
#         tuple: Dictionaries containing metrics and coefficients for 3-fold, 5-fold, and LOO cross-validation.
#     """
#     # Initialize dictionaries to store metrics and coefficients
#     metrics_data_array = {}
#     coefficients_data_array = {}

#     # Define cross-validation splits
#     splits = {'fold3': 3, 'fold5': 5}

#     # Progress tracking with tqdm
#     for split_name, n_splits in tqdm(splits.items(), desc="Cross-Validation Splits"):
#         _, scores, coefficients = model_obj.get_cross_validation_results(features_combination, n_splits=n_splits, return_coefficients=True)
#         metrics_data_array[split_name] = scores
#         coefficients_data_array[split_name] = coefficients

#     # Handle Leave-One-Out cross-validation
#     if model_obj.mode == 'linear_regression':
#         n_splits = len(model_obj.features_df.index)
#     elif model_obj.mode == 'classification':
#         num_classes = len(model_obj.target_vector.unique())
#         n_splits = math.floor(len(model_obj.features_df.index) / num_classes)
#     else:
#         raise ValueError("Invalid mode. Mode should be either 'linear_regression' or 'classification'")

#     _, loo_scores, loo_coefficients = model_obj.get_cross_validation_results(features_combination, n_splits=n_splits, return_coefficients=True)
#     metrics_data_array['loo'] = loo_scores
#     coefficients_data_array['loo'] = loo_coefficients

#     return metrics_data_array, coefficients_data_array

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd

# Modify the cross-validation function to use joblib for parallel processing
def get_cross_validation_results_parallel(features_combination, model_obj, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(model_obj.get_cross_validation_results)(
            features_combination=features_combination,
            n_splits=n_splits,
            return_coefficients=True
        )
        for n_splits in tqdm([3, 5, len(model_obj.features_df.index)], desc="Cross-Validation Splits")
    )

    fold3, fold3_coefficients = results[0][1], results[0][2]
    fold5, fold5_coefficients = results[1][1], results[1][2]
    loo, loo_coefficients = results[2][1], results[2][2]

    metrics_data_array = {'fold3': fold3, 'fold5': fold5, 'loo': loo}
    coefficients_data_array = {'fold3': fold3_coefficients, 'fold5': fold5_coefficients, 'loo': loo_coefficients}

    return metrics_data_array, coefficients_data_array

def generate_metrics_df_list(metrics_data_arrays, coefficients_data_arrays, mode='linear_regression'):
    # Generate tables (dataframes) for final reports
    # input:
    # fold3, fold5, loo (lists of floats): list of R2, Q2 and MAE values 
    # (respectively) for the diffrent kfold cross-validation procedurs.\
    # coef (lists of floats): list of model coefficients (length is the number
    # of included features + 1)
    # feat_comb (list of str): list of features to include in the model
    # print_results (bool): indicate weather to print values to terminal
    # output:
    # fold3, fold5, loo and coef tables as dataframes  
    if mode=='linear_regression':
        print(f"metrics_data_arrays: {metrics_data_arrays}, coefficients_data_arrays: {coefficients_data_arrays}")
        metrics_df_list=[]
        # metrics_df=pd.concat([pd.DataFrame(metrics_data_array) for metrics_data_array in metrics_data_arrays.values()])
        metrics_df = pd.concat([pd.DataFrame([metrics_data_array], index=[key]) for key, metrics_data_array in metrics_data_arrays.items()])
        coefficients_df=pd.concat([pd.DataFrame(coefficients_data_array, index=['coeff_'+str(number) for number in range(len(coefficients_data_array))])
        for coefficients_data_array in coefficients_data_arrays.values()], axis=1)
        coefficients_df.columns=coefficients_data_arrays.keys()
        metrics_df_list.append(coefficients_df) 
        metrics_df_list.append(metrics_df)
        return metrics_df_list
    elif mode=='classification':
        metrics_df_list=[]
        # metrics_df=pd.concat([pd.DataFrame(metrics_data_array) for metrics_data_array in metrics_data_arrays.values()])
        metrics_df = pd.concat([pd.DataFrame([metrics_data_array], index=[key]) for key, metrics_data_array in metrics_data_arrays.items()])
        coefficients_df=pd.concat([pd.DataFrame(coefficients_data_array, index=['coeff_'+str(number) for number in range(len(coefficients_data_array))])
        for coefficients_data_array in coefficients_data_arrays.values()], axis=1)
        coefficients_df.columns=coefficients_data_arrays.keys()
        metrics_df_list.append(coefficients_df) 
        metrics_df_list.append(metrics_df)
        return metrics_df_list

def set_q2_plot_settings(ax, lower_bound, upper_bound, fontsize=15):
    bounds_array = np.array([lower_bound, upper_bound])
    ax.plot(bounds_array, bounds_array, 'k--', linewidth=2)  # black dashed line
    ax.set_xlabel('Measured', fontsize=fontsize)  # Assuming 'Measured' is the label you want
    ax.set_ylabel('Predicted', fontsize=fontsize)
    ax.set_ylim(bounds_array)
    ax.set_xlim(bounds_array)
    ax.grid(True)  # Adding a grid

def plot_accuracy_and_proba(models_obj, features_combination):
    # Determine the color based on prediction correctness and closeness
    targets, predictions, probabilities ,names, accuracy = models_obj.get_train_predictions(features_combination)
    # print(f"targets: {targets}, predictions: {predictions}, probabilities: {probabilities}, names: {names}, accuracy: {accuracy} c")
    probabilities = np.array(probabilities)
    class_names = models_obj.target_vector.unique()
    # class_names = [f'Prob_Class_{i}' for i in range((num_classes))]
    # print(f'class_names: {class_names}')
    colors = []
    for i, (target, pred, prob) in enumerate(zip(targets, predictions, probabilities)):
        # print(f'target: {target}, pred: {pred}, prob: {prob[0]} ')
        print(f'targer: {target}, pred: {pred}')
        if target == pred:
            
            # Correct prediction
            colors.append('green')
        else:
            # Incorrect prediction
            # Sort the probabilities in descending order and get the indices
            sorted_prob_indices = sorted(range(len(prob)), key=lambda k: prob[k], reverse=True)

            # Check if the target is the second highest probability
            if sorted_prob_indices[1] == target:
                # If the second highest probability is the correct class, mark as orange
                colors.append('yellow')
            else:
                # Otherwise, mark as red
                colors.append('red')

    formatted_annotations=[]
    # print(probabilities)
    for i in range(len(probabilities)):
        print(f'target: {targets[i]}, pred: {predictions[i]}')
        formatted_numbers = ' '.join(f"{num:.2%}" for num in probabilities[i])
        annotations = [["                                                           True: {}  Predicted: {}  Probabilities: {}".format(
            targets[i],  # No adjustment if tar is 0-based
            predictions[i],  # No adjustment if pred is 0-based
            formatted_numbers  # Assuming probabilities are aligned with 0-based indices
            )]]
        
        # annotations = [["                                                           True: {}  Predicted: {}  Probabilities: {}".format(
             
        #     tar,  # No adjustment if tar is 0-based
        #     pred,  # No adjustment if pred is 0-based
        #     formatted_numbers  # Assuming probabilities are aligned with 0-based indices
        #     ) for i, (tar, pred) in enumerate(zip(targets, predictions))]]
        formatted_annotations.append(annotations)
    

    # Reshape the annotations to match the shape of df_probabilities
    
    colors = np.array(colors).reshape(-1, 1)

    # Create a DataFrame for the probabilities
    df_probabilities = pd.DataFrame(probabilities, index=names, columns=class_names)

    #2
    annotations = np.array(formatted_annotations).reshape(df_probabilities.shape[0], -1)
    
    ##@
    flat_annotations = [item[0] for item in annotations]
    # Create an array with the same shape as df_probabilities, filled with empty strings
    reshaped_annotations = np.full(df_probabilities.shape, '', dtype=object)
    # Fill the first column of reshaped_annotations with flat_annotations
    for i, annotation in enumerate(flat_annotations):
        reshaped_annotations[i][0] = annotation
    
    annot_kws = {"size": 10, "weight": "bold", "color": "black"}
    fig, ax = plt.subplots(figsize=(10, len(predictions)))
    sns.heatmap(df_probabilities, annot=reshaped_annotations, fmt='', cmap='coolwarm', cbar=True, linecolor='black', 
                linewidths=0.1,annot_kws=annot_kws, yticklabels=1, ax=ax)

    # Apply the colors to each cell based on the prediction correctness
    # for y in range(len(predictions)):
    #     for x in range(len(class_names)):
    #         ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor=colors[y, 0], lw=3))

    for y in range(df_probabilities.shape[0]):
        for x in range(df_probabilities.shape[1]):
            color = colors[y][0]
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=True, color=color, lw=0))

    accuracy=accuracy['accuracy']
    plt.title(f'Classification Predictions Heatmap\nModel Accuracy: {accuracy:.2%}')
    plt.ylabel('Sample Name')
    plt.xlabel('Class')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)  # Rotate y-axis labels to horizontal
    plt.tight_layout()
    plt.show()
    return reshaped_annotations

def generate_q2_scatter_plot(models_obj, features_combination, figsize=(10, 10), fontsize=12, scatter_color='black'):
    # Add print statements before the call
    print(f"Features combination length: {len(features_combination)}")
    print(f"Features combination: {features_combination}")
    show_results(f"Features combination length: {len(features_combination)}\nFeatures combination: {features_combination}")

    y, y_hat, labels = models_obj.get_train_predictions(features_combination)

    # Add print statements after the call
    print(f"After get_train_predictions - Length of y: {len(y)}")
    print(f"After get_train_predictions - Length of y_hat: {len(y_hat)}")
    print(f"After get_train_predictions - Length of labels: {len(labels)}")
    show_results(f"After get_train_predictions - Length of y: {len(y)}\nAfter get_train_predictions - Length of y_hat: {len(y_hat)}\nAfter get_train_predictions - Length of labels: {len(labels)}")

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(y, y_hat, c=scatter_color)
    
    # Add annotations with an offset to avoid overlapping with the point.
    for i, label in enumerate(labels):
        ax.annotate(label, (y[i], y_hat[i]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=fontsize)
    
    set_q2_plot_settings(ax, lower_bound=0, upper_bound=max(y)*1.1, fontsize=fontsize)
    return y, y_hat,labels, fig, ax


def get_model_name(models_df, model_num):
    return models_df.index[model_num - 1]

def print_report(models_obj, features_combination):
    for metrics_df in models_obj.metrics_df_list:
        print(f"{metrics_df.index.to_numpy()[0]} \n {metrics_df} \n \n \n") #Make sure this works properly
        show_results(f"{metrics_df.index.to_numpy()[0]} \n {metrics_df} \n \n \n")
#    print (f"model coefficients \n {coef_table}\n \n")  
    _= generate_q2_scatter_plot(models_obj, features_combination)
    plt.show()

def create_xls_writer(output_dir, file_name, engine='xlswriter'):
    writer=pd.ExcelWriter(os.path.join(output_dir, file_name), engine='xlsxwriter')            
    return writer



def save_single_model_report(model_name, output_dir, models_obj, features_combination):
    # Ensure output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Full path for the Excel file
    xls_file_path = os.path.join(output_dir, f'{model_name}.xlsx')

    # Full path for the image file
    png_img_path = os.path.join(output_dir, f'{model_name}.png')

    # Generate the plot
    y, y_hat, labels, fig, ax = generate_q2_scatter_plot(models_obj, features_combination)
    fig.canvas.draw()
    plot_img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plot_img.save(png_img_path)

    # Create a single Excel writer
    with pd.ExcelWriter(xls_file_path, engine='xlsxwriter') as writer:
        start_row = 1
        for metrics_df in models_obj.metrics_df_list:
            metrics_df.to_excel(writer, sheet_name='Model coefficients', startrow=start_row, startcol=0)
            start_row += len(metrics_df) + 2  # Adjust the start row for the next DataFrame

        plot_table = generate_train_predictions_df(y, y_hat, labels)
        # plot_table.to_excel(writer, sheet_name='Predictions', startrow=0, startcol=0)


def save_single_model_report_classification(model_name, output_dir, models_obj, features_combination):
    # Ensure output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Full path for the Excel file
    xls_file_path = os.path.join(output_dir, f'{model_name}.xlsx')

    # Full path for the image file
    png_img_path = os.path.join(output_dir, f'{model_name}.png')

    # Generate the plot
    annotation = plot_accuracy_and_proba(models_obj, features_combination)

    # Create a single Excel writer
    with pd.ExcelWriter(xls_file_path, engine='xlsxwriter') as writer:
        # Save annotation to a sheet
        annotation.to_excel(writer, sheet_name='Annotations')

        # Convert features_combination to a DataFrame and save it to another sheet
        if isinstance(features_combination, list):
            # Assuming features_combination is a list. Modify as needed for other data structures
            df_features = pd.DataFrame({'Features': features_combination})
            df_features.to_excel(writer, sheet_name='Feature Combination', index=False)

        # Save the Excel file
        writer.save()

        

def print_report_class(models_obj, features_combination):
    for metrics_df in models_obj.metrics_df_list:
        print(f"{metrics_df.index.to_numpy()[0]} \n {metrics_df} \n \n \n") #Make sure this works properly
        show_results(f"{metrics_df.index.to_numpy()[0]} \n {metrics_df} \n \n \n")
#    print (f"model coefficients \n {coef_table}\n \n")
    plot_accuracy_and_proba(models_obj, features_combination)
    

class ModelInfoTkinter():

    def __init__(self, parent, model_obj, mode='classification' ,output_dir=''):
        self.parent = parent
        self.model_obj = model_obj
        self.mode=mode
        self.model_obj.test_and_rank_all_subset_models()
        models_df = self.model_obj.models_df.copy().reset_index(drop=True)
        models_df.index += 1
        self.models_df = models_df
        self.output_dir = output_dir

    def preprocess_report(self, model_num):
        # input:
        # model_num (int): number of model to save its report
        # print_results (bool): indicate weather to print values to terminal
        ## use the native loo python implementation
        self.features_combination=get_features_combination_name(self.models_df, model_num)
        print(f"Features combination: {self.features_combination}")
        if self.mode=='linear_regression':
            metrics_data_array, coefficients_data_array=generate_metrics_data_arrays(self.features_combination, self.model_obj)
            self.metrics_df_list=generate_metrics_df_list(metrics_data_array, coefficients_data_array)
            self.model_obj.metrics_df_list=self.metrics_df_list #run around
        elif self.mode=='classification':
            metrics_data_array, coefficients_data_array=generate_metrics_data_arrays(self.features_combination, self.model_obj)
            self.metrics_df_list=generate_metrics_df_list(metrics_data_array, coefficients_data_array)
            self.model_obj.metrics_df_list=self.metrics_df_list

    

    # def present_model(self):
    #     show_results(str(self.models_df))
    #     indices=self.models_df.index.to_numpy()
    #     model_num_to_present = simpledialog.askstring("Input", f"Pick model number to present (cancel to exit):\n Choose from: {indices}", parent=self.parent)
    #     if model_num_to_present is None or not model_num_to_present.isnumeric():
    #         return

    #     model_num_to_present = int(model_num_to_present)
    #     if model_num_to_present in self.models_df.index:
    #         self.preprocess_report(model_num_to_present)
    #         # Assuming print_report function is adapted to return a string instead of printing
    #         report = print_report(self.model_obj, self.features_combination)
    #         show_results(report)

    #         save_model = messagebox.askyesno("Save", "Save results to output directory?", parent=self.parent)
    #         if save_model:
    #             # Assuming save_single_model_report is adapted for Tkinter
    #             save_single_model_report('model_name', self.output_dir, self.model_obj, self.features_combination)
    #     else:
    #         show_results('Input number not valid')

    def present_model(self):
        """
        An interactive method to present models. Allows the user to select a model
        by its number for presentation. Provides an option to save the model report
        to the output directory.

        Attributes:
        - self.models_df (pd.DataFrame): Dataframe containing model information.
        """
        show_results(str(self.models_df))
        indices=self.models_df.index.to_numpy()
        model_num_to_present = simpledialog.askstring("Input", f"Pick model number to present (cancel to exit):\n Choose from: {indices}", parent=self.parent)
        model_num_to_present = int(model_num_to_present)
        if model_num_to_present in self.models_df.index:
            print(f"Model number {model_num_to_present} \n")
            self.preprocess_report(model_num_to_present)
            if self.mode=='linear_regression':
                report=print_report(self.model_obj, self.features_combination)
                show_results(report)

            elif self.mode=='classification':
                report=print_report_class(self.model_obj, self.features_combination)
                show_results(report)

        save_model = messagebox.askyesno("Save", "Save results to output directory?", parent=self.parent)
        if save_model:
            # Assuming save_single_model_report is adapted for Tkinter
            if self.mode=='linear_regression':
                save_single_model_report('model_name', self.output_dir, self.model_obj, self.features_combination)
            elif self.mode=='classification':
                save_single_model_report_classification('model_name', self.output_dir, self.model_obj, self.features_combination)
            # save_single_model_report('model_name', self.output_dir, self.model_obj, self.features_combination)
        else:
            show_results('Input number not valid')


        
