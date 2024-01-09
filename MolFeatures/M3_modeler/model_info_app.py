import tkinter as tk
from tkinter import simpledialog, messagebox
from .single_model_processing import StrConstants, DfColumns, get_features_combination_name, generate_train_predictions_df
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import os

def show_results(message):
    # Method to display results in the Tkinter application
    tk.messagebox.showinfo("Results", message)       


def generate_metrics_data_arrays(features_combination, model_obj):
    _, fold3, fold3_coefficients=model_obj.get_cross_validation_results(features_combination, n_splits=3, return_coefficients=True)
    _, fold5, fold5_coefficients=model_obj.get_cross_validation_results(features_combination, n_splits=5, return_coefficients=True)
    _, loo, loo_coefficients=model_obj.get_cross_validation_results(features_combination, n_splits=len(model_obj.features_df.index), return_coefficients=True)
    metrics_data_array={'fold3': fold3, 'fold5': fold5, 'loo': loo}
    coefficients_data_array={'fold3': fold3_coefficients, 'fold5': fold5_coefficients, 'loo': loo_coefficients}
    return metrics_data_array, coefficients_data_array

def generate_metrics_df_list(metrics_data_arrays, coefficients_data_arrays):
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
    metrics_df_list=[]
    metrics_df=pd.concat([pd.DataFrame(metrics_data_array) for metrics_data_array in metrics_data_arrays.values()])
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



class ModelInfoTkinter():

    def __init__(self, parent, model_obj, output_dir=''):
        self.parent = parent
        self.model_obj = model_obj
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
        metrics_data_array, coefficients_data_array=generate_metrics_data_arrays(self.features_combination, self.model_obj)
        self.metrics_df_list=generate_metrics_df_list(metrics_data_array, coefficients_data_array)
        self.model_obj.metrics_df_list=self.metrics_df_list #run around

    

    def present_model(self):
        show_results(str(self.models_df))
        indices=self.models_df.index.to_numpy()
        model_num_to_present = simpledialog.askstring("Input", f"Pick model number to present (cancel to exit):\n Choose from: {indices}", parent=self.parent)
        if model_num_to_present is None or not model_num_to_present.isnumeric():
            return

        model_num_to_present = int(model_num_to_present)
        if model_num_to_present in self.models_df.index:
            self.preprocess_report(model_num_to_present)
            # Assuming print_report function is adapted to return a string instead of printing
            report = print_report(self.model_obj, self.features_combination)
            show_results(report)

            save_model = messagebox.askyesno("Save", "Save results to output directory?", parent=self.parent)
            if save_model:
                # Assuming save_single_model_report is adapted for Tkinter
                save_single_model_report('model_name', self.output_dir, self.model_obj, self.features_combination)
        else:
            show_results('Input number not valid')
