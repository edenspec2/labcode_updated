import os
import matplotlib.pyplot as plt
import PIL
import numpy as np
import pandas as pd

from .single_model_processing import StrConstants, DfColumns, get_features_combination_name, generate_train_predictions_df


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

    y, y_hat, labels = models_obj.get_train_predictions(features_combination)

    # Add print statements after the call
    print(f"After get_train_predictions - Length of y: {len(y)}")
    print(f"After get_train_predictions - Length of y_hat: {len(y_hat)}")
    print(f"After get_train_predictions - Length of labels: {len(labels)}")

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
#    print (f"model coefficients \n {coef_table}\n \n")  
    _= generate_q2_scatter_plot(models_obj, features_combination)
    plt.show()

def create_xls_writer(output_dir, file_name, engine='xlswriter'):
    writer=pd.ExcelWriter(os.path.join(output_dir, file_name), engine='xlsxwriter')            
    return writer

import pandas as pd
import os
import PIL.Image

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
       


class Model_info():
    # settings:
    # Model (Model object)
    # output_dir (str): a directory to output produced files   
    def __init__(self, model_obj, output_dir=''):   
        self.model_obj=model_obj
        self.model_obj.test_and_rank_all_subset_models()
        models_df=self.model_obj.models_df
        models_df.index+=1
        self.models_df=models_df
        self.output_dir=output_dir

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
        """
        An interactive method to present models. Allows the user to select a model
        by its number for presentation. Provides an option to save the model report
        to the output directory.

        Attributes:
        - self.models_df (pd.DataFrame): Dataframe containing model information.
        """
        print(self.models_df)
        while True:
            model_num_to_present = input("Pick model number to present (N to exit): ")
            if model_num_to_present == 'N':
                break
            if not model_num_to_present.isnumeric():
                print('Not a number')
                continue
            model_num_to_present = int(model_num_to_present)
            if model_num_to_present in self.models_df.index:
                print(f"Model number {model_num_to_present} \n")
                self.preprocess_report(model_num_to_present)
                print_report(self.model_obj, self.features_combination)  # always prints the report
                while True:
                    save_model = input("Save results to output directory? (Y or N)")
                    if save_model in ['N', 'Y']:
                        break
                if save_model == 'Y':
                    save_single_model_report('model_name',
                                            self.output_dir, self.model_obj, self.features_combination)
            else:
                print('Input number not valid')
                

    def present_model(self):
        """
        An interactive method to present models. Allows the user to select a model
        by its number for presentation. Provides an option to save the model report
        to the output directory.

        Attributes:
        - self.models_df (pd.DataFrame): Dataframe containing model information.
        """
        print(self.models_df)
        while True:
            model_num_to_present = input("Pick model number to present (N to exit): ")
            if model_num_to_present == 'N':
                break
            if not model_num_to_present.isnumeric():
                print('Not a number')
                continue
            model_num_to_present = int(model_num_to_present)
            if model_num_to_present in self.models_df.index:
                print(f"Model number {model_num_to_present} \n")
                self.preprocess_report(model_num_to_present)
                print_report(self.model_obj, self.features_combination)  # always prints the report
                while True:
                    save_model = input("Save results to output directory? (Y or N)")
                    if save_model in ['N', 'Y']:
                        break
                if save_model == 'Y':
                    save_single_model_report('model_name',
                                            self.output_dir, self.model_obj, self.features_combination)
            else:
                print('Input number not valid')
