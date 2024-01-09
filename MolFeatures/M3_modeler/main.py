import warnings
from single_model_processing import Model
from model_info_app import Model_info
import os
import os
import warnings
from typing import Dict

# Assuming the 'Model' and 'Model_info' classes are defined elsewhere in your code

def run_model_in_directory(directory: str, output_csv_filepath: str, min_features_num: int, max_features_num: int, target_csv_filepath= '' ) -> None:
    """
    Runs a model in a specified directory using provided CSV filepaths.

    :param directory: The directory to change to.
    :param csv_filepaths: A dictionary with filepaths for features and target CSV files.
    :param min_features_num: Minimum number of features for the model.
    :param max_features_num: Maximum number of features for the model.
    """
    os.chdir(directory)
    csv_filepaths = {'features_csv_filepath': output_csv_filepath,
                       'target_csv_filepath': target_csv_filepath}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Model(csv_filepaths, min_features_num=min_features_num, max_features_num=max_features_num)  # leave_out=['o_Cl']
        model_info = Model_info(model)

        return model_info
        

if __name__=='__main__':
##    csv_filepath=get_csv_filepath()
    pass
    # os.chdir(r'C:\Users\edens\Documents\GitHub\Automation_code-main\lucas_project\feather_files_new')
    # csv_filepaths = {'features_csv_filepath': 'output_shahar.csv',
    #                    'target_csv_filepath': ''}
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     model=Model(csv_filepaths, min_features_num=2, max_features_num=4) # leave_out=['o_Cl']
    #     model_info=Model_info(model)
    #     model_info.present_model()
    #     #output_example_1.csv