import pandas as pd
import numpy as np
import os
from enum import Enum
from typing import List, Optional
import pandas as pd


class Names(Enum):
    
    DIPOLE_COLUMNS=['dip_x','dip_y','dip_z','total_dipole']
    STANDARD_ORIENTATION_COLUMNS=['atom','x','y','z']
    DF_LIST=['standard_orientation_df', 'dipole_df', 'pol_df', 'info_df']




def df_list_to_dict(df_list):
    my_dict={}
    for name,df in zip(Names.DF_LIST.value,df_list):
        my_dict[name]=df
    return my_dict



def feather_file_handler(feather_file):
    # Read the feather file
    
    data = pd.read_feather(feather_file)
    
    data.columns=range(len(data.columns))
    ## change pandas options to display all columns
  
    xyz = data.iloc[:, 0:4].dropna()
    ## check if xyz rows were drop , if not drop the first 2
    
    # if not xyz.iloc[:2].isnull().any(axis=None):
    # # Drop the first two rows if they contain values
    #     xyz = xyz.iloc[2:]

    
    dipole_df = data.iloc[:, 4:8].dropna()
    pol_df = data.iloc[:, 8:10].dropna()
    nbo_charge_df = data.iloc[:, 10:11].dropna()
    ## if the whole column is NaN, it will be removed
    hirshfeld_charge_df = data.iloc[:,11:12].dropna().reset_index(drop=True)
    cm5_charge_df = data.iloc[:,12:13].dropna().reset_index(drop=True)
    info_df = data.iloc[:, 13:15].dropna()
    vectors = data.iloc[:, 15:].dropna()
    

    
        
    xyz.rename(columns={xyz.columns[0]: 'atom', xyz.columns[1]: 'x', xyz.columns[2]: 'y', xyz.columns[3]: 'z'}, inplace=True)
        # Remove the first two rows if they dont contain NaN values
    # if not xyz.iloc[:2].isnull().any(axis=None):
    # xyz = xyz.iloc[2:]
    
    xyz = xyz.reset_index(drop=True)

    xyz[['x', 'y', 'z']] = xyz[['x', 'y', 'z']].astype(float)
 
    xyz=xyz.dropna()
    # Calculate the length of the DataFrame
    dipole_df.rename(columns={dipole_df.columns[0]: 'dip_x', dipole_df.columns[1]: 'dip_y', dipole_df.columns[2]: 'dip_z', dipole_df.columns[3]: 'total_dipole'}, inplace=True)
    dipole_df=dipole_df.astype(float)
    dipole_df=dipole_df.dropna()
    dipole_df=dipole_df.reset_index(drop=True)
    nbo_charge_df.rename(columns={nbo_charge_df.columns[0]: 'charge'}, inplace=True)
    nbo_charge_df=nbo_charge_df.astype(float)
    nbo_charge_df=nbo_charge_df.dropna()
    nbo_charge_df=nbo_charge_df.reset_index(drop=True)

    charge_dict={'nbo':nbo_charge_df, 'hirshfeld':hirshfeld_charge_df, 'cm5':cm5_charge_df}


    pol_df.rename(columns={pol_df.columns[0]: 'aniso', pol_df.columns[1]: 'iso'}, inplace=True)
    pol_df=pol_df.astype(float)
    pol_df=pol_df.dropna()
    pol_df=pol_df.reset_index(drop=True)
    info_df.rename(columns={info_df.columns[0]: 'Frequency', info_df.columns[1]: 'IR'}, inplace=True)
    info_df=info_df.astype(float)
    info_df=info_df.dropna()
    info_df=info_df.reset_index(drop=True)
    
    def split_to_dict(dataframe):
        
        num_columns = dataframe.shape[1]
        num_dfs = num_columns // 3  # Integer division to get the number of 3-column dataframes

        dfs_dict = {}
        for i in range(num_dfs):
            start_col = i * 3
            end_col = start_col + 3
            key = f'vibration_atom_{i + 1}'
            dfs_dict[key] = np.array(dataframe.iloc[:, start_col:end_col].values.astype(float))  # Storing as a NumPy array

        return dfs_dict
    
    vib_dict=split_to_dict(vectors)
    df_list=[xyz ,dipole_df, pol_df, info_df]
    df_list=[df.dropna(how='all') for df in df_list if not df.empty]
    df_dict=df_list_to_dict(df_list)
    dict_list=[df_dict,vib_dict,charge_dict]

    return dict_list
