import pandas as pd
import numpy as np
from enum import Enum
from typing import List, Dict, Any, Union


class Names(Enum):
    DIPOLE_COLUMNS = ['dip_x', 'dip_y', 'dip_z', 'total_dipole']
    STANDARD_ORIENTATION_COLUMNS = ['atom', 'x', 'y', 'z']
    DF_LIST = ['standard_orientation_df', 'dipole_df', 'pol_df', 'info_df']


def df_list_to_dict(df_list: List[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Maps list of DataFrames to names specified in Names.DF_LIST."""
    return {name: df for name, df in zip(Names.DF_LIST.value, df_list)}


def split_to_vib_dict(vectors: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Splits vibration vector columns into separate NumPy arrays per atom."""
    num_columns = vectors.shape[1]
    vib_dict = {}
    for i in range(num_columns // 3):
        key = f'vibration_atom_{i + 1}'
        vib_dict[key] = vectors.iloc[:, i*3:(i+1)*3].dropna().to_numpy(dtype=float)
    return vib_dict


def feather_file_handler(feather_file: str) -> List[Any]:
    """
    Reads a feather file with molecular data, processes, and returns:
    [df_dict, vib_dict, charge_dict, ev_df]
    """
    data = pd.read_feather(feather_file)
    data.columns = range(len(data.columns))

    # Extract core DataFrames
    xyz = data.iloc[:, 0:4].dropna()
    try:
        xyz.columns = Names.STANDARD_ORIENTATION_COLUMNS.value
        xyz = xyz.reset_index(drop=True)
        xyz[['x', 'y', 'z']] = xyz[['x', 'y', 'z']].astype(float)
    except Exception:
        # fallback: sometimes header is off by a few rows
        xyz = xyz.iloc[2:].reset_index(drop=True)
        xyz.columns = Names.STANDARD_ORIENTATION_COLUMNS.value
        xyz[['x', 'y', 'z']] = xyz[['x', 'y', 'z']].astype(float)
    xyz = xyz.dropna()
    # Extract energy value as single-row DataFrame
    ev = pd.DataFrame({'energy': [float(data.iloc[1, 1])]})
    # Extract charge DataFrames
    nbo_charge_df = data.iloc[:, 10:11].dropna().rename(columns={10: 'charge'}).astype(float).reset_index(drop=True)
    hirshfeld_charge_df = data.iloc[:, 11:12].dropna().rename(columns={11: 'charge'}).astype(float).reset_index(drop=True)
    cm5_charge_df = data.iloc[:, 12:13].dropna().rename(columns={12: 'charge'}).astype(float).reset_index(drop=True)

    # Ensure charge DataFrames align with xyz atoms
    for cdf in [nbo_charge_df, hirshfeld_charge_df, cm5_charge_df]:
        if len(cdf) != len(xyz):
            cdf = cdf[cdf.index.isin(xyz.index)].reset_index(drop=True)

    charge_dict = {'nbo': nbo_charge_df, 'hirshfeld': hirshfeld_charge_df, 'cm5': cm5_charge_df}

    # Dipole, polarizability, info, vibration vectors
    dipole_df = data.iloc[:, 4:8].dropna().rename(
        columns={4: 'dip_x', 5: 'dip_y', 6: 'dip_z', 7: 'total_dipole'}
    ).astype(float).reset_index(drop=True)

    pol_df = data.iloc[:, 8:10].dropna().rename(
        columns={8: 'aniso', 9: 'iso'}
    ).astype(float).reset_index(drop=True)

    info_df = data.iloc[:, 13:15].dropna().rename(
        columns={13: 'Frequency', 14: 'IR'}
    ).astype(float).reset_index(drop=True)

    # Vibration vectors as dict of numpy arrays
    vectors = data.iloc[:, 15:].dropna()
    vib_dict = split_to_vib_dict(vectors)

    # Main DataFrames as dict
    df_list = [xyz, dipole_df, pol_df, info_df]
    df_list = [df for df in df_list if not df.empty and df.dropna(how='all').shape[0] > 0]
    df_dict = df_list_to_dict(df_list)

    return [df_dict, vib_dict, charge_dict, ev]

