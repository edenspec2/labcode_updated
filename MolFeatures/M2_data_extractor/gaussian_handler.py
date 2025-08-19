import pandas as pd
import numpy as np
from enum import Enum
from typing import List, Dict, Any, Union


class Names(Enum):
    DIPOLE_COLUMNS = ['dip_x', 'dip_y', 'dip_z', 'total_dipole']
    STANDARD_ORIENTATION_COLUMNS = ['atom', 'x', 'y', 'z']
    DF_LIST = ['standard_orientation_df', 'dipole_df', 'pol_df', 'info_df','energy_df']


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
    Uses columns by name instead of index.
    """
    # Define the expected columns (for error-proof selection)
    expected_cols = [
        'atom', 'x', 'y', 'z', 'dip_x', 'dip_y', 'dip_z', 'total_dipole', 'aniso', 'iso',
        'nbo_hirshfeld', 'hirshfeld_charge', 'cm5_charge', 'Frequency', 'IR'
    ]
    try:
        data = pd.read_feather(feather_file)
        # Optionally, fix column names in case they are not matching exactly
        data.columns = [c.strip() for c in data.columns]

        # XYZ
        xyz_cols = ['atom', 'x', 'y', 'z']
        xyz = data[xyz_cols].dropna().reset_index(drop=True)
        xyz[['x', 'y', 'z']] = xyz[['x', 'y', 'z']].astype(float)

        # Energy value (optional—get from a suitable column or from metadata)
        try:
            ev = pd.DataFrame({'energy': [float(data.get('energy', [None])[0])]})
        except Exception as e:
            print(f"Error extracting energy value: {e}")
            ev = pd.DataFrame({'energy': [None]})

        # Charges
        nbo_charge_df = data[['nbo_charge']].dropna().rename(columns={'nbo_charge': 'charge'}).astype(float).reset_index(drop=True)
        hirshfeld_charge_df = data[['hirshfeld_charge']].dropna().rename(columns={'hirshfeld_charge': 'charge'}).astype(float).reset_index(drop=True)
        cm5_charge_df = data[['cm5_charge']].dropna().rename(columns={'cm5_charge': 'charge'}).astype(float).reset_index(drop=True)

        # Align charges with xyz
        for cdf in [nbo_charge_df, hirshfeld_charge_df, cm5_charge_df]:
            if len(cdf) != len(xyz):
                cdf = cdf[cdf.index.isin(xyz.index)].reset_index(drop=True)

        charge_dict = {'nbo': nbo_charge_df, 'hirshfeld': hirshfeld_charge_df, 'cm5': cm5_charge_df}

        # Dipole
        dipole_cols = ['dip_x', 'dip_y', 'dip_z', 'total_dipole']
        dipole_df = data[dipole_cols].dropna().astype(float).reset_index(drop=True)

        # Polarizability
        pol_cols = ['aniso', 'iso']
        pol_df = data[pol_cols].dropna().astype(float).reset_index(drop=True)

        # Info
        info_cols = ['Frequency', 'IR']
        info_df = data[info_cols].dropna().astype(float).reset_index(drop=True)

        # Vibration vectors: everything else after IR, or define your own selection
        vib_start = data.columns.get_loc('IR') + 1
        vectors = data.iloc[:, vib_start:].dropna(axis=1, how='all')
        vib_dict = split_to_vib_dict(vectors)

        # Package DataFrames into a dict (use your own helper)
        df_list = [xyz, dipole_df, pol_df, info_df, ev]
        df_list = [df for df in df_list if not df.empty and df.dropna(how='all').shape[0] > 0]
        df_dict = df_list_to_dict(df_list)

        return [df_dict, vib_dict, charge_dict]
    except Exception as e:
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
        try:
            ev = pd.DataFrame({'energy': [float(data.iloc[1, 1])]})
        except Exception as e:
            print(f"Error extracting energy value: {e}")
            ev = pd.DataFrame({'energy': [None]})

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

