import pandas as pd
import numpy as np
import os
import sys
import math
from enum import Enum
import igraph as ig
from typing import *
import warnings
from scipy.spatial.distance import pdist, squareform
from morfeus import Sterimol
import networkx as nx
import numpy.typing as npt
# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Now you can import from the parent directory
    from gaussian_handler import feather_file_handler
    from utils import visualize
    from utils.help_functions import *
    from extractor_utils.sterimol_utils import *
    from extractor_utils.bond_angle_length_utils import *
    from extractor_utils.dipole_utils import *
    from extractor_utils.vibrations_utils import *
except:
    from .gaussian_handler import feather_file_handler
    from ..utils import visualize
    from ..utils.help_functions import *
    from .extractor_utils.sterimol_utils import *
    from .extractor_utils.bond_angle_length_utils import *
    from .extractor_utils.dipole_utils import *
    from .extractor_utils.vibrations_utils import *

warnings.filterwarnings("ignore", category=RuntimeWarning)




class Molecule:
    def __init__(self, molecule_feather_filename, parameter_list=None, new_xyz_df=None):
        """
        Initialize a Molecule object with structural and computational data.
        
        Args:
            molecule_feather_filename (str): Path to Feather file containing molecular data
            parameter_list (list, optional): Precomputed parameters list. Defaults to None.
            new_xyz_df (DataFrame, optional): Alternative coordinates DataFrame. Defaults to None.
        """
        # Initialize core properties first
        self._initialize_core_properties(molecule_feather_filename, new_xyz_df)
        
        # Handle parameters and file loading
        self._load_parameters(molecule_feather_filename, parameter_list)
        
        # Initialize derived structural properties
        self._initialize_structural_properties()
        
        # Initialize computational results
        self._initialize_computational_data()


    def _initialize_core_properties(self, filename, new_xyz_df):
        """Set basic molecule identification and path properties"""
        self.molecule_name = os.path.splitext(os.path.basename(filename))[0]
        self.molecule_path = os.path.abspath(os.path.dirname(filename))
        self.new_index_order = new_xyz_df.index.tolist() if new_xyz_df is not None else None

    def _load_parameters(self, filename, parameter_list):
        """Handle parameter loading from file or existing list"""
        if parameter_list is None:
            original_dir = os.getcwd()
            try:
                os.chdir(self.molecule_path)
                self.parameter_list = feather_file_handler(filename)
            finally:
                os.chdir(original_dir)
        else:
            self.parameter_list = parameter_list

    def _initialize_structural_properties(self):
        """Set up structural components"""
        # XYZ coordinates
        self.xyz_df = self.parameter_list[0]['standard_orientation_df']
        self.coordinates_array = np.array(self.xyz_df[['x', 'y', 'z']].astype(float))
        
        # Connectivity and atom typing
        self.bonds_df = extract_connectivity(self.xyz_df)
        self.atype_list = nob_atype(self.xyz_df, self.bonds_df)

    def _initialize_computational_data(self):
        """Initialize computed properties from calculations"""
        # Electronic properties
        self.gauss_dipole_df = self.parameter_list[0]['dipole_df']
        self.polarizability_df = self.parameter_list[0]['pol_df']
        
        # Charge and vibrational data
        self.charge_dict = self.parameter_list[2]
        self.vibration_dict = self.parameter_list[1]
        
        # General information
        self.info_df = self.parameter_list[0]['info_df']


    def reindex_vibration_dict(self):
        """
        Reindexes a vibration dictionary based on a new index order.

        Parameters:
        vib_dict (dict): The original vibration dictionary to be reindexed.
        new_index_order (list): A list of integers representing the new atom numbers.

        Returns:
        dict: A new vibration dictionary reindexed according to new_index_order.
        """
        # Building a mapping from old to new indices
        mapping = {i + 1: new_index for i, new_index in enumerate(self.new_index_order)}

        # Creating a new reindexed dictionary
        new_vib_dict = {}
        for old_index, new_index in mapping.items():
            old_key = f'vibration_atom_{old_index}'
            new_key = f'vibration_atom_{new_index}'
            if old_key in self.vibration_dict:
                new_vib_dict[new_key] = self.vibration_dict[old_key]

        return new_vib_dict


    def get_molecule_descriptors_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame with various descriptors calculated for the molecule.

        Returns:
            pd.DataFrame: A DataFrame with the calculated descriptors.
        """
        descriptors_df = pd.concat(
            [self.energy_value, self.gauss_dipole_df, self.get_sterimol()], axis=1)
        descriptors_df = descriptors_df.sort_values(by=['energy_value'])
        return descriptors_df


    def write_xyz_file(self) -> None:
        """
        Writes the molecule's coordinates to an XYZ file.
        """
        data_to_xyz(self.xyz_df, self.molecule_name+'.xyz')
    
    def write_csv_files(self) -> None:
        """
        Writes all class variables to CSV files with the variable name as the file 
        """
        for var_name, var_value in vars(self).items():
            if isinstance(var_value, pd.DataFrame):
                var_value.to_csv(f"{var_name}.csv", index=False)


    def visualize_molecule(self, vector=None) -> None:
        """
        Visualizes the molecule using the `visualize` module.
        """
        if vector is not None:
            visualize.show_single_molecule(molecule_name=self.molecule_name, xyz_df=self.xyz_df, dipole_df=vector,origin=[0,0,0])
        else:
            visualize.show_single_molecule(molecule_name=self.molecule_name, xyz_df=self.xyz_df, dipole_df=self.gauss_dipole_df,origin=[0,0,0])


    def process_sterimol_atom_group(self, atoms, radii, sub_structure=True, drop_atoms=None,visualize_bool=None) -> pd.DataFrame:

        connected = get_molecule_connections(self.bonds_df, atoms[0], atoms[1])
        
        return get_sterimol_df(self.xyz_df, self.bonds_df, atoms, connected, radii, sub_structure=sub_structure, drop_atoms=drop_atoms, visualize_bool=visualize_bool)

    def get_sterimol(self, base_atoms: Union[None, Tuple[int, int]] = None, radii: str = 'CPK',sub_structure=True, drop_atoms=None,visualize_bool=None) -> pd.DataFrame:
        """
        Returns a DataFrame with the Sterimol parameters calculated based on the specified base atoms and radii.

        Args:
            base_atoms (Union[None, Tuple[int, int]], optional): The indices of the base atoms to use for the Sterimol calculation. Defaults to None.
            radii (str, optional): The radii to use for the Sterimol calculation. Defaults to 'bondi'.

        Returns:
            pd.DataFrame: A DataFrame with the Sterimol parameters.
            
            to add
            - only_sub- sterimol of only one part.
            - drop some atoms.
        """
        if base_atoms is None:
            base_atoms = get_sterimol_indices(self.xyz_df, self.bonds_df)
        
        if isinstance(base_atoms[0], list):
            # If base_atoms is a list of lists, process each group individually and concatenate the results
            sterimol_list = [self.process_sterimol_atom_group(atoms, radii, sub_structure=sub_structure, drop_atoms=drop_atoms,visualize_bool=visualize_bool) for atoms in base_atoms]
            sterimol_df = pd.concat(sterimol_list, axis=0)

        else:
            # If base_atoms is a single group, just process that group
            sterimol_df = self.process_sterimol_atom_group(base_atoms, radii,sub_structure=sub_structure, drop_atoms=drop_atoms,visualize_bool=visualize_bool)
        return sterimol_df


    def swap_atom_pair(self, pair_indices: Tuple[int, int]) -> pd.DataFrame:
        """
        Swaps the positions of two atoms in the molecule and returns a new DataFrame with the updated coordinates.

        Args:
            pair_indices (Tuple[int, int]): The indices of the atoms to swap.

        Returns:
            pd.DataFrame: A new DataFrame with the updated coordinates.
        """
        pairs = adjust_indices(pair_indices)
        xyz_df = self.xyz_df
        temp = xyz_df.iloc[pairs[0]].copy()
        xyz_df.iloc[pairs[0]] = self.coordinates_array[pairs[1]]
        xyz_df.iloc[pairs[1]] = temp
        return xyz_df

    
    def get_charge_df(self, atoms_indices: List[int], 
                  type: Union[str, List[str]] = 'all'
                 ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Returns a DataFrame with the charges for the specified atoms for the requested type(s).

        Args:
            atoms_indices (List[int]): Indices of atoms to include.
            type (str or List[str], optional): Charge type to extract (e.g., 'nbo', 'hirshfeld', 'cm5').
                                            Use 'all' or a list of types to extract multiple types.
                                            Defaults to 'nbo'.

        Returns:
            pd.DataFrame or Dict[str, pd.DataFrame]: A DataFrame for a single type, or a dictionary 
            of DataFrames if multiple types are requested. If 'all' is specified, any missing/invalid 
            types are skipped instead of raising an error.
        """
        atoms_indices = adjust_indices(atoms_indices)

        def extract_charge(df: pd.DataFrame) -> pd.DataFrame:
            # Extract specified rows, rename indices, and transpose
            return df.iloc[atoms_indices].rename(index=lambda x: f'atom_{x + 1}').T

        # Handle different input cases for 'type'
        if isinstance(type, str):
            if type.lower() == 'all':
                # Instead of raising an error on missing types,
                # we iterate through every type in self.charge_dict
                # and skip any that fail.
                results = {}
                for t, df in self.charge_dict.items():
                    try:
                        results[t] = extract_charge(df)
                    except Exception as e:
                        # Log or print a warning if desired
                        # print(f"Skipping type '{t}' due to error: {e}")
                        pass
                return results

            else:
                # Single specific type
                if type not in self.charge_dict:
                    raise ValueError(f"Charge type '{type}' is not available in charge_dict.")
                return extract_charge(self.charge_dict[type])

        elif isinstance(type, list):
            # Multiple types, specified by name
            result = {}
            for t in type:
                if t not in self.charge_dict:
                    raise ValueError(f"Charge type '{t}' is not available in charge_dict.")
                result[t] = extract_charge(self.charge_dict[t])
            return result

        else:
            raise TypeError("Parameter 'type' must be a string or a list of strings.")


    def get_charge_diff_df(self, diff_indices: Union[List[List[int]], List[int]], 
                       type: Union[str, List[str]] = 'all'
                      ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Returns a DataFrame with the charge differences for specified atom pairs for the requested type(s).

        Args:
            diff_indices (List[List[int]] or List[int]):
                - For multiple pairs: a list of lists (e.g., [[atom1, atom2], [atom3, atom4]]).
                - For a single pair: a list (e.g., [atom1, atom2]).
            type (str or List[str], optional): Charge type to extract (e.g., 'nbo', 'hirshfeld', 'cm5').
                                            Use 'all' or a list of types to extract multiple types.
                                            Defaults to 'nbo'.

        Returns:
            pd.DataFrame or Dict[str, pd.DataFrame]: A DataFrame for a single type, or a dictionary of DataFrames if multiple types are requested.
            When using 'all' (or a list), any type that fails to compute will be skipped.
        """
        def compute_diff(df: pd.DataFrame) -> pd.DataFrame:
            # If the first element of diff_indices is a list, assume multiple pairs are provided.
            try:
                
                if isinstance(diff_indices[0], list):
                    diff_list = []
                    for atoms in diff_indices:
                        atoms = adjust_indices(atoms)
                        # Calculate difference between the two specified atoms.
                        diff = pd.DataFrame(
                            [df.iloc[atoms[0]].values - df.iloc[atoms[1]].values],
                            columns=[f'diff_{atoms[0]+1}-{atoms[1]+1}']
                        )
                        diff_list.append(diff)
                        # print(f"Difference for atoms {atoms[0]+1} and {atoms[1]+1}: {diff.values}")
                    return pd.concat(diff_list, axis=1)
                else:
                    diff_indices_adj = adjust_indices(diff_indices)
                    diff_df=pd.DataFrame(
                        [df.iloc[diff_indices_adj[0]].values - df.iloc[diff_indices_adj[1]].values],
                        columns=[f'diff_{diff_indices_adj[0]+1}-{diff_indices_adj[1]+1}']
                    )
                    # print(f"Difference for atoms {diff_indices_adj[0]+1} and {diff_indices_adj[1]+1}: {diff_df.values}")
                    return diff_df  
            except Exception as e:
                raise ValueError(f"Error in computing charge difference: {e}")

        # Handle different input cases for 'type'
        if isinstance(type, str):
            if type.lower() == 'all':
                results = {}
                for t, df in self.charge_dict.items():
                    try:
                        results[t] = compute_diff(df)
                    except Exception as e:
                        # Optionally log a warning for type t and skip it.
                        print(f"Skipping type '{t}' due to error: {e}")
                        pass
                return results
            else:
                if type not in self.charge_dict:
                    raise ValueError(f"Charge type '{type}' is not available in charge_dict.")
                return compute_diff(self.charge_dict[type])
        elif isinstance(type, list):
            result = {}
            for t in type:
                if t not in self.charge_dict:
                    raise ValueError(f"Charge type '{t}' is not available in charge_dict.")
                try:
                    result[t] = compute_diff(self.charge_dict[t])
                except Exception as e:
                    # Optionally log a warning for type t and skip it.
                    print(f"Skipping type '{t}' due to error: {e}")
                    pass
            return result
        else:
            raise TypeError("Parameter 'type' must be a string or a list of strings.")

    def get_coordinates_mean_point(self, atoms_indices: List[int]) -> np.ndarray:
        """
        Returns the mean point of the specified atoms.

        Args:
            atoms_indices (List[int]): The indices of the atoms to calculate the mean point for.

        Returns:
            np.ndarray: The mean point of the specified atoms.
        """
        atoms_indices = adjust_indices(atoms_indices)
        return np.mean(self.coordinates_array[atoms_indices], axis=0)
 
    def get_coordination_transformation_df(self, base_atoms_indices: List[int],origin=None) -> pd.DataFrame:
        """
        Returns a new DataFrame with the coordinates transformed based on the specified base atoms.

        Args:
            base_atoms_indices (List[int]): The indices of the base atoms to use for the transformation.

        Returns:
            pd.DataFrame: A new DataFrame with the transformed coordinates.
        """
        new_coordinates_df = preform_coordination_transformation(self.xyz_df, base_atoms_indices,origin)
        return new_coordinates_df

    ## not working after renumbering for some reason
    
    
    def get_npa_df_single(self, atoms: List[int], sub_atoms: Optional[List[int]] = None, type='nbo') -> pd.DataFrame:
        """
        Calculates the NPA charges for a single group of atoms.
        
        Parameters
        ----------
        atoms : List[int]
            The indices of the base atoms used to perform the coordination transformation.
        sub_atoms : Optional[List[int]]
            If provided, only these atom indices (rows) will be taken from the transformed coordinates.
            Otherwise, all atoms (rows) are used.
        type : str, optional
            Type of charge dictionary to use (default 'nbo').
        
        Returns
        -------
        pd.DataFrame
            DataFrame with the calculated NPA charges, renamed by the base atoms.
        """
        if sub_atoms ==[]:
            sub_atoms = None
        # Get the transformed coordinates DataFrame for the given atoms.
        df_trans = self.get_coordination_transformation_df(atoms,sub_atoms)
        # If sub_atoms is provided, select only those rows.
        if sub_atoms is not None:
            df_subset = df_trans.loc[sub_atoms, ['x', 'y', 'z']].astype(float)
        else:
            df_subset = df_trans[['x', 'y', 'z']].astype(float)
        coordinates_array = np.array(df_subset)
        
        # Get the charges from the charge dictionary.
        charges = np.array(self.charge_dict[type])

        # Calculate the NPA charges (assuming calc_npa_charges is defined elsewhere)
        npa_df = calc_npa_charges(coordinates_array, charges)
        npa_df = npa_df.rename(index={0: f'NPA_{atoms[0]}-{atoms[1]}-{atoms[2]}'})
        return npa_df

    def get_npa_df(self, base_atoms_indices: List[int], sub_atoms: Optional[List[int]] = None, type='nbo') -> pd.DataFrame:
        """
        Returns a DataFrame with the NPA charges calculated based on the specified base atoms 
        and optionally only the sub atoms.
        
        Parameters
        ----------
        base_atoms_indices : List[int] or List[List[int]]
            The indices of the base atoms (or groups of atoms) to use for the NPA calculation.
        sub_atoms : Optional[List[int]] or Optional[List[List[int]]]
            The indices of the sub atoms to use for the NPA calculation. If provided, only these rows 
            will be taken from the transformed coordinates.
        type : str, optional
            Type of charge dictionary to use (default 'nbo').
        
        Returns
        -------
        pd.DataFrame
            A DataFrame with the calculated NPA charges.
        """
        # If the second element is a list, then base_atoms_indices is a list of groups.
        if isinstance(base_atoms_indices[1], list):
            # If sub_atoms is provided as a list of lists, then zip them.
            if sub_atoms and isinstance(sub_atoms[0], list):
                npa_list = [
                    self.get_npa_df_single(atoms, sub_atoms=sub_group, type=type)
                    for atoms, sub_group in zip(base_atoms_indices, sub_atoms)
                ]
            else:
                # Otherwise, pass the same sub_atoms list to each group.
                npa_list = [
                    self.get_npa_df_single(atoms, sub_atoms=sub_atoms, type=type)
                    for atoms in base_atoms_indices
                ]
            npa_df = pd.concat(npa_list, axis=0)
        else:
            npa_df = self.get_npa_df_single(base_atoms_indices, sub_atoms=sub_atoms, type=type)
        return npa_df
    
    def get_dipole_gaussian_df_single(self, atoms,origin ,visualize_bool=False) -> pd.DataFrame:
        
        dipole_df = calc_dipole_gaussian(self.coordinates_array, np.array(self.gauss_dipole_df), atoms,origin=origin)
        dipole_df = dipole_df.rename(index={0: f'dipole_{atoms[0]}-{atoms[1]}-{atoms[2]}'})
        if visualize_bool:
            visualize.plot_interactions(xyz_df=self.xyz_df,color='black', dipole_df=dipole_df,origin=origin)

        return dipole_df

    def get_dipole_gaussian_df(self, base_atoms_indices: List[int], origin=None, visualize_bool=False) -> pd.DataFrame:
        """
        Returns a DataFrame with the dipole moments calculated based on the specified base atoms.

        Args:
            base_atoms_indices (List[int]): The indices of the base atoms to use for the dipole moment calculation.

        Returns:
            pd.DataFrame: A DataFrame with the dipole moments.
        """
     
        if isinstance(base_atoms_indices[1], list):
           # If base_atoms_indices is a list of lists, process each group individually and concatenate the results
            dipole_list = [self.get_dipole_gaussian_df_single(atoms,origin=origin,visualize_bool=visualize_bool) for atoms in base_atoms_indices]
            dipole_df = pd.concat(dipole_list, axis=0)
        else:
            # If base_atoms_indices is a single group, just process that group
            dipole_df = self.get_dipole_gaussian_df_single(base_atoms_indices,origin=origin,visualize_bool=visualize_bool)
        return dipole_df


    def get_bond_angle(self, atom_indices: List[int]) -> pd.DataFrame:
        """
        Returns a DataFrame with the bond angles calculated based on the specified atom indices.

        Args:
            atom_indices (List[int]): The indices of the atoms to use for the bond angle calculation.

        Returns:
            pd.DataFrame: A DataFrame with the bond angles.
        """
        return get_angle_df(self.coordinates_array, atom_indices)
    
    def get_bond_length_single(self, atom_pair):
        bond_length = calc_single_bond_length(self.coordinates_array, atom_pair)
        bond_length_df = pd.DataFrame([bond_length], index=[f'bond_length_{atom_pair[0]}-{atom_pair[1]}'])
        return bond_length_df

    def get_bond_length(self, atom_pairs):
        """
            Returns a DataFrame with the bond lengths calculated based on the specified atom pairs.

            Args:
                atom_pairs (Union[List[Tuple[int, int]], Tuple[int, int]]): The pairs of atoms to use for the bond length calculation.

            Returns:
                pd.DataFrame: A DataFrame with the bond lengths.
            """
        if isinstance(atom_pairs[0], list):
            # If atom_pairs is a list of lists, process each pair individually and concatenate the results
            bond_length_list = [self.get_bond_length_single(pair) for pair in atom_pairs]
            bond_df = pd.concat(bond_length_list, axis=0)
        else:
            # If atom_pairs is a single pair, just process that pair
            bond_df = self.get_bond_length_single(atom_pairs)
        return bond_df


    def get_vibration_max_frequency(self, vibration_atom_num: int) -> float:
        """
        Returns the maximum frequency magnitude for the vibration data of the specified atom number.

        Args:
            vibration_atom_num (int): The atom number for which to retrieve the vibration data.

        Returns:
            float: The maximum frequency magnitude for the vibration data of the specified atom number.
        """
        vibration_array: Union[List[float], None] = self.vibration_dict.get('vibration_atom_{}'.format(str(vibration_atom_num)))
        return calc_max_frequency_magnitude(vibration_array, self.info_df.T)

    def get_stretch_vibration_single(self, atom_pair: List[int],threshold=3000)-> pd.DataFrame:
        
       
        if check_pair_in_bonds(atom_pair, self.bonds_df) == True:
            try:
                extended_vib_df = calc_vibration_dot_product_from_pairs(
                    self.coordinates_array, self.vibration_dict, atom_pair, self.info_df,threshold=threshold
                )
            except TypeError:
                print(f'Strech Vibration Error: no vibration array for the molecule {self.molecule_name} for {atom_pair} - check atom numbering in molecule')
                return None
            vibration_df, idx = calc_max_frequency_gen_vibration(extended_vib_df)
            return vibration_df.rename(index={idx: f'Stretch_{atom_pair[0]}_{atom_pair[1]}'})
        else:
            print(f'Strech Vibration Error: the following bonds do not exist-check atom numbering in molecule: \n {self.molecule_name} for {atom_pair} \n')
            
            df=pd.DataFrame([[np.nan,np.nan]],columns=[['Frequency','Amplitude']])
            df.rename(index={0: f'Stretch_{atom_pair[0]}_{atom_pair[1]}'},inplace=True)
            
            
            return df
    
    def get_stretch_vibration(self, atom_pairs: List[int],threshold=3000)-> pd.DataFrame:
        """
        Parameters
        ----------
        atom_pairs : molecule_1.get_stretch_vibration([[1,6],[3,4]])
            atom pairs must have a corresponding vibration file and appear in bonds_df.

        Returns
        -------
        dataframe

        [                   0
         IR            0.7310
         Frequency  1655.5756,
                             0
         IR            0.35906
         Frequency  1689.59450]

        """
        
      
        if isinstance(atom_pairs[0], list):
            # If atom_pairs is a list of lists, process each pair individually and concatenate the results
            vibration_list = [self.get_stretch_vibration_single(pair,threshold) for pair in atom_pairs]
            # Filter out None results
            vibration_list = [vib for vib in vibration_list if vib is not None]
            vibration_df = pd.concat(vibration_list, axis=0)
            return vibration_df
        else:
            # If atom_pairs is a single pair, just process that pair
            vibration_df=self.get_stretch_vibration_single(atom_pairs,threshold)
            
            return vibration_df
        
    
    def get_ring_vibrations(self,ring_atom_indices: List[List[int]])-> pd.DataFrame:
        """
        Parameters
        ----------
        ring_atom_indices :working example: molecule_1.get_ring_vibrations([6]) 
            
        enter a list of the primary axis atom and the para atom to it.
        For example - for a ring of atoms 1-6 where 4 is connected to the main group and 1 is para to it
        (ortho will be 3 & 5 and meta will be 2 & 6) - enter the input [1] or [1,4].
            
        Returns
        -------
        dataframe
            cross  cross_angle      para  para_angle
        0  657.3882    81.172063  834.4249   40.674833

        """
        try:
            if isinstance(ring_atom_indices[0], list):
                df_list= []
                
                for atoms in ring_atom_indices:
                    
                    z,x,c,v,b,n=get_benzene_ring_indices(self.bonds_df, atoms)
                    ring_atom_indices=[[z,x],[c,v],[b,n]]
                    try:
                        filtered_df=get_filtered_ring_df(self.info_df,self.coordinates_array,self.vibration_dict,ring_atom_indices)
                    except FileNotFoundError:
                        return print(f'No vibration - Check atom numbering in molecule {self.molecule_name}')
                    df=calc_min_max_ring_vibration(filtered_df)
                    ## edit the index to include the atom numbers
                    df.rename(index={'cross': f'cross_{atoms}','cross_angle':f'cross_angle{atoms}', 'para': f'para{atoms}','para_angle': f'para_angle_{atoms}'},inplace=True)
                    df_list.append(df)
                return pd.concat(df_list, axis=0)
            else:
                try:
                    z,x,c,v,b,n=get_benzene_ring_indices(self.bonds_df, ring_atom_indices)
                    ring_atom_indices=[[z,x],[c,v],[b,n]]
                except Exception as e:
                    print(f'Error in get_ring_vibrations: {e}')
                    log_exception()
                    return None
            
                try:
                    
                    filtered_df=get_filtered_ring_df(self.info_df,self.coordinates_array,self.vibration_dict,ring_atom_indices)
                except FileNotFoundError:
                    return print(f'No vibration - Check atom numbering in molecule {self.molecule_name}')
            
                return calc_min_max_ring_vibration(filtered_df)
        except:
            print("this molecule is ",self.molecule_name)
            log_exception()
    
    def get_bend_vibration_single(self, atom_pair: List[int], threshold: float = 1300)-> pd.DataFrame:
        # Create the adjacency dictionary for the pair of atoms
        adjacency_dict = create_adjacency_dict_for_pair(self.bonds_df, atom_pair)
        center_atom_exists = find_center_atom(atom_pair[0], atom_pair[1], adjacency_dict)
        if not center_atom_exists:
            print(f'Bend Vibration - Atoms do not share a center in molecule {self.molecule_name} - for atoms {atom_pair} check atom numbering in molecule')
            return None
        else:
            # Create the extended DataFrame for the vibration modes
            extended_df = extended_df_for_vib(self.vibration_dict, self.info_df, atom_pair, threshold)
            # Calculate the cross product and magnitude for each row in the extended DataFrame
            cross_list = []
            cross_mag_list = []
            for i in range(extended_df.shape[0]):
                cross_list.append(np.cross(extended_df[[0, 1, 2]].iloc[i], extended_df[[3, 4, 5]].iloc[i]))
                cross_mag_list.append(np.linalg.norm(cross_list[-1]))
            # Add the cross magnitude column to the extended DataFrame
            extended_df['Cross_mag'] = cross_mag_list
            extended_df.reset_index(drop=True, inplace=True)
            # Find the row with the maximum cross magnitude and extract the bending frequency and cross magnitude

            index_max = extended_df['Cross_mag'].idxmax()
            max_frequency_vibration = (pd.DataFrame(extended_df.iloc[index_max]).T)[['Frequency', 'Cross_mag']]
            max_frequency_vibration = max_frequency_vibration.rename(index={index_max: f'Bending_{atom_pair[0]}-{atom_pair[1]}'})
            return max_frequency_vibration


    def get_bend_vibration(self, atom_pairs: List[str], threshold: float = 1300) -> pd.DataFrame:
        """"

        Finds the bending frequency for a pair of atoms in a molecule.

        Args from self:
            atom_pair (list): A list of two atom symbols representing the pair of atoms to find the bending frequency for.
            vibration_dict (dict): A dictionary that maps each vibration mode to a list of its frequencies.
            bonds_df (pd.DataFrame): A DataFrame that contains the bond information for the molecule.
            info_df (pd.DataFrame): A DataFrame that contains the information for each vibration mode in the molecule.
            threshold (float): The frequency threshold for selecting vibration modes.

        Returns:
            pd.DataFrame: If the atoms do not share a center, returns a string indicating this. Otherwise, returns a DataFrame that contains the bending frequency and cross magnitude for the pair of atoms.

        """
        
        if isinstance(atom_pairs[0], list):
            # If atom_pairs is a list of lists, process each pair individually and concatenate the results
            vibration_list = [self.get_bend_vibration_single(pair, threshold) for pair in atom_pairs]
            vibration_df = pd.concat(vibration_list, axis=0)
            return vibration_df
        else:
            # If atom_pairs is a single pair, just process that pair
            return self.get_bend_vibration_single(atom_pairs, threshold)



class Molecules():
    
    def __init__(self,molecules_dir_name, renumber=False):
        self.molecules_path=os.path.abspath(molecules_dir_name)
        os.chdir(self.molecules_path) 
        self.molecules=[]
        self.failed_molecules=[]
        self.success_molecules=[]
        for feather_file in os.listdir(): 
            if feather_file.endswith('.feather'):
                # try:
                self.molecules.append(Molecule(feather_file))
                self.success_molecules.append(feather_file)
                # except Exception as e:
                #     self.failed_molecules.append(feather_file)
                #     print(f'Error: {feather_file} could not be processed : {e}')
                   
        print(f'Molecules Loaded: {self.success_molecules}',f'Failed Molecules: {self.failed_molecules}')

        self.molecules_names=[molecule.molecule_name for molecule in self.molecules]
        self.old_molecules=self.molecules
        self.old_molecules_names=self.molecules_names
        os.chdir('../')

    def export_all_xyz(self):
        os.makedirs('xyz_files', exist_ok=True)
        os.chdir('xyz_files')
        for mol in self.molecules:
            xyz_df=mol.xyz_df
            mol_name=mol.molecule_name
            data_to_xyz(xyz_df, f'{mol_name}.xyz')
        os.chdir('../')

    def filter_molecules(self, indices):
        self.molecules = [self.molecules[i] for i in indices]
        self.molecules_names = [self.molecules_names[i] for i in indices]

    def get_sterimol_dict(self,atom_indices, radii='CPK',sub_structure=True, drop_atoms=None,visualize_bool=None):
        """
        Returns a dictionary with the Sterimol parameters calculated based on the specified base atoms.

        Args:
            base_atoms (Union[None, List[int, int], List[List[int, int]]]): The indices of the base atoms to use for the Sterimol calculation.
            radii (str, optional): The radii to use for the Sterimol calculation. Defaults to 'CPK'.
        Returns:
            Dict[str, pd.DataFrame]: A dictionary where each key is a molecule name and each value is a DataFrame with the Sterimol parameters.
        
        input example: molecules.get_sterimol_dict([[1,6],[3,4]])
        output example: 
        Results for LS1716_optimized:
            B1    B5     L  loc_B1  loc_B5
        1-2  1.91  8.12  8.89    5.35    6.51
        3-4  1.70  5.25  9.44    1.70   -1.65


        Results for LS1717_optimized:
            B1    B5     L  loc_B1  loc_B5
        1-2  1.9  7.98  9.02    3.64    6.68
        3-4  1.7  6.45  9.45    1.70   -1.97
        """
        sterimol_dict={}
        
        for molecule in self.molecules:
            try:
                sterimol_dict[molecule.molecule_name]=molecule.get_sterimol(atom_indices, radii, sub_structure=sub_structure, drop_atoms=drop_atoms, visualize_bool=visualize_bool)
            except Exception as e:
                print(f'Error: {molecule.molecule_name} sterimol could not be processed: {e}')
                log_exception("get_sterimol_dict")
                pass
        return sterimol_dict
    
    def get_npa_dict(self,atom_indices,sub_atoms=None):
        """
        Returns a dictionary with the Natural Population Analysis (NPA) charges calculated for the specified base atoms and sub atoms.

        Args:
            base_atoms (List[int]): The indices of the base atoms to use for the NPA calculation.
            sub_atoms (Union[List[int], None], optional): The indices of the sub atoms to use for the NPA calculation. Defaults to None.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary where each key is a molecule name and each value is a DataFrame with the NPA charges.
        
        input example: molecules.get_npa_dict([1, 2, 3], [5, 6, 7])
        output example: 
        Results for LS1716_optimized:
                    dip_x     dip_y     dip_z  total_dipole
            NPA_1-2-3  0.092108  0.181346 -0.300763      0.363082
            NPA_5-6-7  0.191986  0.297839  0.079456      0.363153

        Results for LS1717_optimized:
                    dip_x     dip_y     dip_z  total_dipole
            NPA_1-2-3  0.126370  0.271384  0.354595      0.464065
            NPA_5-6-7  0.225257  0.399616 -0.069960      0.464035
        """
        
        npa_dict={}
        for molecule in self.molecules:
            try:
                npa_dict[molecule.molecule_name]=molecule.get_npa_df(atom_indices,sub_atoms)
            except:
                print(f'Error: {molecule.molecule_name} npa could not be processed')
                log_exception("get_npa_dict")
                pass
            
        return npa_dict
    

    
    def get_ring_vibration_dict(self,ring_atom_indices,threshold=1600):
        """
        Parameters
        ----------
        ring_atom_indices :working example: molecule_1.get_ring_vibrations([[8,11],[9,12]]) 
            
        enter a list of the primary axis atom and the para atom to it.
        For example - for a ring of atoms 1-6 where 4 is connected to the main group and 1 is para to it
        (ortho will be 3 & 5 and meta will be 2 & 6) - enter the input [1,4].
            
        Returns
        -------
        dataframe
        Results for LS1717_optimized:
                              0
        cross_[8, 11]       1666.188400
        cross_angle[8, 11]    89.079604
        para[8, 11]         1462.659400
        para_angle_[8, 11]    20.657101
        cross_[7, 10]       1666.188400
        cross_angle[7, 10]    86.888386
        para[7, 10]         1462.659400
        para_angle_[7, 10]     8.628947

        """
        ring_dict={}
        for molecule in self.molecules:
            try:
                ring_dict[molecule.molecule_name]=molecule.get_ring_vibrations(ring_atom_indices)
            except Exception as e:
                print(f'Error: {molecule.molecule_name} ring vibration could not be processed: {e}')
                traceback.print_exc()
                pass
        return ring_dict
    
    def get_dipole_dict(self,atom_indices,origin=None,visualize_bool=False):
        """
        Returns a dictionary with the dipole moments calculated for the specified atom indices.
        Args:
            atom_indices (List[int]): The indices of the atoms to use for the dipole moment calculation.
            origin (List[float], optional): The origin point for the dipole moment calculation. Defaults to None.
            visualize_bool (bool, optional): Whether to visualize the dipole moment. Defaults to False.
        Returns:
            Dict[str, pd.DataFrame]: A dictionary where each key is a molecule name and each value is a DataFrame with the dipole moment data.
        input example: molecules.get_dipole_dict([1, 2, 3])
        output example:
        Results for LS1716_optimized:
                        dipole_x     dipole_y     dipole_z  total_dipole
            dipole_1-2-3  0.092108  0.181346 -0.300763      0.363082    
        """

        dipole_dict={}

        for molecule in self.molecules:
            try:
                dipole_dict[molecule.molecule_name]=molecule.get_dipole_gaussian_df(atom_indices,origin=origin,visualize_bool=visualize_bool)
            except Exception as e:
                print(f'Error: {molecule.molecule_name} Dipole could not be processed: {e}')
                log_exception("get_dipole_dict")
        return dipole_dict
    
    def get_bond_angle_dict(self,atom_indices):
        """
        Returns a dictionary with the bond angles calculated for the specified atom indices.

        Args:
            atom_indices (List[List[int, int]]) or : The indices of the atoms to use for the bond angle calculation.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary where each key is a molecule name and each value is a DataFrame with the bond angle data.
        
        input example: molecules.get_bond_angle_dict([1, 2, 3])
        output example: 
        Results for LS1716_optimized:
                        Angle
            Angle_1-2-3  109.5

        Results for LS1717_optimized:
                        Angle
            Angle_1-2-3  108.7
        """

        bond_angle_dict={}
        for molecule in self.molecules:
            try:
                bond_angle_dict[molecule.molecule_name]=molecule.get_bond_angle(atom_indices)
            except:
                print(f'Error: {molecule.molecule_name} Angle could not be processed')
                log_exception("get_bond_angle_dict")
                pass
        return bond_angle_dict
    
    def get_bond_length_dict(self,atom_pairs):
        """
        Returns a dictionary with the bond lengths calculated for the specified atom pairs.

        Args:
            atom_pairs  (List[List[int, int]]) : The pairs of atoms to use for the bond length calculation.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary where each key is a molecule name and each value is a DataFrame with the bond length data.
        
        input example: molecules.get_bond_length_dict([[1, 2], [3, 4]])
        output example: 
        Results for LS1716_optimized:
                        0
        bond_length_1-2  1.529754
        bond_length_3-4  1.466212


        Results for LS1717_optimized:
                                0
        bond_length_1-2  1.511003
        bond_length_3-4  1.466089
        """

        bond_length_dict={}
        for molecule in self.molecules:
            try:
                bond_length_dict[molecule.molecule_name]=molecule.get_bond_length(atom_pairs)
            except:
                print(f'Error: {molecule.molecule_name} Bond Length could not be processed')
                print(log_exception("get_bond_length_dict"))
                pass
        return bond_length_dict
    
    def get_stretch_vibration_dict(self,atom_pairs,threshold=3000):
        """
        Returns a dictionary with the stretch vibrations calculated for the specified atom pairs.

        Args:
            atom_pairs (List[Tuple[int, int]]): The pairs of atoms to use for the stretch vibration calculation.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary where each key is a molecule name and each value is a DataFrame with the stretch vibration data.
        
        input example: molecules.get_stretch_vibration_dict([[1, 2], [3, 4]])
        output example: 
        Results for LS1716_optimized:
                        Frequency  Amplitude
            Stretch_1_2  3174.3565   0.330304
            Stretch_3_4  3242.4530   0.170556

        Results for LS1717_optimized:
                        Frequency  Amplitude
            Stretch_1_2  3242.4465   0.252313
            Stretch_3_4  3175.4029   0.443073
    """
        stretch_vibration_dict={}
        print(f'Calculating stretch vibration for atoms {atom_pairs} with threshold {threshold} \n Remember : ALWAYS LOOK AT THE RESULTING VIBRATION')
        for molecule in self.molecules:
            try:
                stretch_vibration_dict[molecule.molecule_name]=molecule.get_stretch_vibration(atom_pairs,threshold)
            except:
                print(f'Error: could not calculate strech vibration for {molecule.molecule_name} ')
                log_exception("get_stretch_vibration_dict")
                pass

        return stretch_vibration_dict
    
    def get_charge_df_dict(self,atom_indices):
        """
        Returns a dictionary with the Natural Bond Orbital (NBO) charges for the specified atoms.

        Args:
            atoms_indices (List[int]): The indices of the atoms to include in the NBO charge calculation.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary where each key is a molecule name and each value is a DataFrame with the NBO charges.
        
        input example: molecules.get_charge_df_dict([3, 5, 7, 9])
        output example: 
        Results for LS1716_optimized:
                    atom_3   atom_5   atom_7   atom_9
            charge -0.12768 -0.39006  0.14877 -0.00656

        Results for LS1717_optimized:
                    atom_3   atom_5   atom_7   atom_9
            charge -0.12255 -0.38581  0.14691 -0.00681
        """

        nbo_dict={}
        for molecule in self.molecules:
            try:
                nbo_dict[molecule.molecule_name]=molecule.get_charge_df(atom_indices,type='all')
            except:
                print(f'Error: could not calculate nbo value for {molecule.molecule_name} ')
                log_exception("get_charge_df_dict")
                pass
        return nbo_dict
    
    def get_charge_diff_df_dict(self,atom_indices,type='all'):
        """
        Returns a dictionary with the differences in Natural Bond Orbital (NBO) charges for the specified pairs of atoms.

        Args:
            diff_indices (List[List[int]]): The indices of the atom pairs to calculate the NBO charge differences for.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary where each key is a molecule name and each value is a DataFrame with the NBO charge differences.
        
        input example: molecules.get_charge_diff_df_dict([[1, 2], [3, 4]])
        output example: 
        Results for LS1716_optimized:
                        diff_1-2  diff_3-4
            charge -0.12768 -0.39006  0.14877 -0.00656


            Results for LS1717_optimized:
                        diff_1-2  diff_3-4
            charge -0.12255 -0.38581  0.14691 -0.00681
        """
        charge_diff_dict={}
        for molecule in self.molecules:
            try:
                charge_diff_dict[molecule.molecule_name]=molecule.get_charge_diff_df(atom_indices,type)
            except:
                print(f'Error: could not calculate nbo difference for {molecule.molecule_name} ')
                log_exception("get_charge_diff_df_dict")
                pass
        return charge_diff_dict
    
    def get_bend_vibration_dict(self,atom_pairs,threshold=1300):
        """
        Returns a dictionary with the bending vibrations calculated for the specified pairs of atoms.

        Args:
            atom_pairs (List[Tuple[int, int]]): The pairs of atoms to use for the bending vibration calculation.
            threshold (float, optional): The frequency threshold for selecting vibration modes. Defaults to 1300.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary where each key is a molecule name and each value is a DataFrame with the bending vibration data.
        
        input example: molecules.get_bend_vibration_dict([[1, 2], [3, 4]])
        output example: 
        Results for LS1716_optimized:
                        Frequency  Cross_mag
            Bending_1-2  1300.6785    0.123456
            Bending_3-4  1400.5678    0.234567

        Results for LS1717_optimized:
                        Frequency  Cross_mag
            Bending_1-2  1350.1234    0.345678
            Bending_3-4  1450.2345    0.456789
        """
        bending_dict={}
        print(f'Calculating Bend vibration for atoms {atom_pairs} with threshold {threshold} \n Remember : ALWAYS LOOK AT THE RESULTING VIBRATION')
        for molecule in self.molecules:
            try:
                bending_dict[molecule.molecule_name]=molecule.get_bend_vibration(atom_pairs,threshold)
            except:
                print(f'Error: could not calculate bend vibration for {molecule.molecule_name} ')
                log_exception("get_bend_vibration_dict")
                pass
        return bending_dict
    
    def visualize_molecules(self,indices=None):
        if indices is not None:
            for idx in indices:
                self.molecules[idx].visualize_molecule()
        else:
            for molecule in self.molecules:
                molecule.visualize_molecule()
        
    def visualize_smallest_molecule(self):
        """
        Visualizes the smallest molecule based on the number of atoms.

        Args:
            None
        
        Returns:
            None
        
        input example: molecules.visualize_smallest_molecule()
        output example: 
        # This will open a visualization window or generate a visualization file for the smallest molecule.

        """
        idx=0
        smallest= len(self.molecules[0].xyz_df)
        for id, molecule in enumerate(self.molecules[1:]):
            if len(molecule.xyz_df)<smallest:
                smallest=len(molecule.xyz_df)
                idx=id
        html=self.molecules[idx].visualize_molecule()
        return html
    
    def visualize_smallest_molecule_morfeus(self, indices=None):
        idx=0
        
        smallest= len(self.molecules[0].xyz_df)
        for id, molecule in enumerate(self.molecules[1:]):
            if len(molecule.xyz_df)<smallest:
                smallest=len(molecule.xyz_df)
                idx=id
        mol=self.molecules[idx]
        elements=mol.xyz_df['atom'].values
        coordinates=mol.xyz_df[['x','y','z']].values
        sterimol=Sterimol(elements, coordinates, indices[0], indices[1], radii_type='crc')
        
        sterimol.draw_3D()
        
    
    def get_molecules_comp_set_app(self,
                               answers_dict: dict,
                               radii='CPK',
                               export_csv=False,
                               answers_list_load=None,
                               iso=False):
        """
        molecules.get_molecules_comp_set()
        ...
        """
        answers_list = []
        print(answers_dict)
        if answers_dict is not None:
            for question, answer in answers_dict.items():
                if answer is not None:
                    if answer == []:
                        answers_list.append([])
                    else:
                        answers_dict[question] = convert_to_list_or_nested_list(answer)
                        answers_list.append(answers_dict[question])

        if answers_list_load is not None:
            answers_list = answers_list_load

        res_df = pd.DataFrame()

        # ring vibration
        if len(answers_list) > 0 and answers_list[0]:
            try:
                res_df = dict_to_horizontal_df(
                    self.get_ring_vibration_dict(answers_list[0])
                )
            except Exception:
                log_exception("get_molecules_comp_set_app – ring vibration")

        # stretch vibration
        if len(answers_list) > 2 and answers_list[2]:
            try:
                res_df = pd.concat([
                    res_df,
                    dict_to_horizontal_df(
                        self.get_stretch_vibration_dict(
                            answers_list[2],
                            answers_list[1][0]
                        )
                    )
                ], axis=1)
            except Exception:
                log_exception("get_molecules_comp_set_app – stretch vibration")

        # bend vibration
        if len(answers_list) > 4 and answers_list[4]:
            try:
                res_df = pd.concat([
                    res_df,
                    dict_to_horizontal_df(
                        self.get_bend_vibration_dict(
                            answers_list[4],
                            answers_list[3][0]
                        )
                    )
                ], axis=1)
            except Exception:
                log_exception("get_molecules_comp_set_app – bend vibration")

        # NPA
        if len(answers_list) > 5 and answers_list[5]:
            try:
                res_df = pd.concat([
                    res_df,
                    dict_to_horizontal_df(
                        self.get_npa_dict(
                            answers_list[5],
                            sub_atoms=answers_list[6]
                        )
                    )
                ], axis=1)
            except Exception:
                log_exception("get_molecules_comp_set_app – NPA")

        # Dipole
        if len(answers_list) > 7 and answers_list[7]:
            try:
                res_df = pd.concat([
                    res_df,
                    dict_to_horizontal_df(
                        self.get_dipole_dict(
                            answers_list[7],
                            origin=answers_list[6]
                        )
                    )
                ], axis=1)
            except Exception:
                log_exception("get_molecules_comp_set_app – dipole")

        # Charge
        if len(answers_list) > 8 and answers_list[8]:
            try:
                res_df = pd.concat([
                    res_df,
                    charge_dict_to_horizontal_df(
                        self.get_charge_df_dict(answers_list[8])
                    )
                ], axis=1)
                if res_df.empty:
                    try:
                        res_df = pd.concat([
                            res_df,
                            dict_to_horizontal_df(
                                self.get_charge_diff_df_dict(answers_list[8])
                            )
                        ], axis=1)
                    except Exception:
                        log_exception("get_molecules_comp_set_app – charge fallback")
            except Exception:
                log_exception("get_molecules_comp_set_app – charge")

        # Charge differences
        if len(answers_list) > 9 and answers_list[9]:
            try:
                res_df = pd.concat([
                    res_df,
                    charge_dict_to_horizontal_df(
                        self.get_charge_diff_df_dict(answers_list[9])
                    )
                ], axis=1)
                if res_df.empty:
                    try:
                        res_df = pd.concat([
                            res_df,
                            dict_to_horizontal_df(
                                self.get_charge_diff_df_dict(answers_list[9])
                            )
                        ], axis=1)
                    except Exception:
                        log_exception("get_molecules_comp_set_app – charge diff fallback")
            except Exception:
                log_exception("get_molecules_comp_set_app – charge diff")

        # Sterimol
        if len(answers_list) > 10 and answers_list[10]:
            try:
                res_df = pd.concat([
                    res_df,
                    dict_to_horizontal_df(
                        self.get_sterimol_dict(answers_list[10], radii=radii)
                    )
                ], axis=1)
            except Exception:
                log_exception("get_molecules_comp_set_app – sterimol")

        # Bond lengths
        if len(answers_list) > 11 and answers_list[11]:
            try:
                res_df = pd.concat([
                    res_df,
                    dict_to_horizontal_df(
                        self.get_bond_length_dict(answers_list[11])
                    )
                ], axis=1)
            except Exception:
                log_exception("get_molecules_comp_set_app – bond length")

        # Bond angles
        if len(answers_list) > 12 and answers_list[12]:
            try:
                res_df = pd.concat([
                    res_df,
                    dict_to_horizontal_df(
                        self.get_bond_angle_dict(answers_list[12])
                    )
                ], axis=1)
            except Exception:
                log_exception("get_molecules_comp_set_app – bond angle")

        # Isotropic polarizability
        if iso:
            self.polarizability_df_concat = pd.DataFrame()
            for molecule in self.molecules:
                info = molecule.polarizability_df.rename(index={0: molecule.molecule_name})
                self.polarizability_df_concat = pd.concat(
                    [self.polarizability_df_concat, info],
                    axis=0
                )
            res_df = pd.concat([res_df, self.polarizability_df_concat], axis=1)

        return res_df








            
    def extract_all_dfs(self):
        os.chdir(self.molecules_path)
        for molecule in self.molecules:
            try:
                os.mkdir(molecule.molecule_name)
            except FileExistsError:
                pass
            os.chdir(molecule.molecule_name)
            molecule.write_xyz_file()
            molecule.write_csv_files()
            os.chdir('../')
        
    
    def extract_all_xyz(self):
        os.chdir(self.molecules_path)
        try:
            os.mkdir('xyz_files')
        except FileExistsError:
            pass
        os.chdir('xyz_files')
        for molecule in self.molecules:
            data_to_xyz(molecule.xyz_df,(molecule.molecule_name+'.xyz'))
        os.chdir('../')

if __name__=='__main__':
    pass