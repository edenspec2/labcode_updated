from typing import List, Optional
import numpy as np
import numpy.typing as npt
import pandas as pd
import math
import networkx as nx

try:
    from ...utils.help_functions import *
except ImportError:
    from utils.help_functions import * 

def calc_max_frequency_magnitude(vibration_array: npt.ArrayLike, info_df: pd.DataFrame,
                                 threshhold: int = 1500) -> pd.DataFrame:  ##add option to return ordered_info_df-like dot.prod.info
    """
    a function that gets vibration and info dataframes and returns
    the frequecy and IR with the max magnitude for the vibration.
    splits the coordinates of vibration to 3 coordinates and calculates the magnituede. takes frequencys greter than 1500
    and returns the frequency and IR corresponding to max magnitude.

    Parameters
    ----------
    vibration_array: np.array
        organized vibration file in array.
    info_df: np.dataframe
        organized info file in dataframe.
    threshhold: int
        the threshhold for the frequency. default is 1500.
        Frequency      IR
   0      20.3253  0.0008
   1      25.3713  0.0023
   2      29.0304  0.0019

    Returns
    -------
    dataframe
        max frequency and IR for specific vibration.

    Output:
                                    54
            Frequency         1689.5945
            IR intensity        6.5260
    """

    magnitude = np.linalg.norm(vibration_array, axis=1)
    df = info_df.T
    df['magnitude'] = magnitude
    outer_finger = (df['Frequency'].astype(float) > threshhold)
    index_max = df[outer_finger]['magnitude'].idxmax()
    return info_df[index_max]


def check_pair_in_bonds(pair, bonds_df):  ##help functions for gen_vibration
    """
    a function that checks that the all given atom pair exists as a bond in the bonds_df
    """
    bonds_list = (bonds_df.astype(int)).values.tolist()
    bool_check = (pair in bonds_list) or (pair[::-1] in bonds_list)
    
    return bool_check


def vibrations_dict_to_list(vibration_dict: dict, vibration_atom_nums: list[int]):
    """
    a function that gets a vibration dictionary and a list of atom numbers and returns a list of vibration arrays
    for each atom number.
    ----------
    vibration_dict : dict
        dictionary containing vibration arrays.
    vibration_atom_nums : list
        a list of chosen atoms to get vibration for.
    Returns
    -------
    vibration_array_pairs ([0]) : list
    containing the arrays from the vibration file, each variable containing two
    arrays corresponding to the bond pairs
        .
    vibration_array_list ([1]): list
        containing all vibration array coordinates.

    """
    try:
        vibration_array_list = [vibration_dict[f'vibration_atom_{num}']for num in vibration_atom_nums] 
    except Exception as e:
        return print('Error: no vibration for those atoms-pick another one')
    vibration_array_pairs = list(
        zip(vibration_array_list[::2], vibration_array_list[1::2]))  # [::2] means every second element
    return vibration_array_pairs, vibration_array_list  ##optional- do not split to pairs
    


def calc_vibration_dot_product(extended_vib_df, coordinates_vector):
    vibration_dot_product_list=[]
    for i in range (extended_vib_df.shape[0]):
        vibration_dot_product_list.append(abs(np.dot(extended_vib_df[[0,1,2]].iloc[i], coordinates_vector)) + abs(np.dot(extended_vib_df[[3,4,5]].iloc[i], coordinates_vector)))
    # vibration_dot_product=abs(np.dot(extended_vib_df[[0,1,2]], coordinates_vector)) + abs(np.dot(extended_vib_df[[3,4,5]], coordinates_vector))
    return vibration_dot_product_list

def extended_df_for_vib(vibration_dict: dict, info_df:pd.DataFrame,atom_pair,threshhold: int = 3000):
    vibration_array_pairs,_= vibrations_dict_to_list(vibration_dict, atom_pair)
    array=pd.DataFrame(np.hstack([vibration_array_pairs[0][0],vibration_array_pairs[0][1]]))
    df=pd.concat([array,info_df['Frequency']],axis=1)
    filter_df=df[df['Frequency']>threshhold]
    return filter_df


def calc_vibration_dot_product_from_pairs(coordinates_array: npt.ArrayLike,
                                          vibration_dict: dict,
                                          atom_pair: list, info_df: pd.DataFrame, operation:str='dot',threshold=3000) -> List[float]:
    """
    Calculates the dot product between a vibration mode vector and the bond vector between two atoms.

    Parameters
    ----------
    coordinates_array: np.array
        An array of x, y, z coordinates for each atom in the molecule.
    vibration_dict: dict
        A dictionary where the keys are atom indices and the values are numpy arrays representing the
        x, y, z components of the vibration mode vectors for that atom.
    atom_pairs: list
        A list of pairs of atom indices representing the bond vectors to calculate the dot product with.

    Returns
    -------
    vibration_dot_product: list
        A list of dot products between each vibration mode vector and the bond vector between the two atoms
        in each pair in `atom_pairs`.
    """
    atoms = adjust_indices(atom_pair)
    extended_df=extended_df_for_vib(vibration_dict,info_df,atom_pair,threshold)
    coordinates_vector=coordinates_array[atoms[0]]-coordinates_array[atoms[1]]
    vibration_dot_product = calc_vibration_dot_product(extended_df, coordinates_vector)
    extended_df['Amplitude']=vibration_dot_product
    extended_df.reset_index(drop=True,inplace=True)
    return extended_df


def calc_max_frequency_gen_vibration(extended_df):  ## fix so 0 is IR and 1 is frequency
    """
    a function that gets info dataframe and vibration dot product and returns the max frequency and IR for each vibration.
    Parameters
    ----------
    vibration_dot_product: list
        list of dot product for each vibration.
    info_df:
        Frequency      IR
   0      20.3253  0.0008
   1      25.3713  0.0023
   2      29.0304  0.0019
    threshhold: int
        the threshhold for the frequency. default is 500.
    Returns
    -------
    IR            0.556928
    Frequency  1124.742600
    IR            0.403272
    Frequency  1128.584700
    """

    # df=pd.DataFrame(np.vstack([vibration_dot_product, (info_df)['Frequency']]),
    #                         index=XYZConstants.VIBRATION_INDEX.value).T
    
    index_max=extended_df['Amplitude'].idxmax()

    max_frequency_vibration=(pd.DataFrame(extended_df.iloc[index_max]).T)[XYZConstants.VIBRATION_INDEX.value]

    return max_frequency_vibration, index_max


def vibration_ring_array_list_to_vector(vibration_array_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    A function that receives a list of vibration arrays, representing a ring, and returns the sum of
    the first, third and fifth arrays as one vector and the sum of the second, fourth and sixth arrays as
    another vector. These vectors represent the ring's vibrations.
    Parameters
    ----------
    vibration_array_list: List[np.ndarray]
        A list containing six vibration arrays.
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]:
        A tuple containing two np.ndarray vectors representing the ring's vibrations.
    """
    atom_one,atom_two,atom_three=vibration_array_list[0], vibration_array_list[4],vibration_array_list[2]
    atom_four,atom_five,atom_six=vibration_array_list[1], vibration_array_list[3],vibration_array_list[5]
    vec_sum_1_3_5 = (atom_one+atom_three+atom_five) ## try
    vec_sum_2_4_6 = (atom_two+atom_four+atom_six)

  
    return vec_sum_1_3_5, vec_sum_2_4_6



def get_data_for_ring_vibration(info_df: pd.DataFrame, vibration_array_list: List[np.ndarray],
                                coordinates_vector: np.ndarray) -> pd.DataFrame:
    
    product = [np.dot(row_1, row_2) for row_1, row_2 in zip(*vibration_ring_array_list_to_vector(vibration_array_list))]
    _, vibration_array_list = vibration_ring_array_list_to_vector(vibration_array_list)

    
    sin_angle = [abs(math.sin(calc_angle(row, coordinates_vector))) for row in vibration_array_list]

    data_df = pd.DataFrame(np.vstack([product, (info_df)['Frequency'], sin_angle]),index=['Product','Frequency','Sin_angle'] )
  
    return data_df.T



def get_filter_ring_vibration_df(data_df: pd.DataFrame, prods_threshhold: float = 0.1,
                                 frequency_min_threshhold: float = 1600,
                                 frequency_max_threshhold: float = 1780) -> pd.DataFrame:
    # Filter based on product value
    filter_prods = (abs(data_df[XYZConstants.RING_VIBRATION_INDEX.value[0]]) > prods_threshhold) & \
                   (data_df[XYZConstants.RING_VIBRATION_INDEX.value[0]] != 0)
    filter_frequency = (data_df[XYZConstants.RING_VIBRATION_INDEX.value[1]] > frequency_min_threshhold) & \
                       (data_df[XYZConstants.RING_VIBRATION_INDEX.value[1]] < frequency_max_threshhold)
    # Apply combined filter
    filtered_df = data_df[filter_prods & filter_frequency].reset_index()
    if filtered_df.empty:
        print('No data within the specified thresholds. Adjust your thresholds.')
    
    return filtered_df



def indices_to_coordinates_vector(coordinates_array,indices):
    """
    a function that recives coordinates_array and indices of two atoms
    and returns the bond vector between them
    """
    try:
        if  isinstance(indices[0], (list, np.ndarray, tuple)):
            bond_vector=[(coordinates_array[index[1]]-coordinates_array[index[0]]) for index in indices]
            
        else:
            bond_vector= coordinates_array[indices[1]]-coordinates_array[indices[0]]
    except:
        if  isinstance(indices[0], tuple):
            bond_vector=[(coordinates_array[index[0]]-coordinates_array[index[1]]) for index in indices]
        else:
            bond_vector= coordinates_array[indices[0]]-coordinates_array[indices[1]]

       
    return bond_vector

def get_filtered_ring_df(info_df: pd.DataFrame, coordinates_array: np.ndarray, vibration_dict: dict,
                         ring_atom_indices: list) -> pd.DataFrame:
    """
    A function that returns a filtered DataFrame of ring vibrations based on their product, frequency, and sin(angle) values.

    Parameters
    ----------
    info_df : pd.DataFrame
        A DataFrame that contains the frequencies and intensities of the vibrational modes.

    coordinates_array : np.ndarray
        A numpy array that contains the x, y, z coordinates of each atom in the molecule.

    vibration_dict : dict
        A dictionary that contains the vibrational modes and their corresponding frequencies and intensities.

    ring_atom_indices : list
        A list of atom indices that define the atoms in the ring.

    Returns
    -------
    filtered_df : pd.DataFrame
        A DataFrame that contains the filtered ring vibrations based on their product, frequency, and sin(angle) values.
    """
    ring_indices = adjust_indices(ring_atom_indices)
    coordinates_vector = indices_to_coordinates_vector(coordinates_array, ring_indices)[0]
    vibration_atom_nums = flatten_list(ring_atom_indices)
    _, vibration_array_list = vibrations_dict_to_list(vibration_dict, vibration_atom_nums)
    data_df = get_data_for_ring_vibration(info_df, vibration_array_list, coordinates_vector)
    filtered_df = get_filter_ring_vibration_df(data_df)
    return filtered_df


def calc_min_max_ring_vibration(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the minimum and maximum vibration frequency and the angle between the vibration vector and the plane of the ring.

    Parameters
    ----------
    filtered_df : pd.DataFrame
        A filtered DataFrame from `get_filtered_ring_df`.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame containing the minimum and maximum vibration frequency and the angle between the vibration vector and the plane of the ring.
    """
    # filtered_df['Product']=filtered_df['Product'].abs()
    max_vibration_frequency = filtered_df.iloc[filtered_df[
        XYZConstants.RING_VIBRATION_INDEX.value[2]].idxmin()][2]  
    asin_max = math.asin(filtered_df[XYZConstants.RING_VIBRATION_INDEX.value[2]].min()) * (
                180 / np.pi)
    min_vibration_frequency = filtered_df.iloc[filtered_df[
        XYZConstants.RING_VIBRATION_INDEX.value[2]].idxmax()][2]
    asin_min = math.asin(filtered_df[XYZConstants.RING_VIBRATION_INDEX.value[2]].max()) * (
                180 / np.pi)
    df = pd.DataFrame((max_vibration_frequency, asin_max, min_vibration_frequency, asin_min),
                      index=XYZConstants.RING_VIBRATION_COLUMNS.value)
    
    return df

### bending vibration

def find_center_atom(atom1: str, atom2: str, adjacency_dict: Dict[str, List[str]]) -> bool:
    """
    Determines if there is a common atom between two atoms in a molecule.

    Args:
        atom1 (str): The first atom.
        atom2 (str): The second atom.
        adjacency_dict (dict): A dictionary that maps each atom to a list of its neighboring atoms.

    Returns:
        bool: True if there is a common atom between atom1 and atom2, False otherwise.
    """
    # Get the lists of neighbors for atom1 and atom2
    neighbors1 = adjacency_dict.get(atom1, [])
    neighbors2 = adjacency_dict.get(atom2, [])
    # Check if there is a common atom in the two lists
    common_atoms = set(neighbors1).intersection(neighbors2)
    # If there are common atoms, return True. Otherwise, return False.
    return bool(common_atoms)

def create_adjacency_dict_for_pair(bond_df: pd.DataFrame, pair: List[str]) -> Dict[str, List[str]]:
    """
    Creates an adjacency dictionary for a pair of atoms in a molecule.

    Args:
        bond_df (pd.DataFrame): A DataFrame that contains the bond information for the molecule.
        pair (list): A list of two atom symbols representing the pair of atoms to create the adjacency dictionary for.

    Returns:
        dict: A dictionary that maps each atom in the pair to a list of its neighboring atoms.
    """
    def create_adjacency_list(bond_df: pd.DataFrame) -> Dict[int, List[int]]:
        """
        Creates an adjacency list for all atoms in a molecule.

        Args:
            bond_df (pd.DataFrame): A DataFrame that contains the bond information for the molecule.

        Returns:
            dict: A dictionary that maps each atom index to a list of its neighboring atom indices.
        """
        adjacency_list = {i: [] for i in np.unique(bond_df.values)}
        for atom1, atom2 in bond_df.values:
            adjacency_list[atom1].append(atom2)
            adjacency_list[atom2].append(atom1)
        return adjacency_list

    # Create the adjacency dictionary for the pair of atoms
    adjacency_dict = create_adjacency_list(bond_df)
    return {atom: adjacency_dict[atom] for atom in pair}

def reindex_and_preserve(df, new_index_order):
        # Select rows that are in the new index order
        reindexed_part = df.loc[df.index.intersection(new_index_order)]

        # Select rows that are not in the new index order
        non_reindexed_part = df.loc[~df.index.isin(new_index_order)]

        # Concatenate the two parts
        return pd.concat([reindexed_part, non_reindexed_part])


def get_benzene_ring_indices(bonds_df, ring_atoms):
    """
    Identifies benzene ring indices from a bond dataframe and a set of ring atoms.
    Also detects and prints fused benzene rings (two rings sharing an edge).
    """
    # Read atom indices
    atom1_idx = ring_atoms[0]
    atom2_idx = ring_atoms[1] if len(ring_atoms) > 1 else None

    # Build the molecular graph
    G = nx.Graph()
    for _, row in bonds_df.iterrows():
        a1, a2 = int(row[0]), int(row[1])
        G.add_edge(a1, a2)

    # Find all simple cycles and filter 6-membered rings that include atom1_idx
    cycles = nx.cycle_basis(G)
    benzene_rings = [cycle for cycle in cycles if len(cycle) == 6 and atom1_idx in cycle]

    if not benzene_rings:
        print("No benzene ring found.")
        print(f"Debug: Found {len(benzene_rings)} benzene rings")
        print(bonds_df)
        # Debug information about each ring
        for i, ring in enumerate(benzene_rings):
            print(f"Debug: Ring {i+1}: {sorted(ring)}")
            
        # Check if atom2_idx was provided and is expected to be in a ring
        if atom2_idx is not None:
            rings_with_atom2 = [i for i, ring in enumerate(benzene_rings) if atom2_idx in ring]
            print(f"Debug: Rings containing atom {atom2_idx}: {rings_with_atom2}")
            
            # Find rings containing both atoms (if atom2_idx was specified)
            rings_with_both = [i for i, ring in enumerate(benzene_rings) 
                    if atom1_idx in ring and atom2_idx in ring]
            if rings_with_both:
                print(f"Debug: Rings containing both atoms {atom1_idx} and {atom2_idx}: {rings_with_both}")
            else:
                print(f"Debug: No rings found containing both atoms {atom1_idx} and {atom2_idx}")
        
        return None

  

    # Detect fused rings: those sharing exactly two atoms
    if len(benzene_rings) > 1:
        print("\nDetected fused benzene ring pairs:")
        for i in range(len(benzene_rings)):
            for j in range(i + 1, len(benzene_rings)):
                shared = set(benzene_rings[i]).intersection(benzene_rings[j])
                if len(shared) == 2:
                    print(f"  Ring1: {benzene_rings[i]}\n  Ring2: {benzene_rings[j]}\n  Shared atoms: {sorted(shared)}\n")
                # take the second ring if it exists

    # fix for future
    selected_ring = benzene_rings[1] if len(benzene_rings) > 1 else benzene_rings[0]
    return (
        selected_ring[3],
        selected_ring[0],
        selected_ring[1],
        selected_ring[-1],
        selected_ring[2],
        selected_ring[4]
    )
