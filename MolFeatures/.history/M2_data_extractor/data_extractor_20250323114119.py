# import pandas as pd
# import numpy as np
# import os
# import sys
# import math
# from enum import Enum
# import igraph as ig
# from .gaussian_handler import feather_file_handler
# from typing import *
# from ..utils import visualize
# import warnings
# from scipy.spatial.distance import pdist, squareform
# from ..utils import help_functions
# from ..Mol_align import renumbering
# from sklearn.preprocessing import MinMaxScaler
# from morfeus import Sterimol, read_xyz

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

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
# Now you can import from the parent directory
    from gaussian_handler import feather_file_handler
    from utils import visualize, help_functions

except:
    from .gaussian_handler import feather_file_handler
    from ..utils import visualize
    from ..utils import help_functions

warnings.filterwarnings("ignore", category=RuntimeWarning)

class GeneralConstants(Enum):
    """
    Holds constants for calculations and conversions
    1. covalent radii from Alvarez (2008) DOI: 10.1039/b801115j
    2. atomic numbers
    2. atomic weights
    """
    
    PYYKKO_RADII= {
            'H': 0.31, 'He': 0.28, 'Li': 1.28,
            'Be': 0.96, 'B': 0.84, 'C': 0.76, 
            'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
            'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 
            'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
            'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 
            'V': 1.53, 'Cr': 1.39, 'Mn': 1.61, 'Fe': 1.52, 
            'Co': 1.50, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 
            'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 
            'Br': 1.20, 'Kr': 1.16, 'Rb': 2.20, 'Sr': 1.95,
            'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54,
            'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39,
            'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39,
            'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40,
            'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04,
            'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98,
            'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92,
            'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90, 'Yb': 1.87,
            'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.70, 'W': 1.62,
            'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36,
            'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46,
            'Bi': 1.48, 'Po': 1.40, 'At': 1.50, 'Rn': 1.50, 
            'Fr': 2.60, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06,
            'Pa': 2.00, 'U': 1.96, 'Np': 1.90, 'Pu': 1.87,
            'Am': 1.80, 'Cm': 1.69
    }
    
    BONDI_RADII={
        'H': 1.10, 'C': 1.70, 'F': 1.47,
        'S': 1.80, 'B': 1.92, 'I': 1.98, 
        'N': 1.55, 'O': 1.52, 'Co': 2.00, 
        'Br': 1.83, 'Si': 2.10,'Ni': 2.00,
        'P': 1.80, 'Cl': 1.75, 
    }
    CPK_RADII = {
    'C': 1.50,
    'C3': 1.60,
    'C6/N6': 1.70,
    'H': 1.00,
    'N': 1.50,
    'N4': 1.45,
    'O': 1.35,
    'O2': 1.35,
    'P': 1.40,
    'S': 1.70,
    'S1': 1.00,
    'F': 1.95,
    'Cl': 1.80,
    'S4': 1.40,
    'Br': 1.95,
    'I': 2.15,
    'X': 1.92
}


    REGULAR_BOND_TYPE = {

        'O.2': 'O', 'N.2': 'N', 'S.3': 'S',
        'O.3': 'O', 'N.1': 'N', 'S.O2': 'S',
        'O.co2': 'O', 'N.3': 'N', 'P.3': 'P',
        'C.1': 'C', 'N.ar': 'N',
        'C.2': 'C', 'N.am': 'N',
        "C.cat": 'C', 'N.pl3': 'N',
        'C.3': 'C', 'N.4': 'N',
        'C.ar': 'C', 'S.2': 'S',
    }

    BOND_TYPE={
        
        'O.2':'O2', 'N.2':'C6/N6','S.3':'S4',
        'O.3':'O', 'N.1':'N', 'S.O2':'S',
        'O.co2':'O', 'N.3':'C6/N6','P.3':'P',
        'C.1':'C', 'N.ar':'C6/N6',
        'C.2':'C3', 'N.am':'C6/N6',
        "C.cat":'C3', 'N.pl3':'C6/N6',
        'C.3':'C', 'N.4':'N4',
        'C.ar':'C6/N6', 'S.2':'S','H':'H' 
        }
    
    ATOMIC_NUMBERS ={
    '1':'H', '5':'B', '6':'C', '7':'N', '8':'O', '9':'F', '14':'Si',
             '15':'P', '16':'S', '17':'Cl', '35':'Br', '53':'I', '27':'Co', '28':'Ni'}
        

    ATOMIC_WEIGHTS = {
            'H' : 1.008,'He' : 4.003, 'Li' : 6.941, 'Be' : 9.012,
            'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,
            'F' : 18.998, 'Ne' : 20.180, 'Na' : 22.990, 'Mg' : 24.305,
            'Al' : 26.982, 'Si' : 28.086, 'P' : 30.974, 'S' : 32.066,
            'Cl' : 35.453, 'Ar' : 39.948, 'K' : 39.098, 'Ca' : 40.078,
            'Sc' : 44.956, 'Ti' : 47.867, 'V' : 50.942, 'Cr' : 51.996,
            'Mn' : 54.938, 'Fe' : 55.845, 'Co' : 58.933, 'Ni' : 58.693,
            'Cu' : 63.546, 'Zn' : 65.38, 'Ga' : 69.723, 'Ge' : 72.631,
            'As' : 74.922, 'Se' : 78.971, 'Br' : 79.904, 'Kr' : 84.798,
            'Rb' : 84.468, 'Sr' : 87.62, 'Y' : 88.906, 'Zr' : 91.224,
            'Nb' : 92.906, 'Mo' : 95.95, 'Tc' : 98.907, 'Ru' : 101.07,
            'Rh' : 102.906, 'Pd' : 106.42, 'Ag' : 107.868, 'Cd' : 112.414,
            'In' : 114.818, 'Sn' : 118.711, 'Sb' : 121.760, 'Te' : 126.7,
            'I' : 126.904, 'Xe' : 131.294, 'Cs' : 132.905, 'Ba' : 137.328,
            'La' : 138.905, 'Ce' : 140.116, 'Pr' : 140.908, 'Nd' : 144.243,
            'Pm' : 144.913, 'Sm' : 150.36, 'Eu' : 151.964, 'Gd' : 157.25,
            'Tb' : 158.925, 'Dy': 162.500, 'Ho' : 164.930, 'Er' : 167.259,
            'Tm' : 168.934, 'Yb' : 173.055, 'Lu' : 174.967, 'Hf' : 178.49,
            'Ta' : 180.948, 'W' : 183.84, 'Re' : 186.207, 'Os' : 190.23,
            'Ir' : 192.217, 'Pt' : 195.085, 'Au' : 196.967, 'Hg' : 200.592,
            'Tl' : 204.383, 'Pb' : 207.2, 'Bi' : 208.980, 'Po' : 208.982,
            'At' : 209.987, 'Rn' : 222.081, 'Fr' : 223.020, 'Ra' : 226.025,
            'Ac' : 227.028, 'Th' : 232.038, 'Pa' : 231.036, 'U' : 238.029,
            'Np' : 237, 'Pu' : 244, 'Am' : 243, 'Cm' : 247
    }

import numpy.typing as npt

def compare_cosine_distance_matrices(matrix1, matrix2):
    def cosine_distance(matrix):
        # Calculate norms
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = np.finfo(float).eps
        # Normalize the rows to unit length
        norm_matrix = matrix / norms
        # Use dot product to find cosine similarity and subtract from 1 to get cosine distance
        similarity = np.dot(norm_matrix, norm_matrix.T)
        return 1 - similarity

    # Calculate the cosine distance matrices
    matrix1_distances = cosine_distance(matrix1)
    matrix2_distances = cosine_distance(matrix2)
    
    # Calculate differences between the two distance matrices
    differences = matrix1_distances - matrix2_distances
    average_difference = np.mean(np.abs(differences))

    return average_difference




def generate_circle(center_x, center_y, radius, n_points=20):
    """
    Generate circle coordinates given a center and radius.
    Returns a DataFrame with columns 'x' and 'y'.
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return np.column_stack((x, y))


def adjust_indices(element):
    
    if isinstance(element, list):
        return [adjust_indices(sub_element) for sub_element in element]
    elif isinstance(element, int):
        return element - 1
    elif isinstance(element, np.ndarray):
        return element - 1
    elif isinstance(element, np.int64):
        return element - 1
    else:
        raise ValueError("Unsupported element type")




def calc_angle(p1: npt.ArrayLike, p2: npt.ArrayLike, degrees: bool=False) -> float: ###works, name in R: 'angle' , radians
    dot_product=np.dot(p1, p2)
    norm_p1=np.linalg.norm(p1)
    norm_p2=np.linalg.norm(p2)
    thetha=np.arccos(dot_product/(norm_p1*norm_p2))
    if degrees:
        thetha=np.degrees(thetha)   
    return thetha
    
def calc_new_base_atoms(coordinates_array: npt.ArrayLike, atom_indices: npt.ArrayLike):  #help function for calc_coordinates_transformation
    """
    a function that calculates the new base atoms for the transformation of the coordinates.
    optional: if the atom_indices[0] is list, compute the new origin as the middle of the first atoms.
    """
   
    if isinstance(atom_indices[0], list):
        new_origin=np.mean(coordinates_array[atom_indices[0]], axis=0)
    else:
        new_origin=coordinates_array[atom_indices[0]]
    new_y=(coordinates_array[atom_indices[1]]-new_origin)/np.linalg.norm((coordinates_array[atom_indices[1]]-new_origin))
    coplane=((coordinates_array[atom_indices[2]]-new_origin)/np.linalg.norm((coordinates_array[atom_indices[2]]-new_origin)+0.00000001))
    
    return (new_origin,new_y,coplane)

def np_cross_and_vstack(plane_1, plane_2):
    cross_plane=np.cross(plane_1, plane_2)
    united_results=np.vstack([plane_1, plane_2, cross_plane])
    return united_results

from numpy.typing import ArrayLike

def calc_basis_vector(origin, y: ArrayLike, coplane: ArrayLike):
    """
    Calculate the new basis vector.
    
    Parameters
    ----------
    origin : array-like
        The origin of the new basis.
    y : array-like
        The new basis's y direction.
    coplane : array-like
        A vector coplanar with the new y direction.
    
    Returns
    -------
    new_basis : np.array
        The computed new basis matrix.
    """
    coef_mat = np_cross_and_vstack(coplane, y)
    angle_new_y_coplane = calc_angle(coplane, y)
    cop_ang_x = angle_new_y_coplane - (np.pi/2)
    result_vector = [np.cos(cop_ang_x), 0, 0]
    new_x, _, _, _ = np.linalg.lstsq(coef_mat, result_vector, rcond=None)
    new_basis = np_cross_and_vstack(new_x, y)
    return new_basis

def transform_row(row_array, new_basis, new_origin, round_digits):
    """
    Transform a single row of coordinates.
    
    Parameters
    ----------
    row_array : array-like
        A row of coordinates.
    new_basis : array-like
        New basis matrix.
    new_origin : array-like
        The new origin.
    round_digits : int
        Number of decimal places to round to.
    
    Returns
    -------
    np.array
        The transformed coordinates.
    """
    row_array = np.squeeze(row_array)
    translocated_row = row_array - new_origin
    ## make sure or change to shape (3,) if needed
    if translocated_row.shape != (3,):
        try:
            translocated_row = np.reshape(translocated_row, (3,))
        except Exception as e:
            print("transform_row: Error reshaping translocated_row:", e)
    
    result = np.dot(new_basis, translocated_row).round(round_digits)
    return result

def calc_coordinates_transformation(coordinates_array: ArrayLike, base_atoms_indices: ArrayLike, round_digits: int = 4, origin: ArrayLike = None) -> ArrayLike:
    """
    Transform a coordinates array to a new basis defined by the atoms in base_atoms_indices.
    
    Parameters
    ----------
    coordinates_array : np.array
        The original xyz molecule coordinates.
    base_atoms_indices : list or array-like
        Indices of atoms to define the new basis.
    round_digits : int, optional
        Number of digits to round the result.
    origin : array-like, optional
        A new origin for the transformation. If None, the first base atom is used.
    
    Returns
    -------
    transformed_coordinates : np.array
        The transformed coordinates.
    """
    indices = adjust_indices(base_atoms_indices)
    # Calculate new basis using helper function calc_new_base_atoms (assumed to return origin, y, and coplane)
    new_basis = calc_basis_vector(*calc_new_base_atoms(coordinates_array, indices))
    if origin is None:
        new_origin = coordinates_array[indices[0]]
    else:
        new_origin = origin
    transformed_coordinates = np.apply_along_axis(lambda x: transform_row(x, new_basis, new_origin, round_digits), 1, coordinates_array)
  
    return transformed_coordinates

def preform_coordination_transformation(xyz_df, indices=None, origin=None):
    """
    Perform a coordination transformation on the xyz DataFrame.
    
    Parameters
    ----------
    xyz_df : pd.DataFrame
        DataFrame containing columns 'x', 'y', 'z'.
    indices : array-like, optional
        Atom indices to use for the new basis. If None, default indices [1,2,3] are used.
    
    Returns
    -------
    xyz_copy : pd.DataFrame
        DataFrame with transformed coordinates.
    """
    xyz_copy = xyz_df.copy()
    coordinates = np.array(xyz_copy[['x', 'y', 'z']].values)
    if indices is None:
        transformed = calc_coordinates_transformation(coordinates, [1,2,3], origin=origin)
    else:
        transformed = calc_coordinates_transformation(coordinates, indices, origin=origin)
    
    xyz_copy[['x', 'y', 'z']] = transformed
    return xyz_copy


def calc_npa_charges(coordinates_array: npt.ArrayLike,charge_array: npt.ArrayLike):##added option for subunits
    """
    a function that recives coordinates and npa charges, transform the coordinates
    by the new base atoms and calculates the dipole in each axis
    
    Parameters
    ---------
    coordinates_array: np.array
        contains x y z atom coordinates
        
    charge_array: np.array
        array of npa charges
    base_atoms_indices:list
        3/4 atom indices for coordinates transformation
        
    optional-sub_atoms:list
        calculate npa charges from a set of sub_atoms instead of all atoms.       
    Returns:
    -------
    dipole_df=calc_npa_charges(coordinates_array,charges,base_atoms_indices,sub_atoms)
    Output:
    dipole_df : pd.DataFrame
        output:            dip_x     dip_y     dip_z     total
                       0  0.097437 -0.611775  0.559625  0.834831
    """


    dipole_xyz = np.vstack([(row[0] * row[1])for row in
                            list(zip(coordinates_array, charge_array))])
    dipole_vector=np.sum(dipole_xyz,axis=0)
    array_dipole=np.hstack([dipole_vector,np.linalg.norm(dipole_vector)])
    dipole_df=pd.DataFrame(array_dipole,index=help_functions.XYZConstants.DIPOLE_COLUMNS.value).T
 
    return dipole_df

def calc_dipole_gaussian(coordinates_array, gauss_dipole_array, base_atoms_indices, origin ):
    """
    a function that recives coordinates and gaussian dipole, transform the coordinates
    by the new base atoms and calculates the dipole in each axis
    """

   

    indices=adjust_indices(base_atoms_indices)
    if origin is None:
        new_origin = coordinates_array[indices[0]]
    else:
        new_origin = origin
    
    # Recenter the coordinates array using the new origin
    recentered_coords = coordinates_array - new_origin
    basis_vector=calc_basis_vector(*calc_new_base_atoms(recentered_coords, indices))
    gauss_dipole_array = [np.concatenate((np.matmul(basis_vector, gauss_dipole_array[0, 0:3]), [gauss_dipole_array[0, 3]]))]
    dipole_df=pd.DataFrame(gauss_dipole_array,columns=['dipole_x','dipole_y','dipole_z','total'])

    return dipole_df

def check_imaginary_frequency(info_df):##return True if no complex frequency, called ground.state in R
        bool_imaginary=not any([isinstance(frequency, complex) for frequency in info_df['Frequency']])
        return bool_imaginary



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





def get_bonds_vector_for_calc_angle(coordinates_array,atoms_indices): ##for calc_angle_between_atoms

    indices=adjust_indices(atoms_indices)#t
    augmented_indices=[indices[0],indices[1],indices[1],indices[2]]
    if len(indices)==4:
        augmented_indices.extend([indices[2],indices[3]])
    indices_pairs=list(zip(augmented_indices[::2],augmented_indices[1::2]))
  
    bond_vector=indices_to_coordinates_vector(coordinates_array,indices_pairs)
    return bond_vector
  

def calc_angle_between_atoms(coordinates_array,atoms_indices): #gets a list of atom indices
    """
    a function that gets 3/4 atom indices, and returns the angle between thos atoms.
    Parameters
    ----------
    coordinates_array: np.array
        contains x y z atom coordinates
        
    atoms_indices- list of ints
        a list of atom indices to calculate the angle between- [2,3,4]
   
    Returns
    -------
    angle: float
        the bond angle between the atoms
    """
    bonds_list=get_bonds_vector_for_calc_angle(coordinates_array,atoms_indices)
    if len(atoms_indices)==3:
 
        angle=calc_angle(bonds_list[0], bonds_list[1]*(-1), degrees=True)
    else:
        first_cross=np.cross(bonds_list[0],bonds_list[1]*(-1))
        second_cross=np.cross(bonds_list[2]*(-1),bonds_list[1]*(-1)) 
        angle=calc_angle(first_cross, second_cross, degrees=True)
    return angle

def get_angle_df(coordinates_array, atom_indices):
    """
    a function that gets a list of atom indices, and returns a dataframe of angles between thos atoms.
    Parameters
    ----------
    coordinates_array: np.array
        contains x y z atom coordinates
    
    atom_indices- list of lists of ints
        a list of atom indices to calculate the angle between- [[2,3,4],[2,3,4,5]]
    """
 
    if isinstance(atom_indices, list) and all(isinstance(elem, list) for elem in atom_indices):
        indices_list=['angle_{}'.format(index) if len(index)==3 else 'dihedral_{}'.format(index) for index in atom_indices]
        angle_list=[calc_angle_between_atoms(coordinates_array,index) for index in atom_indices]
        return pd.DataFrame(angle_list,index=indices_list)
    else:
        indices_list=['angle_{}'.format(atom_indices) if len(atom_indices)==3 else 'dihedral_{}'.format(atom_indices)]
        angle=[calc_angle_between_atoms(coordinates_array,atom_indices)]
        return pd.DataFrame(angle,index=indices_list)


def calc_single_bond_length(coordinates_array,atom_indices):
    """
    a function that gets 2 atom indices, and returns the distance between thos atoms.
    Parameters
    ----------
    coordinates_array: np.array
        contains x y z atom coordinates
        
    atom_indices- list of ints
        a list of atom indices to calculate the distance between- [2,3]
   
    Returns
    -------
    distance: float
        the bond distance between the atoms
    """
    indices=adjust_indices(atom_indices)
    distance=np.linalg.norm(coordinates_array[indices[0]]-coordinates_array[indices[1]])
    return distance

def calc_bonds_length(coordinates_array,atom_pairs): 
    """
    a function that calculates the distance between each pair of atoms.
    help function for molecule class
    
    Parameters
    ----------
    coordinates_array: np.array
        xyz coordinates 
        
    atom_pairs : iterable
        list containing atom pairs-(([2,3],[4,5]))
        
    Returns
    -------
    pairs_df : dataframe
        distance between each pair 
        
        Output:
                               0
    bond length[2, 3]            1.525692
    bond length[4, 5]            2.881145
    
    """
    # Check the order of bond list and pairs
    bond_list=[calc_single_bond_length(coordinates_array,pair) for pair in atom_pairs]
    pairs=adjust_indices(atom_pairs)
    index=[('bond_length')+str(pair) for pair in pairs]
    pairs_df=pd.DataFrame(bond_list,index=index)
    return pairs_df



def direction_atoms_for_sterimol(bonds_df,base_atoms)->list: #help function for sterinol
    """
    a function that return the base atom indices for coordination transformation according to the bonded atoms.
    you can insert two atom indicess-[1,2] output [1,2,8] or the second bonded atom
    if the first one repeats-[1,2,1] output [1,2,3]
    """
    
    base_atoms_copy=base_atoms[0:2]
    origin,direction=base_atoms[0],base_atoms[1]
    bonds_df = bonds_df[~((bonds_df[0] == origin) & (bonds_df[1] == direction)) & 
                              ~((bonds_df[0] == direction) & (bonds_df[1] == origin))]
    
    try :
        base_atoms[2]==origin
        if(any(bonds_df[0]==direction)):
            # take the second atom in the bond where the first equeal to the direction, second option
            base_atoms_copy[2]=int(bonds_df[(bonds_df[0]==direction)][1].iloc[1])
        else:
            # take the first atom in the bond where the first equeal to the direction, second option
            base_atoms_copy[2]=int(bonds_df[(bonds_df[1]==direction)][0].iloc[1])
            return base_atoms_copy
    except: 

        for _, row in bonds_df.iterrows():

            if row[1] == direction:
                base_atoms_copy.append(row[0])
                return base_atoms_copy
          
        for _, row in bonds_df.iterrows():
            
            if row[0] == direction:
                base_atoms_copy.append(row[1])
                return base_atoms_copy
                # return base_atoms_copy
        
        
    

def get_molecule_connections(bonds_df,source,direction):
    graph=ig.Graph.DataFrame(edges=bonds_df,directed=True)
    paths=graph.get_all_simple_paths(v=source,mode='all')
    with_direction=[path for path in paths if (direction in path)]
    longest_path=np.unique(help_functions.flatten_list(with_direction))
   
    return longest_path




def get_specific_bonded_atoms_df(bonds_df,longest_path,coordinates_df):
    """
    a function that returns a dataframe of the atoms that are bonded in the longest path.
    bonded_atoms_df: dataframe
        the atom type and the index of each bond.

       atom_1 atom_2 index_1 index_2
0       C      N       1       4
1       C      N       1       5
2       C      N       2       3
    """
    if longest_path is not None:
        edited_bonds_df=bonds_df[(bonds_df.isin(longest_path))].dropna().reset_index(drop=True)
    else:
        edited_bonds_df=bonds_df
    bonds_array=(np.array(edited_bonds_df)-1).astype(int) # adjust indices? 
    atom_bonds=np.vstack([(coordinates_df.iloc[bond]['atom'].values) for bond in bonds_array]).reshape(-1,2)
    bonded_atoms_df=(pd.concat([pd.DataFrame(atom_bonds),edited_bonds_df],axis=1))
    bonded_atoms_df.columns=[help_functions.XYZConstants.BONDED_COLUMNS.value]
    return bonded_atoms_df


def remove_atom_bonds(bonded_atoms_df,atom_remove='H'):
    atom_bonds_array=np.array(bonded_atoms_df)
    delete_rows_left=np.where(atom_bonds_array[:,0]==atom_remove)[0] #itterrow [0] is index [1] are the values
    delete_rows_right=np.where(atom_bonds_array[:,1]==atom_remove)[0]
    atoms_to_delete=np.concatenate((delete_rows_left,delete_rows_right))
    new_bonded_atoms_df=bonded_atoms_df.drop((atoms_to_delete),axis=0)
    return new_bonded_atoms_df



def extract_connectivity(xyz_df, threshhold_distance=1.82):
    coordinates=np.array(xyz_df[['x','y','z']].values)
    atoms_symbol=np.array(xyz_df['atom'].values)
    # compute the pairwise distances between the points
    distances = pdist(coordinates)
    # convert the flat array of distances into a distance matrix
    dist_matrix = squareform(distances)
    dist_df=pd.DataFrame(dist_matrix).stack().reset_index()
    dist_df.columns = ['a1', 'a2', 'value']
    dist_df['first_atom']=[atoms_symbol[i] for i in dist_df['a1']]
    dist_df['second_atom']=[atoms_symbol[i] for i in dist_df['a2']]
    remove_list=[]
    dist_array=np.array(dist_df)
    remove_list = []
    special_atoms = {'Cl', 'Br', 'F', 'I'}
    for idx, row in enumerate(dist_array):
        remove_flag = False
      
        if row[0] == row[1]:
            remove_flag = True
          
        if ((row[3] == 'H') & (row[4] not in help_functions.XYZConstants.NOF_ATOMS.value)):
            remove_flag = True
           
        if ((row[3] == 'H') & (row[4] == 'H')):
            remove_flag = True
          
        if (((row[3] == 'H') | (row[4] == 'H')) & (row[2] >= 1.5)):
            remove_flag = True
           
        if ((row[2] >= threshhold_distance) | (row[2] == 0)):
            remove_flag = True
        # Special condition for Cl and Br atoms, if the distance is greater than 2, don't remove
        if (row[3] in special_atoms or row[4] in special_atoms) and (row[2] > 1.8 and row[2] < 2.6):
            remove_flag = False  # Don't remove this bond if one atom is Cl, Br, F, or I
        

        if remove_flag:
            remove_list.append(idx)
    dist_df=dist_df.drop(remove_list)
    dist_df[['min_col', 'max_col']] = pd.DataFrame(np.sort(dist_df[['a1', 'a2']], axis=1), index=dist_df.index)
    dist_df = dist_df.drop(columns=['a1', 'a2']).rename(columns={'min_col': 0, 'max_col': 1})
    dist_df = dist_df.drop_duplicates(subset=[0, 1])
    return pd.DataFrame(dist_df[[0,1]]+1) ## should be +1 

def get_center_of_mass(xyz_df):
    coordinates=np.array(xyz_df[['x','y','z']].values,dtype=float)
    atoms_symbol=np.array(xyz_df['atom'].values)
    masses=np.array([GeneralConstants.ATOMIC_WEIGHTS.value[symbol] for symbol in atoms_symbol])
    center_of_mass=np.sum(coordinates*masses[:,None],axis=0)/np.sum(masses)
    return center_of_mass

def get_closest_atom_to_center(xyz_df,center_of_mass):
    distances = np.sqrt((xyz_df['x'] - center_of_mass[0]) ** 2 + (xyz_df['y'] - center_of_mass[1]) ** 2 + (xyz_df['z'] - center_of_mass[2]) ** 2)
    idx_closest = np.argmin(distances)
    center_atom = xyz_df.loc[idx_closest]
    return center_atom

def get_sterimol_base_atoms(center_atom, bonds_df):
    
    center_atom_id=int(center_atom.name)+1
    base_atoms = [center_atom_id]
    if (any(bonds_df[0] == center_atom_id)):
        base_atoms.append(int(bonds_df[(bonds_df[0]==center_atom_id)][1].iloc[0]))
    else:
        base_atoms.append(int(bonds_df[(bonds_df[1]==center_atom_id)][0].iloc[0]))
    
    return base_atoms

def center_substructure(coordinates_array,atom_indices):
    atom_indices=adjust_indices(atom_indices)
    substructure=coordinates_array[atom_indices]
    center_substructure=np.mean(substructure,axis=0)
    return center_substructure

def nob_atype(xyz_df, bonds_df):
    
    symbols = xyz_df['atom'].values
    
    list_results=[]
    for index,symbol in enumerate(symbols):
        index+=1
        nob = bonds_df[(bonds_df[0] == index) | (bonds_df[1] == index)].shape[0]
        if symbol == 'H':
            result = 'H'
        elif symbol == 'F':
            result = 'F'
        elif symbol == 'P':
            result = 'P'
        elif symbol == 'Cl':
            result = 'Cl'
        elif symbol == 'Br':
            result = 'Br'
        elif symbol == 'I':
            result = 'I'
        elif symbol == 'O':
            if nob < 1.5:
                result = 'O2'
            elif nob > 1.5:
                result = 'O'
        elif symbol == 'S':
            if nob < 2.5:
                result = 'S'
            elif 2.5 < nob < 5.5:
                result = 'S4'
            elif nob > 5.5:
                result = 'S1'
        elif symbol == 'N':
            if nob < 2.5:
                result = 'C6/N6'
            elif nob > 2.5:
                result = 'N'
        elif symbol == 'C':
            if nob < 2.5:
                result = 'C3'
            elif 2.5 < nob < 3.5:
                result = 'C6/N6'
            elif nob > 3.5:
                result = 'C'
        else:
            result = 'X'

        list_results.append(result)

    return list_results

def get_sterimol_indices(coordinates,bonds_df):
    center=get_center_of_mass(coordinates)
    center_atom=get_closest_atom_to_center(coordinates,center)
    base_atoms=get_sterimol_base_atoms(center_atom,bonds_df)
    return base_atoms

def filter_atoms_for_sterimol(bonded_atoms_df,coordinates_df):
    """
    a function that filter out NOF bonds and H bonds and returns
     a dataframe of the molecule coordinates without them.
    """
    allowed_bonds_indices= pd.concat([bonded_atoms_df['index_1'],bonded_atoms_df['index_2']],axis=1).reset_index(drop=True)
    atom_filter=adjust_indices(np.unique([atom for sublist in allowed_bonds_indices.values.tolist() for atom in sublist]))
    edited_coordinates_df=coordinates_df.loc[atom_filter].reset_index(drop=True)
   
    return edited_coordinates_df


def get_extended_df_for_sterimol(coordinates_df, bonds_df, radii='CPK'):
    """
    A function that adds information to the regular coordinates_df

    Parameters
    ----------
    coordinates_df : dataframe
    bond_type : str
        The bond type of the molecule
    radii : str, optional
        The type of radii to use ('bondi' or 'CPK'), by default 'bondi'

    Returns
    -------
    dataframe
        The extended dataframe with additional columns

    """
    
    bond_type_map_regular = GeneralConstants.REGULAR_BOND_TYPE.value
    bond_type_map=GeneralConstants.BOND_TYPE.value
    ## if radius is cpk mapping should be done on atype, else on atom
    radii_map = GeneralConstants.CPK_RADII.value if radii == 'CPK' else GeneralConstants.BONDI_RADII.value
    
    df = coordinates_df.copy()  # make a copy of the dataframe to avoid modifying the original
    
    
    
    if radii == 'bondi':
        df['atype']=df['atom']
    else:
        df['atype']=nob_atype(coordinates_df, bonds_df)

    df['magnitude'] = calc_magnitude_from_coordinates_array(df[['x', 'y']].astype(float))
    
    df['radius'] = df['atype'].map(radii_map)
    df['B5'] = df['radius'] + df['magnitude']
    df['L'] = df['y'] + df['radius']
    return df


def get_transfomed_plane_for_sterimol(plane,degree):
    """
    a function that gets a plane and rotates it by a given degree
    in the case of sterimol the plane is the x,z plane.
    Parameters:
    ----------
    plane : np.array
        [x,z] plane of the molecule coordinates.
        example:
            [-0.6868 -0.4964]
    degree : float
    """
    
    cos_deg=np.cos(degree*(np.pi/180))
    sin_deg=np.sin(degree*(np.pi/180))
    rot_matrix=np.array([[cos_deg,-1*sin_deg],[sin_deg,cos_deg]])
    transformed_plane=np.vstack([np.matmul(rot_matrix,row) for row in plane]).round(4)
    avs=np.abs([max(transformed_plane[:,0]),min(transformed_plane[:,0]), 
                    max(transformed_plane[:,1]),min(transformed_plane[:,1])])
    
    ## return the inversed rotation matrix to transform the plane back

    return transformed_plane


def calc_B1(transformed_plane,avs,edited_coordinates_df,column_index):
    """
    Parameters
    ----------
    transformed_plane : np.array
        [x,z] plane of the molecule coordinates.
        example:
            [-0.6868 -0.4964]
            [-0.7384 -0.5135]
            [-0.3759 -0.271 ]
            [-1.1046 -0.8966]
            [ 0.6763  0.5885]
    avs : list
        the max & min of the [x,z] columns from the transformed_plane.
        example:[0.6763, -1.1046, 0.5885, -0.8966
                 ]
    edited_coordinates_df : TYPE
        DESCRIPTION.
    column_index : int
        0 or 1 depending- being used for transformed plane.
    """
    
    ## get the index of the min value in the column compared to the avs.min
    idx=np.where(np.isclose(np.abs(transformed_plane[:,column_index]),(avs.min())))[0][0]  ## .round(4)
    # Compute number of points per substituent (assumes evenly divided)
    # print('inside calc_B1')
    # n_total = transformed_plane.shape[0]
    # n_subs = edited_coordinates_df.shape[0]
    # n_points = n_total // n_subs if n_subs > 0 else 1

    # # Print debug info
    # print("calc_B1: transformed_plane.shape =", transformed_plane.shape)
    # print("calc_B1: len(extended_df) =", n_subs)
    # print("calc_B1: n_points per substituent =", n_points)
    
    # # Find first index where the absolute value is close to the minimum of avs
    # idx = np.where(np.isclose(np.abs(transformed_plane[:, column_index]), avs.min()))[0][0]
    # # Map the plane index back to the corresponding DataFrame row
    # index_df = idx // n_points
    # print("calc_B1: raw idx =", idx, "mapped index_df =", index_df)
    # idx=index_df
    if transformed_plane[idx,column_index]<0:
       
        new_idx=np.where(np.isclose(transformed_plane[:,column_index],transformed_plane[:,column_index].min()))[0][0]
        bool_list=np.logical_and(transformed_plane[:,column_index]>=transformed_plane[new_idx,column_index],
                                 transformed_plane[:,column_index]<=transformed_plane[new_idx,column_index]+1)
        
        transformed_plane[:,column_index]=-transformed_plane[:,column_index]
    else:
     
        bool_list=np.logical_and(transformed_plane[:,column_index]>=transformed_plane[idx,column_index]-1,
                                 transformed_plane[:,column_index]<=transformed_plane[idx,column_index])
        
    against,against_loc=[],[]
    B1,B1_loc=[],[]

    ### return the part of the transformed plane that b1 is calculated from
    ### convert it back with the inverse matrix to get the original coordinates of b1 location
    
    for i in range(0,transformed_plane.shape[0]): 
        if bool_list[i]:
            against.append(np.array(transformed_plane[i,column_index]+edited_coordinates_df['radius'].iloc[i]))
            against_loc.append(edited_coordinates_df['L'].iloc[i])
        

        if len(against)>0:
        
            B1.append(max(against))
            B1_loc.append(against_loc[against.index(max(against))])
          
        else:
       
            B1.append(np.abs(transformed_plane[idx,column_index]+edited_coordinates_df['radius'].iloc[idx]))
            B1_loc.append(edited_coordinates_df['radius'].iloc[idx])
            
            
      
    return [B1,B1_loc] 

def b1s_for_loop_function(extended_df, b1s, b1s_loc, degree_list, plane, b1_planes):
    """
    For each degree in degree_list, rotate the plane and compute B1 values and the B1-B5 angle.
    Instead of returning after the first iteration, this version accumulates all results
    into a DataFrame.
    
    Parameters
    ----------
    extended_df : pd.DataFrame
        DataFrame with at least columns 'x', 'z', 'radius', 'L'.
    b1s : list
        (Unused here; kept for compatibility)
    b1s_loc : list
        (Unused here; kept for compatibility)
    degree_list : list
        List of rotation angles (in degrees) to scan.
    plane : np.array
        Array of shape (n_points_total, 2) that contains the combined circle points.
    b1_planes : list
        List to store the rotated plane from each iteration.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per degree, containing:
         - 'degree': the rotation angle,
         - 'B1': the minimum extreme value from the rotated plane,
         - 'B1_B5_angle': the angle between the B1 and B5 arrows in degrees,
         - 'b1_coords': the coordinate (as a tuple) chosen for B1,
         - 'b5_value': the distance of the farthest point (B5) from the origin.
    """
    results = []  # List to accumulate results
    
    for degree in degree_list:
        transformed_plane = get_transfomed_plane_for_sterimol(plane, degree)  # Rotate the plane
        max_x = np.max(transformed_plane[:, 0])
        min_x = np.min(transformed_plane[:, 0])
        max_y = np.max(transformed_plane[:, 1])
        min_y = np.min(transformed_plane[:, 1])
        avs = np.abs([max_x, min_x, max_y, min_y])
        min_val = np.min(avs)
        min_index = np.argmin(avs)
        
        # Mimic R's switch to pick the B1 coordinate:
        if min_index == 0:
            b1_coords = (max_x, 0)
        elif min_index == 1:
            b1_coords = (min_x, 0)
        elif min_index == 2:
            b1_coords = (0, max_y)
        else:
            b1_coords = (0, min_y)
        
        # Determine B5 as the farthest point from the origin.
        norms_sq = np.sum(transformed_plane**2, axis=1)
        b5_index = np.argmax(norms_sq)
        b5_point = transformed_plane[b5_index]
        b5_value = np.linalg.norm(b5_point)
        
        # Calculate angles for the arrows.
        angle_b1 = np.arctan2(b1_coords[1], b1_coords[0]) % (2*np.pi)
        angle_b5 = np.arctan2(b5_point[1], b5_point[0]) % (2*np.pi)
        angle_diff = abs(angle_b5 - angle_b1)
        if angle_diff > np.pi:
            angle_diff = 2*np.pi - angle_diff
        # Convert the angle difference to degrees.
        B1_B5 = np.degrees(angle_diff)
        B1 = min_val
        
        # Save the transformed plane for this iteration.
        b1_planes.append(transformed_plane)
        
        # Accumulate the result for this degree.
        results.append({
            'degree': degree,
            'B1': B1,
            'B1_B5_angle': B1_B5,
            'b1_coords': b1_coords,
            'b5_value': b5_value,
            'plane': transformed_plane
        })
    
    # Create a DataFrame from the accumulated results.
    sterimol_df = pd.DataFrame(results)
    return sterimol_df

        # avs=np.abs([max(transformed_plane[:,0]),min(transformed_plane[:,0]), 
        #             max(transformed_plane[:,1]),min(transformed_plane[:,1])])
        
        # if min(avs) == 0:
      
        #     min_avs_indices = np.where(avs == min(avs))[0]
        #     if any(index in [0, 1] for index in min_avs_indices):
             
        #         tc = np.round(transformed_plane, 4)
        #         B1 = max(extended_df['radius'].iloc[np.where(tc[:, 0] == 0)])
        #         B1_loc = extended_df['L'].iloc[np.argmax(extended_df['radius'].iloc[np.where(tc[:, 0] == 0)])]
        #         b1s.append(B1)
        #         b1s_loc.append(B1_loc)
        #         continue  # Skip the rest of the loop

        #     elif any(index in [2, 3] for index in min_avs_indices):
               
        #         tc = np.round(transformed_plane, 4)
        #         B1 = max(extended_df['radius'].iloc[np.where(tc[:, 1] == 0)])
        #         B1_loc = extended_df['L'].iloc[np.argmax(extended_df['radius'].iloc[np.where(tc[:, 1] == 0)])]
        #         b1s.append(B1)
        #         b1s_loc.append(B1_loc)
        #         continue

        # if np.where(avs==avs.min())[0][0] in [0,1]:
            
        #     B1,B1_loc=calc_B1(transformed_plane,avs,extended_df,0)
        # elif np.where(avs==avs.min())[0][0] in [2,3]:
         
        #     B1,B1_loc=calc_B1(transformed_plane,avs,extended_df,1)

        # b1s.append(np.unique(np.vstack(B1)).max())####check
        # b1s_loc.append(np.unique(np.vstack(B1_loc)).max())
        # b1_planes.append(transformed_plane)
    
def get_b1s_list(extended_df, scans=90//5,plot_result=False):
    """
    Calculate B1 values by scanning over a range of rotation angles.
    Instead of using only the center points, this version generates circle points
    for each substituent from extended_df and then stacks them into one plane.
    
    Parameters
    ----------
    extended_df : pd.DataFrame
        DataFrame with at least the columns: 'x', 'z', 'radius', 'L'.
    scans : int, optional
        Degree step for the initial scan.
        
    Returns
    -------
    tuple
        (np.array of B1 values, np.array of B1 location values, list of rotated planes)
    """
    b1s, b1s_loc, b1_planes = [], [], []
    degree_list = list(range(18, 108, scans))
    
    # Generate circles for each substituent and combine them.
    circles = []
    for idx, row in extended_df.iterrows():
        circle_points = generate_circle(row['x'], row['z'], row['radius'], n_points=100)
        circles.append(circle_points)
    plane = np.vstack(circles)  # All circle points combined.
    
    sterimol_df=b1s_for_loop_function(extended_df, b1s, b1s_loc, degree_list, plane, b1_planes)

    b1s=sterimol_df['B1']
    try:
        back_ang=degree_list[np.where(b1s==min(b1s))[0][0]]-scans   
        front_ang=degree_list[np.where(b1s==min(b1s))[0][0]]+scans
        new_degree_list=range(back_ang,front_ang+1)
    except:
        
        back_ang=degree_list[np.where(np.isclose(b1s, min(b1s), atol=1e-8))[0][0]]-scans
        front_ang=degree_list[np.where(np.isclose(b1s, min(b1s), atol=1e-8))[0][0]]+scans
        new_degree_list=range(back_ang,front_ang+1)

     
    b1s, b1s_loc, b1_planes = [], [], []
    sterimol_df=b1s_for_loop_function(extended_df, b1s, b1s_loc, list(new_degree_list), plane, b1_planes)
  
    b1s=sterimol_df['B1']
 
    b1_b5_angle=sterimol_df['B1_B5_angle']
    plane=sterimol_df['plane']
    

    return [b1s, b1_b5_angle, plane]

import matplotlib.pyplot as plt

def plot_b1_visualization(rotated_plane, extended_df, n_points=100, title="Rotated Plane Visualization"):
    """
    Visualize the rotated plane by plotting:
      - Complete circles (each generated from a substituent),
      - Dashed lines at extreme x and y values,
      - Arrows for the four extreme directions (with the B1 arrow highlighted),
      - A B5 arrow (the farthest point from the origin),
      - An arc indicating the angle between the B1 and B5 arrows.
    
    Parameters
    ----------
    rotated_plane : np.array
        Rotated plane points (stacked complete circles; shape: [n_total_points, 2]).
    extended_df : pd.DataFrame
        DataFrame with columns 'radius' and 'L' (used for annotations).
    n_points : int, optional
        Number of points per circle (default is 20).
    title : str, optional
        Title for the plot.
    """
    # Compute extreme values from all points
    max_x = np.max(rotated_plane[:, 0])
    min_x = np.min(rotated_plane[:, 0])
    max_y = np.max(rotated_plane[:, 1])
    min_y = np.min(rotated_plane[:, 1])
    avs = np.abs([max_x, min_x, max_y, min_y])
    min_val = np.min(avs)
    min_index = np.argmin(avs)
    
    # Determine B1 arrow coordinates based on the minimum extreme
    if min_index == 0:
        b1_coords = np.array([max_x, 0])
    elif min_index == 1:
        b1_coords = np.array([min_x, 0])
    elif min_index == 2:
        b1_coords = np.array([0, max_y])
    else:
        b1_coords = np.array([0, min_y])
    
    # Determine B5 as the farthest point from the origin
    norms_sq = np.sum(rotated_plane**2, axis=1)
    b5_index = np.argmax(norms_sq)
    b5_point = rotated_plane[b5_index]
    b5_value = np.linalg.norm(b5_point)
    
    # Calculate angles for the arrows
    angle_b1 = np.arctan2(b1_coords[1], b1_coords[0]) % (2 * np.pi)
    angle_b5 = np.arctan2(b5_point[1], b5_point[0]) % (2 * np.pi)
    angle_diff = abs(angle_b5 - angle_b1)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff
    angle_diff_deg = np.degrees(angle_diff)
    
    plt.figure(figsize=(8, 8))
    
    # Plot complete circles.
    n_total = rotated_plane.shape[0]
    n_circles = n_total // n_points
    for i in range(n_circles):
        circle_points = rotated_plane[i * n_points:(i + 1) * n_points, :]
        # Close the circle by appending the first point to the end
        circle_points = np.vstack([circle_points, circle_points[0]])
        plt.plot(circle_points[:, 0], circle_points[:, 1], color='cadetblue', linewidth=1.5)
    
    # Plot dashed extreme lines
    plt.axvline(x=max_x, color='darkred', linestyle='dashed')
    plt.axvline(x=min_x, color='darkred', linestyle='dashed')
    plt.axhline(y=max_y, color='darkgreen', linestyle='dashed')
    plt.axhline(y=min_y, color='darkgreen', linestyle='dashed')
    
    # Draw arrows for each extreme (all black except the B1 arrow highlighted)
    arrow_colors = ['black'] * 4
    arrow_colors[min_index] = '#8FBC8F'
    plt.arrow(0, 0, max_x, 0, head_width=0.1, length_includes_head=True, color=arrow_colors[0])
    plt.arrow(0, 0, min_x, 0, head_width=0.1, length_includes_head=True, color=arrow_colors[1])
    plt.arrow(0, 0, 0, max_y, head_width=0.1, length_includes_head=True, color=arrow_colors[2])
    plt.arrow(0, 0, 0, min_y, head_width=0.1, length_includes_head=True, color=arrow_colors[3])
    
    # Draw the B5 arrow in red
    plt.arrow(0, 0, b5_point[0], b5_point[1], head_width=0.1, length_includes_head=True, color="#CD3333")
    
    # Annotate B1 and B5 values
    plt.text(b1_coords[0] * 0.5, b1_coords[1] * 0.5, f"B1\n{min_val:.2f}", 
             fontsize=12, ha='center', va='bottom', fontweight='bold')
    plt.text(b5_point[0] * 0.66, b5_point[1] * 0.66, f"B5\n{b5_value:.2f}", 
             fontsize=12, ha='center', va='bottom', fontweight='bold')
    
    # Draw an arc between the B1 and B5 arrows to represent the angle difference.
    arc_theta = np.linspace(min(angle_b1, angle_b5), max(angle_b1, angle_b5), 100)
    arc_x = 0.5 * np.cos(arc_theta)
    arc_y = 0.5 * np.sin(arc_theta)
    plt.plot(arc_x, arc_y, color='gray', linewidth=1.5)
    
    # Annotate the angle in degrees at the midpoint of the arc.
    mid_angle = (min(angle_b1, angle_b5) + max(angle_b1, angle_b5)) / 2
    plt.text(0.8 * np.cos(mid_angle), 0.8 * np.sin(mid_angle), f"{angle_diff_deg:.1f}°",
             fontsize=12, ha='center', va='center', fontweight='bold')
    
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.show()



# def get_b1s_list(extended_df, scans=90//5):
    
#     b1s,b1s_loc,b1_planes=[],[],[]
#     scans=scans
#     degree_list=list(range(18,108,scans))
#     plane=np.array(extended_df[['x','z']].astype(float))
#     b1s_for_loop_function(extended_df, b1s, b1s_loc, degree_list, plane,b1_planes)

#     if b1s:
#         try:
#             back_ang=degree_list[np.where(b1s==min(b1s))[0][0]]-scans   
#             front_ang=degree_list[np.where(b1s==min(b1s))[0][0]]+scans
#             degree_list=range(back_ang,front_ang+1)
#         except:
            
#             back_ang=degree_list[np.where(np.isclose(b1s, min(b1s), atol=1e-8))[0][0]]-scans
#             front_ang=degree_list[np.where(np.isclose(b1s, min(b1s), atol=1e-8))[0][0]]+scans
#             degree_list=range(back_ang,front_ang+1)
#     else:
     
#         return [np.array(b1s),np.array(b1s_loc)]
    
#     b1s,b1s_loc,b1_planes=[],[] ,[]
#     b1s_for_loop_function(extended_df, b1s, b1s_loc, degree_list, plane,b1_planes)
   
#     return [np.array(b1s),np.array(b1s_loc),b1_planes]

def calc_sterimol(bonded_atoms_df,extended_df,visualize=False):
    edited_coordinates_df=filter_atoms_for_sterimol(bonded_atoms_df,extended_df)
  
    b1s,b1_b5_angle,plane=get_b1s_list(edited_coordinates_df)
   
    valid_indices = np.where(b1s >= 0)[0]
    best_idx = valid_indices[np.argmin(b1s[valid_indices])]
    best_b1_plane = plane[best_idx]

    B1= min(b1s[b1s>=0])
    b1_index=np.where(b1s==B1)[0][0]
    angle=b1_b5_angle[b1_index]
   
    # loc_B1=max(b1s_loc[np.where(b1s[b1s>=0]==min(b1s[b1s>=0]))])
    B5=max(edited_coordinates_df['B5'].values)

    b5_idx = np.where(edited_coordinates_df['B5'].values == B5)[0][0]
    b5_vector = edited_coordinates_df.loc[b5_idx, ['x', 'y', 'z']].to_numpy()
    L=max(edited_coordinates_df['L'].values)
    loc_B5 = min(edited_coordinates_df['y'].iloc[np.where(edited_coordinates_df['B5'].values == B5)[0]])
    
    sterimol_df = pd.DataFrame([B1, B5, L ,loc_B5,angle], index=help_functions.XYZConstants.STERIMOL_INDEX.value)
    if visualize:
        plot_b1_visualization(best_b1_plane, edited_coordinates_df)

    return sterimol_df.T ,b5_vector


def get_sterimol_df(coordinates_df, bonds_df, base_atoms,connected_from_direction, radii='bondi', sub_structure=True, drop_atoms=None,visualize=False):

    if drop_atoms is not None:
        drop_atoms=adjust_indices(drop_atoms)
        for atom in drop_atoms:
            bonds_df = bonds_df[~((bonds_df[0] == atom) | (bonds_df[1] == atom))]
            ## drop the rows from coordinates_df
            coordinates_df = coordinates_df.drop(atom)

    bonds_direction = direction_atoms_for_sterimol(bonds_df, base_atoms)
    
    new_coordinates_df = preform_coordination_transformation(coordinates_df, bonds_direction)


    if sub_structure:
        if connected_from_direction is None:
            connected_from_direction = get_molecule_connections(bonds_df, base_atoms[0], base_atoms[1])
        else:
            connected_from_direction = connected_from_direction
    else:
        connected_from_direction = None
    

    bonded_atoms_df = get_specific_bonded_atoms_df(bonds_df, connected_from_direction, new_coordinates_df)
    
    extended_df = get_extended_df_for_sterimol(new_coordinates_df, bonds_df, radii)
    
    ###calculations
    
    sterimol_df,b5_vector = calc_sterimol(bonded_atoms_df, extended_df,visualize)
    sterimol_df= sterimol_df.rename(index={0: str(base_atoms[0]) + '-' + str(base_atoms[1])})
   
    sterimol_df = sterimol_df.round(4)
    
    return sterimol_df

  


def calc_magnitude_from_coordinates_array(coordinates_array: npt.ArrayLike) -> List[float]:
    """
    Calculates the magnitudes of each row in the given coordinates array.

    Parameters
    ----------
    coordinates_array: np.ndarray
        A nx3 array representing the x, y, z coordinates of n atoms.

    Returns
    -------
    magnitudes: List[float]
        A list of magnitudes corresponding to the rows of the input array.

    """
    magnitude = np.linalg.norm(coordinates_array, axis=1)
    return magnitude


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

    magnitude = calc_magnitude_from_coordinates_array(vibration_array)
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
    #                         index=help_functions.XYZConstants.VIBRATION_INDEX.value).T
    
    index_max=extended_df['Amplitude'].idxmax()

    max_frequency_vibration=(pd.DataFrame(extended_df.iloc[index_max]).T)[help_functions.XYZConstants.VIBRATION_INDEX.value]

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


# def get_data_for_ring_vibration(info_df: pd.DataFrame, vibration_array_list: List[np.ndarray],
#                                 coordinates_vector: np.ndarray) -> pd.DataFrame:
#     """
#     Get the data needed to analyze ring vibrations.

#     Parameters
#     ----------
#     info_df: pd.DataFrame
#         a dataframe with the frequency values of the vibrations

#     vibration_array_list: list of np.ndarray
#         a list containing arrays of the vibration vectors for each atom in the ring

#     coordinates_vector: np.ndarray
#         a vector representing the ring in three dimensions

#     Returns
#     -------
#     data_df: pd.DataFrame
#         a dataframe with the product, frequency, and sin(angle) values for each vibration in the ring

#         data_df.T:
#           product  frequency  sin_angle
#         0    0.0000    20.3253        NaN
#         1    0.0000    25.3713   0.811000
#         2    0.0062    29.0304   0.969047
#     """
    
#     product = [np.dot(row_1, row_2) for row_1, row_2 in zip(*vibration_ring_array_list_to_vector(vibration_array_list))]
#     _, vibration_array_list = vibration_ring_array_list_to_vector(vibration_array_list)
#     sin_angle = [abs(math.sin(calc_angle(row, coordinates_vector))) for row in vibration_array_list]
#     data_df = pd.DataFrame(np.vstack([product, (info_df)['Frequency'], sin_angle]),index=['Product','Frequency','Sin_angle'] )
#     return data_df.T

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
    filter_prods = (abs(data_df[help_functions.XYZConstants.RING_VIBRATION_INDEX.value[0]]) > prods_threshhold) & \
                   (data_df[help_functions.XYZConstants.RING_VIBRATION_INDEX.value[0]] != 0)
    filter_frequency = (data_df[help_functions.XYZConstants.RING_VIBRATION_INDEX.value[1]] > frequency_min_threshhold) & \
                       (data_df[help_functions.XYZConstants.RING_VIBRATION_INDEX.value[1]] < frequency_max_threshhold)
    # Apply combined filter
    filtered_df = data_df[filter_prods & filter_frequency].reset_index()
    if filtered_df.empty:
        print('No data within the specified thresholds. Adjust your thresholds.')
    
    return filtered_df



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
    vibration_atom_nums = help_functions.flatten_list(ring_atom_indices)
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
        help_functions.XYZConstants.RING_VIBRATION_INDEX.value[2]].idxmin()][2]  
    asin_max = math.asin(filtered_df[help_functions.XYZConstants.RING_VIBRATION_INDEX.value[2]].min()) * (
                180 / np.pi)
    min_vibration_frequency = filtered_df.iloc[filtered_df[
        help_functions.XYZConstants.RING_VIBRATION_INDEX.value[2]].idxmax()][2]
    asin_min = math.asin(filtered_df[help_functions.XYZConstants.RING_VIBRATION_INDEX.value[2]].max()) * (
                180 / np.pi)
    df = pd.DataFrame((max_vibration_frequency, asin_max, min_vibration_frequency, asin_min),
                      index=help_functions.XYZConstants.RING_VIBRATION_COLUMNS.value)
    
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
    """

    # Create a graph from the bonds dataframe
    atom1_idx = ring_atoms[0]
    try:
        atom2_idx = ring_atoms[1]
    except IndexError:
        atom2_idx = None

    graph = {}
    for _, row in bonds_df.iterrows():
        atom1, atom2 = int(row[0]), int(row[1])
        if atom1 not in graph:
            graph[atom1] = []
        if atom2 not in graph:
            graph[atom2] = []
        graph[atom1].append(atom2)
        graph[atom2].append(atom1)

    visited = set()
    ring_indices = []

    def dfs(atom_idx, prev_idx, depth):
        """
        Depth-first search to find a benzene ring.
        """
        visited.add(atom_idx)
        ring_indices.append(atom_idx)

        # Check if we completed a cycle of 6 atoms
        if depth == 5 and atom1_idx in graph[atom_idx]:
            return True

        for neighbor_idx in graph[atom_idx]:
            if neighbor_idx != prev_idx and neighbor_idx not in visited:
                if dfs(neighbor_idx, atom_idx, depth + 1):
                    return True

        ring_indices.pop()
        return False

    # Call DFS to find a benzene ring
    if dfs(atom1_idx, None, 0):
        print("Benzene ring found:", ring_indices)
    else:
        print("No benzene ring found.")

    if len(ring_indices) == 6:
        if atom2_idx in ring_indices:
            print("Second atom is in the benzene ring.")
            return ring_indices[3], ring_indices[0], ring_indices[1], ring_indices[-1], ring_indices[2], ring_indices[4]
        else:
            print("Second atom is NOT in the benzene ring.")
            return ring_indices[3], ring_indices[0], ring_indices[1], ring_indices[-1], ring_indices[2], ring_indices[4]
    else:
        print("No benzene ring found.")
        return None


def calculate_bond_lengths_matrix(coords, connections_df):
    num_atoms = coords.shape[0]
    bond_lengths = np.zeros((num_atoms, num_atoms))

    for _, row in connections_df.iterrows():
        atom1 = int(row[0])
        atom2 = int(row[1])
        length = np.linalg.norm(coords[atom1] - coords[atom2])
        bond_lengths[atom1, atom2] = length
        bond_lengths[atom2, atom1] = length  # Symmetric matrix

    return bond_lengths

def calculate_angles_matrix(coords, connections_df):
    num_atoms = coords.shape[0]
    angles = np.zeros((num_atoms, num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(num_atoms):
            for k in range(num_atoms):
                if i != j and j != k and i != k:
                    if (connections_df[(connections_df[0] == i) & (connections_df[1] == j)].empty == False or 
                        connections_df[(connections_df[0] == j) & (connections_df[1] == i)].empty == False) and (
                        connections_df[(connections_df[0] == i) & (connections_df[1] == k)].empty == False or 
                        connections_df[(connections_df[0] == k) & (connections_df[1] == i)].empty == False):
                        
                        v1 = coords[j] - coords[i]
                        v2 = coords[k] - coords[i]
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        angles[i, j, k] = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    return angles

# Example usage:
# coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
# connections_df = pd.DataFrame({'atom_1': [0, 0, 1], 1: [1, 2, 3]})
# angles_matrix = calculate_angles_matrix(coords, connections_df)


class Molecule():

    def __init__(self, molecule_feather_filename, parameter_list=None , new_xyz_df=None):

        self.new_index_order=new_xyz_df.index.tolist() if new_xyz_df is not None else None
        self.molecule_name = molecule_feather_filename.split('.')[0]
        self.molecule_path = os.path.dirname(os.path.abspath(molecule_feather_filename))
        os.chdir(self.molecule_path)
        
        if parameter_list == None:

            self.parameter_list = feather_file_handler(molecule_feather_filename)
        else: 
            self.parameter_list = parameter_list
        self.xyz_df = self.parameter_list[0]['standard_orientation_df']
        
        self.coordinates_array = np.array(self.xyz_df[['x', 'y', 'z']].astype(float))
        self.gauss_dipole_df = self.parameter_list[0]['dipole_df']
        self.polarizability_df = self.parameter_list[0]['pol_df']
        self.bonds_df = extract_connectivity(self.xyz_df)
        self.atype_list = nob_atype(self.xyz_df, self.bonds_df)
        self.info_df = self.parameter_list[0]['info_df']
        self.charge_dict = self.parameter_list[2]
 
        self.vibration_dict = self.parameter_list[1]
      


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
        help_functions.data_to_xyz(self.xyz_df, self.molecule_name+'.xyz')
    
    def write_csv_files(self) -> None:
        """
        Writes all class variables to CSV files with the variable name as the file 
        """
        for var_name, var_value in vars(self).items():
            if isinstance(var_value, pd.DataFrame):
                var_value.to_csv(f"{var_name}.csv", index=False)


    def visualize_molecule(self) -> None:
        """
        Visualizes the molecule using the `visualize` module.
        """
        
        visualize.show_single_molecule(molecule_name=self.molecule_name, xyz_df=self.xyz_df)


    def process_sterimol_atom_group(self, atoms, radii, sub_structure=True, drop_atoms=None,visualize=False) -> pd.DataFrame:

        connected = get_molecule_connections(self.bonds_df, atoms[0], atoms[1])
        print(connected)
        return get_sterimol_df(self.xyz_df, self.bonds_df, atoms, connected, radii, sub_structure=sub_structure, drop_atoms=drop_atoms, visualize=visualize)

    def get_sterimol(self, base_atoms: Union[None, Tuple[int, int]] = None, radii: str = 'bondi',sub_structure=True, drop_atoms=None,visualize=False) -> pd.DataFrame:
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
            sterimol_list = [self.process_sterimol_atom_group(atoms, radii, sub_structure=sub_structure, drop_atoms=drop_atoms,visualize=visualize) for atoms in base_atoms]
            sterimol_df = pd.concat(sterimol_list, axis=0)

        else:
            # If base_atoms is a single group, just process that group
            sterimol_df = self.process_sterimol_atom_group(base_atoms, radii,sub_structure=sub_structure, drop_atoms=drop_atoms,visualize=visualize)
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
            if isinstance(diff_indices[0], list):
                diff_list = []
                for atoms in diff_indices:
                    atoms = adjust_indices(atoms)
                    # Calculate difference between the two specified atoms.
                    diff = pd.DataFrame(
                        df.iloc[atoms[0]] - df.iloc[atoms[1]],
                        columns=[f'diff_{atoms[0]+1}-{atoms[1]+1}']
                    )
                    diff_list.append(diff)
                return pd.concat(diff_list, axis=1)
            else:
                diff_indices_adj = adjust_indices(diff_indices)
                return pd.DataFrame(
                    df.iloc[diff_indices_adj[0]] - df.iloc[diff_indices_adj[1]],
                    columns=[f'diff_{diff_indices_adj[0]+1}-{diff_indices_adj[1]+1}']
                )

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
            if sub_atoms is not None and isinstance(sub_atoms[0], list):
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
    
    def get_dipole_gaussian_df_single(self, atoms,origin):
        
        dipole_df = calc_dipole_gaussian(self.coordinates_array, np.array(self.gauss_dipole_df), atoms,origin=origin)
        dipole_df = dipole_df.rename(index={0: f'dipole_{atoms[0]}-{atoms[1]}-{atoms[2]}'})
        return dipole_df

    def get_dipole_gaussian_df(self, base_atoms_indices: List[int], origin=None) -> pd.DataFrame:
        """
        Returns a DataFrame with the dipole moments calculated based on the specified base atoms.

        Args:
            base_atoms_indices (List[int]): The indices of the base atoms to use for the dipole moment calculation.

        Returns:
            pd.DataFrame: A DataFrame with the dipole moments.
        """
     
        if isinstance(base_atoms_indices[1], list):
           
            # If base_atoms_indices is a list of lists, process each group individually and concatenate the results
            dipole_list = [self.get_dipole_gaussian_df_single(atoms,origin=origin) for atoms in base_atoms_indices]
            dipole_df = pd.concat(dipole_list, axis=0)
        else:
            # If base_atoms_indices is a single group, just process that group
            dipole_df = self.get_dipole_gaussian_df_single(base_atoms_indices,origin=origin)
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
                
                z,x,c,v,b,n=get_benzene_ring_indices(self.bonds_df, ring_atom_indices)
                ring_atom_indices=[[z,x],[c,v],[b,n]]
            
                try:
                    
                    filtered_df=get_filtered_ring_df(self.info_df,self.coordinates_array,self.vibration_dict,ring_atom_indices)
                    

                except FileNotFoundError:
                    return print(f'No vibration - Check atom numbering in molecule {self.molecule_name}')
            
                return calc_min_max_ring_vibration(filtered_df)
        except:
            print("this molecule is ",self.molecule_name)
    
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

    

    def get_molecule_comp_set_hetro(self,dipole_mode = 'gaussian', radii = 'bondi'):
        """
      Ring atoms - by order -> primary axis (para first), ortho atoms and meta atoms: 19 20 19 21 20 22
      your atom pairs: 19 20 19 21
      Enter atoms - origin atom, y axis atom and xy plane atom: 1 4 5
      Primary axis along: 2 3
      Distances - Atom pairs: 2 3 2 5
      __main__:192: RuntimeWarning: invalid value encountered in double_scalars
      __main__:483: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      __main__:485: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      Dot products are lower than 0.1 - returning the default 500 - 700 1/cm
      Do you want to compute any angles/dihedrals? y/n: n
        -------
        res : dataframe
        	molecule1
        cross	657.3882
        cross_angle	81.17206344670865
        para	834.4249
        para_angle	40.67483293898943
        dipole_x	0.058148468892639825
        dipole_y	1.210906929546481
        dipole_z	0.9491739479893856
        total	1.5397
        nbo charge[0 1]	0.2706980000000003
        nbo charge[2 3]	0.013509999999999689
        B1	2.556
        B5	4.131529971026473
        L	5.8357
        loc_B5	1.4121
        loc_B1	3.6855
        bond length[0 1]	1.4566677726921815
        bond length[2 3]	1.2277597484850202
        angle_[1, 2, 3]	69.7212447900024
        dihedral_[1, 2, 3, 4]	104.74132137986373


        """
        res=[]
        ring_atom_indices= help_functions.split_to_pairs(input("Ring atoms - by order -> primary axis (para first), ortho atoms and meta atoms: "))
        gen_vibration_indices= help_functions.split_to_pairs(input("your atom pairs: "))
        dipole_indices= help_functions.string_to_int_list(input('Enter atoms - origin atom, y axis atom and xy plane atom: '))
        # nbo_indices=help_functions.split_to_pairs(input('Insert atom pairs for which you wish calculate differences: '))
        sterimol_indices= help_functions.string_to_int_list(input('Primary axis along: '))
        indices_for_dist= help_functions.split_to_pairs(input('Distances - Atom pairs: '))
        res.append(self.get_ring_vibrations(ring_atom_indices))
        res.append(self.get_stretch_vibration(gen_vibration_indices))
        if dipole_mode=='compute':
            res.append(self.get_npa_df(dipole_indices))
        elif dipole_mode=='gaussian':
            res.append(self.get_dipole_gaussian_df(dipole_indices))
        # res.append(self.get_nbo_info(nbo_indices))
        if radii=='CPK':
            res.append(self.get_sterimol(sterimol_indices,radii))
        else:
           res.append(self.get_sterimol(sterimol_indices) )
        res.append(self.get_bond_length(indices_for_dist))
        
        angle_answer=input('Do you want to compute any angles/dihedrals? y/n: ')
        if angle_answer=='y':
            print('Insert a list of atom triads/quartets for which you wish to have angles/dihedrals.\n'
        'For several angles/dihedrals, insert triads/quartets with a double space between them.\n'
        'Make sure there are spaces between atoms as well.\n'
        'For example - 1 2 3  1 2 3 4 will give angle(1, 2, 3) and dihedral(1, 2, 3, 4)')
            angle_string= help_functions.split_for_angles(input('Insert a list of atom triads/quartets for which you wish to have angles/dihedrals: '))
            res.append(self.get_bond_angle(angle_string))
        res.append(self.polarizability_df)
        
        df=pd.concat(res).rename(columns={0:self.molecule_name}).T
        df.to_csv('comp_set.csv',mode='x')
        return df

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

    #     if renumber:
    #         self.renumber_molecules()

    

    # def renumber_molecules(self):
        
    #     os.chdir(self.molecules_path)
    #     try:
    #         os.mkdir('xyz_files')
    #     except FileExistsError:
    #         pass
    #     directories = [d for d in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), d)) and d[-1].isdigit()]
    #     if directories:
    #         os.chdir(directories[0])
    #         print(f'renumbering files exist {os.getcwd(),os.listdir()}')
    #     else:
    #         os.chdir('xyz_files')
    #         [mol.write_xyz_file() for mol in self.molecules]
  
    #         renumbering.batch_renumbering(os.getcwd())
    #         os.chdir('../')
    #         directories = [d for d in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), d)) and d[-1].isdigit()]
    #         os.chdir(directories[0])
   
    #     xyz_files=[file for file in os.listdir() if file.endswith('.xyz')]
    #     xyz_dfs=[help_functions.get_df_from_file(file) for file in xyz_files] 

    #     self.molecules=[]
    #     self.failed_molecules=[]
    #     os.chdir(self.molecules_path)
    #     for feather_file, df in zip(os.listdir(), xyz_dfs): 
    #         if feather_file.endswith('.feather'):
    #             try:
    #                 self.molecules.append(Molecule(feather_file,parameter_list=None, new_xyz_df=df))
    #             except:
    #                 self.failed_molecules.append(feather_file)
    #                 print(f'Error: {feather_file} could not be processed')
    #             # self.molecules.append(Molecule(log_file))
         
    #     self.molecules_names=[molecule.molecule_name for molecule in self.molecules]
    #     self.old_molecules=self.molecules
    #     self.old_molecules_names=self.molecules_names
    #     os.chdir('../')
    
    def export_all_xyz(self):
        os.makedirs('xyz_files', exist_ok=True)
        os.chdir('xyz_files')
        for mol in self.molecules:
            xyz_df=mol.xyz_df
            mol_name=mol.molecule_name
            help_functions.data_to_xyz(xyz_df, f'{mol_name}.xyz')
        os.chdir('../')

    def filter_molecules(self, indices):
        self.molecules = [self.molecules[i] for i in indices]
        self.molecules_names = [self.molecules_names[i] for i in indices]

    def get_sterimol_dict(self,atom_indices, radii='CPK',sub_structure=True, drop_atoms=None):
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
            sterimol_dict[molecule.molecule_name]=molecule.get_sterimol(atom_indices, radii, sub_structure=sub_structure, drop_atoms=drop_atoms)
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
      
            ring_dict[molecule.molecule_name]=molecule.get_ring_vibrations(ring_atom_indices)

        return ring_dict
    
    def get_dipole_dict(self,atom_indices,origin):
        dipole_dict={}
        for molecule in self.molecules:
            try:
                dipole_dict[molecule.molecule_name]=molecule.get_dipole_gaussian_df(atom_indices,origin=origin)
            except:
                print(f'Error: {molecule.molecule_name} Dipole could not be processed')
                pass
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
                pass
        return nbo_dict
    
    def get_charge_diff_df_dict(self,atom_indices,type='nbo'):
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
        
    
    def get_molecules_comp_set_app(self,answers_dict: dict,
                                    radii = 'bondi', export_csv=False, answers_list_load=None, iso=False):
        """
        molecules.get_molecules_comp_set()
        Ring atoms - by order -> primary axis (para first), ortho atoms and meta atoms: 1,3 1,6 3,4
        your atom pairs: 1,6 3,4
        Enter atoms - origin atom, y axis atom and xy plane atom: 1,2,3
        Insert atom pairs for which you wish calculate differences: 1,2 3,4
        Primary axis along: 1,2
        Distances - Atom pairs: 1,2 3,4
        Bending - Atom pairs: 1,2 3,4
        Do you want to compute any angles/dihedrals? y/n: y
        Insert a list of atom triads/quartets for which you wish to have angles/dihedrals.
        For several angles/dihedrals, insert triads/quartets with a double space between them.
        Make sure there are spaces between atoms as well.
        For example - 1,2,3  1,2,3,4 will give angle(1, 2, 3) and dihedral(1, 2, 3, 4)
        Insert a list of atom triads/quartets for which you wish to have angles/dihedrals: 1,2,3  1,2,3,4
        -------
        df : 
            
        """
        answers_list=[]
        print(answers_dict)
        if answers_dict is not None:
            for question, answer in answers_dict.items():
                if answer is not None:
                    if answer==[]:
                        answers_list.append([])
                    elif question.startswith('Dipole atoms'):
                        
                        answers_dict[question]=help_functions.convert_to_list_or_nested_list(answer)
                        answers_list.append(answers_dict[question])
                    
                    else:
                        answers_dict[question]=help_functions.convert_to_list_or_nested_list(answer)
                        answers_list.append(answers_dict[question])
        
        if answers_list_load is not None:
            answers_list=answers_list_load
        
      
        res_df=pd.DataFrame()
        
        if  answers_list[0] and answers_list[0] != []:
            try:
                res_df=help_functions.dict_to_horizontal_df(self.get_ring_vibration_dict(answers_list[0])) ### get_ring_vibration_dict change all of them later
            except Exception as e:
                print(e)
                pass
        if answers_list[2] and answers_list[2]!= []:
            try:
                res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df(self.get_stretch_vibration_dict(answers_list[2],answers_list[1][0]))],axis=1)
            except Exception as e:
                print(e)
                pass

        if answers_list[4] and answers_list[4]!= []:
            try:
                res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df(self.get_bend_vibration_dict(answers_list[4],answers_list[3][0]))],axis=1)
            except Exception as e:
                print(e)
                pass
        if answers_list[5] and answers_list[5]!= []:
            try:
    
                res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df(self.get_npa_dict(answers_list[5],sub_atoms=answers_list[6]))],axis=1) ## add sub_atoms
            except Exception as e:
                print(e)
                pass
        if answers_list[7] and answers_list[7]!= []:
            try:
                res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df(self.get_dipole_dict(answers_list[7]))],axis=1)
            except Exception as e:
                print(e)
                pass
                
        if answers_list[8] and answers_list[8]!= []:
            try:  
                res_df=pd.concat([res_df,help_functions.charge_dict_to_horizontal_df(self.get_charge_df_dict(answers_list[8]))],axis=1)
                if res_df.empty:
                    try:
                        res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df(self.get_charge_diff_df_dict(answers_list[8]))],axis=1)
                    except Exception as e:
                        print(e)
                        pass
            except Exception as e:
                print(e)
                pass
        if answers_list[9] and answers_list[9]!= []:
            try:
                res_df=pd.concat([res_df,help_functions.charge_dict_to_horizontal_df(self.get_charge_diff_df_dict(answers_list[9]))],axis=1)
                if res_df.empty:
                    try:
                        res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df(self.get_charge_diff_df_dict(answers_list[9]))],axis=1)
                    except Exception as e:
                        print(e)
                        pass
            except Exception as e:
                print(e)
                pass

        if answers_list[10] and answers_list[10]!= []:
            try:
                
                res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df(self.get_sterimol_dict(answers_list[10],radii=radii))],axis=1) ## add cpk and bondi
                
            except Exception as e:
                print(e)
                pass

        if answers_list[11] and answers_list[11]!= []:
            try:
                res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df(self.get_bond_length_dict(answers_list[11]))],axis=1)
            except Exception as e:
                print(e)
                pass
        if answers_list[12] and answers_list[12]!= []:
            try:
                res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df(self.get_bond_angle_dict(answers_list[12]))],axis=1)
            except Exception as e:
                print(e)
                pass
        if iso:
            self.polarizability_df_concat=pd.DataFrame()
            for molecule in self.molecules:
                info=molecule.polarizability_df
                info=info.rename(index={0:molecule.molecule_name})
                self.polarizability_df_concat=pd.concat([self.polarizability_df_concat,info],axis=0)
            res_df=pd.concat([res_df,self.polarizability_df_concat],axis=1)

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
            help_functions.data_to_xyz(molecule.xyz_df,(molecule.molecule_name+'.xyz'))
        os.chdir('../')

if __name__=='__main__':
    pass