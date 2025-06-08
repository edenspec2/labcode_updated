from typing import List, Optional
import numpy as np
import numpy.typing as npt
import pandas as pd


try:
    from ...utils.help_functions import *
    from ...utils.visualize import *
except ImportError:
    from utils.help_functions import * 
    from utils.visualize import *



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