import pandas as pd
import numpy as np
import os
import sys
import math
from enum import Enum
import igraph as ig
from .gaussian_handler import feather_file_handler
from typing import *
from ..utils import visualize
import warnings
from scipy.spatial.distance import pdist, squareform
from ..utils import help_functions
from ..Mol_align import renumbering


warnings.filterwarnings("ignore", category=RuntimeWarning)

class GeneralConstants(Enum):
    """
    Holds constants for calculations and conversions
    1. covalent radii from Alvarez (2008) DOI: 10.1039/b801115j
    2. atomic numbers
    2. atomic weights
    """
    COVALENT_RADII= {
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
    
    CPK_RADII={
        'C':1.50,   'H':1.00,   'S.O':1.70,  'Si':2.10,
        'C2':1.60,  'N':1.50,   'S1':1.00,   'Co':2.00,
        'C3':1.60,  'C66':1.70, 'F':1.35,    'Ni':2.00,
        'C4':1.50,  'N4':1.45,  'Cl':1.75,
        'C5/N5':1.70, 'O':1.35, 'S4':1.40,
        'C6/N6':1.70, 'O2':1.35, 'Br':1.95,
        'C7':1.70,    'P':1.40,  'I':2.15,
        'C8':1.50,    'S':1.70,  'B':1.92,
    
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
        'O.3':'O', 'N.1':'N', 'S.O2':'S1',
        'O.co2':'O', 'N.3':'C6/N6','P.3':'P',
        'C.1':'C3', 'N.ar':'C6/N6',
        'C.2':'C2', 'N.am':'C6/N6',
        "C.cat":'C3', 'N.pl3':'C6/N6',
        'C.3':'C', 'N.4':'N',
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


def adjust_indices(indices: npt.ArrayLike, adjustment_num: int=1) -> npt.ArrayLike:
    """
    adjust indices by adjustment_num
    """
    return np.array(indices)-adjustment_num

def adjust_indices_xyz(indices: npt.ArrayLike) -> npt.ArrayLike:
    """
    adjust indices by adjustment_num
    """
    return adjust_indices(indices, adjustment_num=1)

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
    optional: if the atom_indices is 4, the origin will be the middle of the first two atoms.
    """
    new_origin=coordinates_array[atom_indices[0]]
    if (len(atom_indices)==4):
        new_origin=(new_origin+coordinates_array[atom_indices[1]])/2
    new_y=(coordinates_array[atom_indices[-2]]-new_origin)/np.linalg.norm((coordinates_array[atom_indices[-2]]-new_origin))
    coplane=((coordinates_array[atom_indices[-1]]-new_origin)/np.linalg.norm((coordinates_array[atom_indices[-1]]-new_origin)+0.00000001))
    return (new_origin,new_y,coplane)

def np_cross_and_vstack(plane_1, plane_2):
    cross_plane=np.cross(plane_1, plane_2)
    united_results=np.vstack([plane_1, plane_2, cross_plane])
    return united_results

def calc_basis_vector(origin, y: npt.ArrayLike, coplane: npt.ArrayLike):#help function for calc_coordinates_transformation
    """
    origin: origin of the new basis
    y: y direction of the new basis
    coplane: a vector that is coplanar with the new y direction
    """
    # cross_y_plane=np.cross(coplane,y)
    # coef_mat=np.vstack([y, coplane, cross_y_plane])
    coef_mat=np_cross_and_vstack(coplane, y)
    angle_new_y_coplane=calc_angle(coplane,y)
    cop_ang_x=angle_new_y_coplane-(np.pi/2)
    # result_vector=[0,np.cos(cop_ang_x),0]
    result_vector=[np.cos(cop_ang_x), 0, 0]
    new_x,_,_,_=np.linalg.lstsq(coef_mat,result_vector,rcond=None)
    new_basis=np_cross_and_vstack(new_x, y)
    # new_z=np.cross(new_x,y)
    # new_basis=np.vstack([new_x, y, new_z])
    return new_basis

def transform_row(row_array, new_basis, new_origin, round_digits):
    translocated_row = row_array - new_origin
    return np.dot(new_basis, translocated_row).round(round_digits)



def calc_coordinates_transformation(coordinates_array: npt.ArrayLike, base_atoms_indices: npt.ArrayLike, round_digits:int=4 ,origin:npt.ArrayLike=None) -> npt.ArrayLike:#origin_atom, y_direction_atom, xy_plane_atom
    """
    a function that recives coordinates_array and new base_atoms_indices to transform the coordinates by
    and returns a dataframe with the shifted coordinates
    parameters:
    ----------
    coordinates_array: np.array
        xyz molecule array
    base_atoms_indices: list of nums
        indices of new atoms to shift coordinates by.
    origin: in case you want to change the origin of the new basis, middle of the ring for example. used in npa_df
    returns:
        transformed xyz molecule dataframe
    -------
        
    example:
    -------
    calc_coordinates_transformation(coordinates_array,[2,3,4])
    
    Output:
        atom       x       y       z
      0    H  0.3477 -0.5049 -1.3214
      1    B     0.0     0.0     0.0
      2    B    -0.0  1.5257     0.0
    """
    indices=adjust_indices_xyz(base_atoms_indices)
    new_basis=calc_basis_vector(*calc_new_base_atoms(coordinates_array,indices))    
    if origin is None:
        new_origin=coordinates_array[indices[0]]
    else:
        new_origin=origin

    transformed_coordinates = np.apply_along_axis(lambda x: transform_row(x, new_basis, new_origin, round_digits), 1,
                                                  coordinates_array)
    # transformed_coordinates=np.array([np.dot(new_basis,(row-new_origin)) for row in coordinates_array]).round(round_digits)
    return transformed_coordinates

def preform_coordination_transformation(xyz_df, indices=None):
    xyz_copy=xyz_df.copy()
    coordinates=np.array(xyz_copy[['x','y','z']].values)
    if indices is None:
        xyz_copy[['x','y','z']]=calc_coordinates_transformation(coordinates, [1,2,3])
    else:
        # print('indices', indices)
        xyz_copy[['x','y','z']]=calc_coordinates_transformation(coordinates, indices)
    # xyz_copy[['x','y','z']]=calc_coordinates_transformation(coordinates, get_indices([1,2,3])
    return xyz_copy

def calc_npa_charges(coordinates_array: npt.ArrayLike,charge_array: npt.ArrayLike,  geom_transform_indices=None):##added option for subunits
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
    # what is NPA here??
    # indices=adjust_indices(base_atoms_indices)
    # transformed_coordinates=calc_coordinates_transformation(coordinates_array, indices)
    # if sub_atoms:
    #     atom_mask=sub_atoms
    # else:
    #     atom_mask=range(len(charge_array))
    # atom_mask=range(charge_array) if sub_atoms==None else sub_atoms
#TODO: Add option for sub_atoms!
    # Apply geometric transformation if specified
    if geom_transform_indices is not None:
        geometric_center = np.mean(coordinates_array[geom_transform_indices], axis=0)
        coordinates_array -= geometric_center

    dipole_xyz = np.vstack([(row[0] * row[1])for row in
                            list(zip(coordinates_array, charge_array))])
    dipole_vector=np.sum(dipole_xyz,axis=0)
    array_dipole=np.hstack([dipole_vector,np.linalg.norm(dipole_vector)])
    dipole_df=pd.DataFrame(array_dipole,index=help_functions.XYZConstants.DIPOLE_COLUMNS.value).T
    
    return dipole_df

def calc_dipole_gaussian(coordinates_array, gauss_dipole_array, base_atoms_indices ,geometric_transformation_indices=None):
    """
    a function that recives coordinates and gaussian dipole, transform the coordinates
    by the new base atoms and calculates the dipole in each axis
    """
    if geometric_transformation_indices:
        # Calculate the geometric center of specified indices
        geometric_center = np.mean(coordinates_array[geometric_transformation_indices], axis=0)
        # Translate all coordinates
        coordinates_array -= geometric_center

    indices=adjust_indices(base_atoms_indices)
    basis_vector=calc_basis_vector(*calc_new_base_atoms(coordinates_array, indices))
    gauss_dipole_array[0,0:3]=np.matmul(basis_vector,gauss_dipole_array[0,0:3])
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

    if  isinstance(indices[0], tuple):
        bond_vector=[(coordinates_array[index[0]]-coordinates_array[index[1]]) for index in indices]
    else:
        bond_vector= coordinates_array[indices[0]]-coordinates_array[indices[1]]

    return bond_vector

def get_bonds_vector_for_calc_angle(coordinates_array,atoms_indices): ##for calc_angle_between_atoms

    indices=adjust_indices(atoms_indices)#three atoms-angle four atoms-dihedral
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
    except: 
        # if (any(bonds_df[0]==direction)):
        #     print(bonds_df[(bonds_df[0]==direction)][1].iloc[0],'h')
        #     base_atoms_copy.append(int(bonds_df[(bonds_df[0]==direction)][1].iloc[0])) if int(bonds_df[(bonds_df[0]==direction)][1].iloc[0])!=origin else base_atoms_copy.append(int(bonds_df[(bonds_df[0]==direction)][0].iloc[0]))
        # else:
        #     print(bonds_df[(bonds_df[1]==direction)][0].iloc[0],'w')
        #     base_atoms_copy.append(int(bonds_df[(bonds_df[1]==direction)][0].iloc[0])) if int(bonds_df[(bonds_df[1]==direction)][0].iloc[0])!=origin else base_atoms_copy.append(int(bonds_df[(bonds_df[1]==direction)][1].iloc[0]))
        for _, row in bonds_df.iterrows():
            if row[0] == direction:
                base_atoms_copy.append(row[1])
                break
            elif row[1] == direction:
                base_atoms_copy.append(row[0])
                break
    return base_atoms_copy

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



def extract_connectivity(xyz_df, threshhold_distance=2.02):
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
    for idx,row in enumerate(dist_array):
        if ((row[3]=='H') & (row[4] not in help_functions.XYZConstants.NOF_ATOMS.value)):
            remove_list.append(idx)
        if ((row[3] == 'H') & (row[4] == 'H')):
            remove_list.append(idx)
        if (((row[3] == 'H') | (row[4] == 'H')) & (row[2]>=1.5) ):
            remove_list.append(idx)
        if ((row[2]>=threshhold_distance) | (row[2]==0)):
            remove_list.append(idx)
    dist_df=dist_df.drop(remove_list)
    dist_df=dist_df.drop_duplicates(subset=['value'])
    dist_array=np.array(dist_df[['a1','a2']])+1
    return pd.DataFrame(dist_array)

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
    # print(center_atom)
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


def get_extended_df_for_sterimol(coordinates_df, bonds_df, radii='bondi'):
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
    
    bond_type_map = GeneralConstants.BOND_TYPE.value
    ## if radius is cpk mapping should be done on atype, else on atom
    radii_map = GeneralConstants.BONDI_RADII.value if radii == 'bondi' else GeneralConstants.COVALENT_RADII.value
    df = coordinates_df.copy()  # make a copy of the dataframe to avoid modifying the original
    df['atype']=nob_atype(coordinates_df, bonds_df)
    
    # df['atype'] = df['atom'].map(bond_type_map).fillna(bond_type)
    df['magnitude'] = calc_magnitude_from_coordinates_array(df[['x', 'z']].astype(float))
   
    df['radius'] = df['atom'].map(radii_map)
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
    # print(degree,plane)
    cos_deg=np.cos(degree*(np.pi/180))
    sin_deg=np.sin(degree*(np.pi/180))
    rot_matrix=np.array([[cos_deg,-1*sin_deg],[sin_deg,cos_deg]])
    transformed_plane=np.vstack([np.matmul(rot_matrix,row) for row in plane]).round(4)
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
    idx=np.where(np.isclose(np.abs(transformed_plane[:,column_index]),(avs.min()).round(4)))[0][0]
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
    for i in range(1,transformed_plane.shape[0]): 
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

def b1s_for_loop_function(extended_df, b1s, b1s_loc, degree_list, plane):
    """
    a function that gets a plane transform it and calculate the b1s for each degree.
    checks if the plane is in the x or z axis and calculates the b1s accordingly.
    Parameters:
    ----------
    extended_df : pd.DataFrame
    b1s : list
    b1s_loc : list
    degree_list : list
    plane : np.array
    """
    for degree in degree_list:
        transformed_plane=get_transfomed_plane_for_sterimol(plane, degree)
        
        avs=np.abs([max(transformed_plane[:,0]),min(transformed_plane[:,0]), 
                    max(transformed_plane[:,1]),min(transformed_plane[:,1])])
        
        if np.where(avs==avs.min())[0][0] in [0,1]:
                B1,B1_loc=calc_B1(transformed_plane,avs,extended_df,0)
  
        elif np.where(avs==avs.min())[0][0] in [2,3]:
            B1,B1_loc=calc_B1(transformed_plane,avs,extended_df,1)
            
        b1s.append(np.unique(np.vstack(B1)).max())####check
        b1s_loc.append(np.unique(np.vstack(B1_loc)).max())

def get_b1s_list(extended_df, scans=90//5):
    
    b1s,b1s_loc=[],[]
    scans=scans
    degree_list=list(range(18,108,scans))
    plane=np.array(extended_df[['x','z']].astype(float))
    b1s_for_loop_function(extended_df, b1s, b1s_loc, degree_list, plane)
    back_ang=degree_list[np.where(b1s==min(b1s))[0][0]]-scans   
    front_ang=degree_list[np.where(b1s==min(b1s))[0][0]]+scans
    degree_list=range(back_ang,front_ang+1)
    print('e')
    b1s_for_loop_function(extended_df, b1s, b1s_loc, degree_list, plane)

    return [np.array(b1s),np.array(b1s_loc)]

def calc_sterimol(bonded_atoms_df,extended_df):
    edited_coordinates_df=filter_atoms_for_sterimol(bonded_atoms_df,extended_df)
    b1s,b1s_loc=get_b1s_list(edited_coordinates_df)
    B1=min(b1s[b1s>=0])
    loc_B1=max(b1s_loc[np.where(b1s[b1s>=0]==min(b1s[b1s>=0]))])
    B5=max(edited_coordinates_df['B5'].values)
    L=max(edited_coordinates_df['L'].values)
    loc_B5 = min(edited_coordinates_df['y'].iloc[np.where(edited_coordinates_df['B5'].values == B5)[0]])
    sterimol_df = pd.DataFrame([B1, B5, L, loc_B1,loc_B5], index=help_functions.XYZConstants.STERIMOL_INDEX.value)
    return sterimol_df.T


def get_sterimol_df(coordinates_df, bonds_df, base_atoms,connected_from_direction, radii='bondi', sub_structure=True):

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
    
    sterimol_df = calc_sterimol(bonded_atoms_df, extended_df)
    sterimol_df= sterimol_df.rename(index={0: str(base_atoms[0]) + '-' + str(base_atoms[1])})

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
    bool_check = (pair in bonds_list)
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
        vibration_array_list = [vibration_dict[f'vibration_atom_{num}'] for num in vibration_atom_nums]
    except KeyError:
        return print('there is no vibration for those atoms-pick another one')
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
                                          atom_pair: list, info_df: pd.DataFrame, operation:str='dot') -> List[float]:
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
    extended_df=extended_df_for_vib(vibration_dict,info_df,atom_pair)
    coordinates_vector=coordinates_array[atoms[0]]-coordinates_array[atoms[1]]
    vibration_dot_product = calc_vibration_dot_product(extended_df, coordinates_vector)
    extended_df['Dot_product']=vibration_dot_product
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
    
    index_max=extended_df['Dot_product'].idxmax()

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
    vec_sum_1_3_5 = (vibration_array_list[0] + vibration_array_list[2] + vibration_array_list[3])
    vec_sum_2_4_6 = (vibration_array_list[4] + vibration_array_list[1] + vibration_array_list[5])
    return vec_sum_1_3_5, vec_sum_2_4_6


def get_data_for_ring_vibration(info_df: pd.DataFrame, vibration_array_list: List[np.ndarray],
                                coordinates_vector: np.ndarray) -> pd.DataFrame:
    """
    Get the data needed to analyze ring vibrations.

    Parameters
    ----------
    info_df: pd.DataFrame
        a dataframe with the frequency values of the vibrations

    vibration_array_list: list of np.ndarray
        a list containing arrays of the vibration vectors for each atom in the ring

    coordinates_vector: np.ndarray
        a vector representing the ring in three dimensions

    Returns
    -------
    data_df: pd.DataFrame
        a dataframe with the product, frequency, and sin(angle) values for each vibration in the ring

        data_df.T:
          product  frequency  sin_angle
        0    0.0000    20.3253        NaN
        1    0.0000    25.3713   0.811000
        2    0.0062    29.0304   0.969047
    """
    product = [np.dot(row_1, row_2) for row_1, row_2 in zip(*vibration_ring_array_list_to_vector(vibration_array_list))]
    _, vibration_array_list = vibration_ring_array_list_to_vector(vibration_array_list)
    sin_angle = [abs(math.sin(calc_angle(row, coordinates_vector))) for row in vibration_array_list]
    data_df = pd.DataFrame(np.vstack([product, (info_df)['Frequency'], sin_angle]),index=['Product','Frequency','Sin_angle'] )
    return data_df.T


def get_filter_ring_vibration_df(data_df: pd.DataFrame, prods_threshhold: float = 0.1,
                                 frequency_min_threshhold: float = 1500,
                                 frequency_max_threshhold: float = 1700) -> pd.DataFrame:
    """
    Returns a filtered dataframe based on thresholds for product value, minimum and maximum frequency.

    Parameters
    ----------
    data_df: pd.DataFrame
        DataFrame containing ring vibration data.

    prods_threshhold: float, optional
        The minimum threshold for the product value. Default is 0.1.

    frequency_min_threshhold: float, optional
        The minimum threshold for frequency. Default is 500.

    frequency_max_threshhold: float, optional
        The maximum threshold for frequency. Default is 700.

    Returns
    -------
    pd.DataFrameindex=help_functions.XYZConstants.RING_VIBRATION_INDEX.value
        Filtered DataFrame containing ring vibration data.
    """
    filter_prods_null = (data_df[help_functions.XYZConstants.RING_VIBRATION_INDEX.value[0]] != 0)
    filter_prods_value = (
                abs(data_df[help_functions.XYZConstants.RING_VIBRATION_INDEX.value[0]]) > prods_threshhold)
    filter_min_frequency = (data_df[
                                help_functions.XYZConstants.RING_VIBRATION_INDEX.value[
                                    1]] > frequency_min_threshhold)  # wierd part - check about dot product
    filtered_df = data_df[filter_prods_null & filter_prods_value][
        filter_min_frequency].reset_index()  ##randon threshhold to work, no >1500
    if (filtered_df.shape[0] == 0):
        filter_max_frequency = (data_df[
                                    help_functions.XYZConstants.RING_VIBRATION_INDEX.value[
                                        1]] < frequency_max_threshhold)
        filtered_df = data_df[filter_prods_null][filter_min_frequency & filter_max_frequency].reset_index()
        print('Dot products are lower than 0.1 - returning the default 1500 - 1700 1/cm')
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

    max_vibration_frequency = filtered_df.iloc[filtered_df[
        help_functions.XYZConstants.RING_VIBRATION_INDEX.value[1]].idxmax()][2]  # [1] is product
    asin_max = math.asin(filtered_df[help_functions.XYZConstants.RING_VIBRATION_INDEX.value[2]].max()) * (
                180 / np.pi)
    min_vibration_frequency = filtered_df.iloc[filtered_df[
        help_functions.XYZConstants.RING_VIBRATION_INDEX.value[1]].idxmin()][2]
    asin_min = math.asin(filtered_df[help_functions.XYZConstants.RING_VIBRATION_INDEX.value[2]].min()) * (
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
        self.charge_df = self.parameter_list[0]['charge_df']
        self.energy_value=self.parameter_list[0]['energy_value']
        self.vibration_dict = self.parameter_list[1]
        if self.new_index_order is not None:
            self.renumber_dfs(new_xyz_df)
            
    def renumber_dfs(self, new_xyz_df):
        # Reindexing DataFrames
        
        self.xyz_df = new_xyz_df
        self.coordinates_array = np.array(self.xyz_df[['x', 'y', 'z']].astype(float))
        
        self.gauss_dipole_df = self.gauss_dipole_df.reindex(self.new_index_order).reset_index(drop=True)
        # self.polarizability_df = self.polarizability_df.reindex(new_index_order)
        self.bonds_df = extract_connectivity(self.xyz_df)
        
        self.atype_list = nob_atype(self.xyz_df, self.bonds_df)
        
        self.info_df = self.info_df
        self.charge_df = self.charge_df.reindex(self.new_index_order).reset_index(drop=True)
        # For scalar values like `energy_value`, no reindexing is needed
        self.energy_value = self.parameter_list[0]['energy_value']
        self.vibration_dict = self.reindex_vibration_dict()
        
        print(f'successfully reindexed {self.molecule_name} ')


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


    def process_sterimol_atom_group(self, atoms, radii):

        connected = get_molecule_connections(self.bonds_df, atoms[0], atoms[1])
        return get_sterimol_df(self.xyz_df, self.bonds_df, atoms, connected, radii)

    def get_sterimol(self, base_atoms: Union[None, Tuple[int, int]] = None, radii: str = 'bondi') -> pd.DataFrame:
        """
        Returns a DataFrame with the Sterimol parameters calculated based on the specified base atoms and radii.

        Args:
            base_atoms (Union[None, Tuple[int, int]], optional): The indices of the base atoms to use for the Sterimol calculation. Defaults to None.
            radii (str, optional): The radii to use for the Sterimol calculation. Defaults to 'bondi'.

        Returns:
            pd.DataFrame: A DataFrame with the Sterimol parameters.
            
            to add
            - only_sub- sterimol of only one part - i only have that.
            - drop some atoms.
        """
        if base_atoms is None:
            base_atoms = get_sterimol_indices(self.xyz_df, self.bonds_df)

        if isinstance(base_atoms[0], list):
            # If base_atoms is a list of lists, process each group individually and concatenate the results
            sterimol_list = [self.process_sterimol_atom_group(atoms, radii) for atoms in base_atoms]
            sterimol_df = pd.concat(sterimol_list, axis=0)

        else:
            # If base_atoms is a single group, just process that group
            sterimol_df = self.process_sterimol_atom_group(base_atoms, radii)
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

    def get_nbo_df(self, atoms_indices: List[int], diff_indices: List[int] = None) -> pd.DataFrame:
        """
        Returns a DataFrame with the NBO charges for the specified atoms.

        Args:
            atoms_indices (List[int]): The indices of the atoms to include in the DataFrame.
            diff_indices (List[int], optional): The indices of the atoms to calculate charge differences for.
                If None, no charge differences are calculated. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame with the NBO charges and charge differences (if specified).
        input: molecule.get_nbo_df([5,7,3,4],[[1,2],[3,4]])
        output: 	atom_5	atom_7	atom_3	atom_4	diff_1-2	diff_3-4
    	-0.26153	-0.28229	-0.03702	-0.19552	0.13263	0.1585
        """
        atoms_indices=adjust_indices(atoms_indices)
        nbo_df=(self.charge_df.iloc[atoms_indices]).rename(index=lambda x: f'atom_{x + 1}').T
        if diff_indices is not None:
            diff_list=[]
            if isinstance(diff_indices, list) and all(isinstance(elem, list) for elem in diff_indices):
                for atoms in diff_indices:
                    atoms=adjust_indices(atoms)
                    diff_list.append(pd.DataFrame(self.charge_df.iloc[atoms[0]]-self.charge_df.iloc[atoms[1]],columns=[f'diff_{atoms[0]+1}-{atoms[1]+1}']))
                diff_df=pd.concat(diff_list,axis=1)
                nbo_df=pd.concat([nbo_df,diff_df],axis=1)
            else:
                diff_indices=adjust_indices(diff_indices)
                diff=pd.DataFrame(self.charge_df.iloc[diff_indices[0]]-self.charge_df.iloc[diff_indices[1]],columns=[f'diff_{diff_indices[0]+1}-{diff_indices[1]+1}'])
                nbo_df=pd.concat([nbo_df,diff],axis=1)
        return nbo_df
    


         

    def get_coordination_transformation_df(self, base_atoms_indices: List[int]) -> pd.DataFrame:
        """
        Returns a new DataFrame with the coordinates transformed based on the specified base atoms.

        Args:
            base_atoms_indices (List[int]): The indices of the base atoms to use for the transformation.

        Returns:
            pd.DataFrame: A new DataFrame with the transformed coordinates.
        """
        new_coordinates_df = preform_coordination_transformation(self.xyz_df, base_atoms_indices)
        return new_coordinates_df

    ## not working after renumbering for some reason
    
    
    def get_npa_df_single(self, atoms: List[int], sub_atoms):
                ## activate this option - the origin vector should change not the index
        # if sub_atoms is not None:
        #     if isinstance(sub_atoms, list) and all(isinstance(elem, list) for elem in sub_atoms):
        #         for sub,atoms in zip(sub_atoms,base_atoms_indices):
        #             sub = adjust_indices(sub)
        #             origin= center_substructure(self.coordinates_array, sub)
        #             atoms[0]=origin
        #     else:
        #         sub_atoms = adjust_indices(sub_atoms)
        #         origin = center_substructure(self.coordinates_array, sub_atoms)
        #         base_atoms_indices[0] = origin
        coordinates_array = np.array(
            self.get_coordination_transformation_df(atoms)[['x', 'y', 'z']].astype(float))
        charges = np.array(self.charge_df)
        # print(f'len of charges is {len(charges)}, len of coordinates is {len(coordinates_array)}')
        npa_df = calc_npa_charges(coordinates_array, charges, sub_atoms)
        npa_df = npa_df.rename(index={0: f'NPA_{atoms[0]}-{atoms[1]}-{atoms[2]}'})
        return npa_df

    def get_npa_df(self, base_atoms_indices: List[int], sub_atoms: Union[List[int], None] = None) -> pd.DataFrame:
        """
        Returns a DataFrame with the NPA charges calculated based on the specified base atoms and sub atoms.

        Args:
            base_atoms_indices (List[int]): The indices of the base atoms to use for the NPA calculation.
            sub_atoms (Union[List[int], None], optional): The indices of the sub atoms to use for the NPA calculation. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame with the NPA charges.
        """
        if isinstance(base_atoms_indices[0], list):
            # If base_atoms_indices is a list of lists, process each group individually and concatenate the results
            npa_list = [self.get_npa_df_single(atoms, sub_atoms) for atoms in base_atoms_indices]
            npa_df = pd.concat(npa_list, axis=0)
        else:
            # If base_atoms_indices is a single group, just process that group
            npa_df = self.get_npa_df_single(base_atoms_indices, sub_atoms)
        return npa_df

    
    def get_dipole_gaussian_df_single(self, atoms):
        dipole_df = calc_dipole_gaussian(self.coordinates_array, np.array(self.gauss_dipole_df), atoms)
        dipole_df = dipole_df.rename(index={0: f'dipole_{atoms[0]}-{atoms[1]}-{atoms[2]}'})
        return dipole_df

    def get_dipole_gaussian_df(self, base_atoms_indices: List[int]) -> pd.DataFrame:
        """
        Returns a DataFrame with the dipole moments calculated based on the specified base atoms.

        Args:
            base_atoms_indices (List[int]): The indices of the base atoms to use for the dipole moment calculation.

        Returns:
            pd.DataFrame: A DataFrame with the dipole moments.
        """
        if isinstance(base_atoms_indices[0], list):
            # If base_atoms_indices is a list of lists, process each group individually and concatenate the results
            dipole_list = [self.get_dipole_gaussian_df_single(atoms) for atoms in base_atoms_indices]
            dipole_df = pd.concat(dipole_list, axis=0)
        else:
            # If base_atoms_indices is a single group, just process that group
            dipole_df = self.get_dipole_gaussian_df_single(base_atoms_indices)
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

    def get_stretch_vibration_single(self, atom_pair: List[int])-> pd.DataFrame:
        if check_pair_in_bonds(atom_pair, self.bonds_df) == True:
            try:
                extended_vib_df = calc_vibration_dot_product_from_pairs(
                    self.coordinates_array, self.vibration_dict, atom_pair, self.info_df
                )
            except TypeError:
                print('Error: no vibration array for the molecule')
                return None
            vibration_df, idx = calc_max_frequency_gen_vibration(extended_vib_df)
            return vibration_df.rename(index={idx: f'Stretch_{atom_pair[0]}_{atom_pair[1]}'})
        else:
            print('Error: the following bonds do not exist-check atom numbering')
            return None
    
    def get_stretch_vibration(self, atom_pairs: List[int])-> pd.DataFrame:
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
            vibration_list = [self.get_stretch_vibration_single(pair) for pair in atom_pairs]
            # Filter out None results
            vibration_list = [vib for vib in vibration_list if vib is not None]
            vibration_df = pd.concat(vibration_list, axis=0)
            return vibration_df
        else:
            # If atom_pairs is a single pair, just process that pair
            return self.get_stretch_vibration_single(atom_pairs)
        
    
    def get_ring_vibrations(self,ring_atom_indices: List[List[int]])-> pd.DataFrame:
        """

        Parameters
        ----------
        ring_atom_indices :working example: molecule_1.get_ring_vibrations([[19,20],[19,21],[20,22]]) 
            a list of atom pairs, there must be a vibration file with a corresponding number to work.
            
            # Enter a list of ring atoms with the order: primary axis - para followed by primary,
            # ortho - ortho atoms relative to primary atom and meta - meta atoms relative to primary.
            # For example - for a ring of atoms 1-6 where 4 is connected to the main group and 1 is para to it
            # (ortho will be 3 & 5 and meta will be 2 & 6) - enter the input [[1,4],[3,5],[2,6]].
        Returns
        -------
        dataframe
            cross  cross_angle      para  para_angle
        0  657.3882    81.172063  834.4249   40.674833

        """
        try:
            filtered_df=get_filtered_ring_df(self.info_df,self.coordinates_array,self.vibration_dict,ring_atom_indices)
        except FileNotFoundError:
            return print('no vibration data file exists for this atom')
        # bool_check=filtered_df[2].duplicated().any() #check for duplicates in calc, is it needed ?? slim chances
        return calc_min_max_ring_vibration(filtered_df)
    
    def get_bend_vibration_single(self, atom_pair: List[int], threshold: float = 1300)-> pd.DataFrame:
        # Create the adjacency dictionary for the pair of atoms
        adjacency_dict = create_adjacency_dict_for_pair(self.bonds_df, atom_pair)
        center_atom_exists = find_center_atom(atom_pair[0], atom_pair[1], adjacency_dict)
        if not center_atom_exists:
            print('Atoms do not share a center')
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
        for feather_file in os.listdir(): 
            if feather_file.endswith('.feather'):
                try:
                    self.molecules.append(Molecule(feather_file))
                except:
                    self.failed_molecules.append(feather_file)
                    print(f'Error: {feather_file} could not be processed')
                    failed_file = feather_file.rsplit('.feather', 1)[0] + '.feather_fail'
                # self.molecules.append(Molecule(log_file))
        self.molecules_names=[molecule.molecule_name for molecule in self.molecules]
        self.old_molecules=self.molecules
        self.old_molecules_names=self.molecules_names
        os.chdir('../')

        if renumber:
            self.renumber_molecules()

    

    def renumber_molecules(self):
        
        os.chdir(self.molecules_path)
        try:
            os.mkdir('xyz_files')
        except FileExistsError:
            pass
        directories = [d for d in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), d)) and d[-1].isdigit()]
        if directories:
            os.chdir(directories[0])
            print(f'renumbering files exist {os.getcwd(),os.listdir()}')
        else:
            os.chdir('xyz_files')
            [mol.write_xyz_file() for mol in self.molecules]
            # renumbered_indices=[list(df.index) for df in renumbered_list]
            renumbering.batch_renumbering(os.getcwd())
            os.chdir('../')
            directories = [d for d in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), d)) and d[-1].isdigit()]
            os.chdir(directories[0])
        ## enter the new dir by searching for it
        # directories = [d for d in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), d)) and d[-1].isdigit()]
        xyz_files=[file for file in os.listdir() if file.endswith('.xyz')]
        xyz_dfs=[help_functions.get_df_from_file(file) for file in xyz_files] 
        # self.molecules=[Molecule(log_file) for log_file in os.listdir() if log_file.endswith('.log')]
        self.molecules=[]
        self.failed_molecules=[]
        os.chdir(self.molecules_path)
        for feather_file, df in zip(os.listdir(), xyz_dfs): 
            if feather_file.endswith('.feather'):
                try:
                    self.molecules.append(Molecule(feather_file,parameter_list=None, new_xyz_df=df))
                except:
                    self.failed_molecules.append(feather_file)
                    print(f'Error: {feather_file} could not be processed')
                # self.molecules.append(Molecule(log_file))
         
        self.molecules_names=[molecule.molecule_name for molecule in self.molecules]
        self.old_molecules=self.molecules
        self.old_molecules_names=self.molecules_names
        os.chdir('../')
    


    def filter_molecules(self, indices):
        self.molecules = [self.molecules[i] for i in indices]
        self.molecules_names = [self.molecules_names[i] for i in indices]

    def get_sterimol_dict(self,base_atoms):
        sterimol_dict={}
        for molecule in self.molecules:
            sterimol_dict[molecule.molecule_name]=molecule.get_sterimol(base_atoms)
        return sterimol_dict
    
    def get_npa_dict(self,base_atoms):
        npa_dict={}
        for molecule in self.molecules:
            npa_dict[molecule.molecule_name]=molecule.get_npa_df(base_atoms)
        return npa_dict
    
    # def get_stretch_dict(self,atom_pairs):
    #     stretch_dict={}
    #     for molecule in self.molecules:
    #         stretch_dict[molecule.molecule_name]=molecule.get_stretch_vibration(atom_pairs)
    #     return stretch_dict
    
    def get_ring_vibration_dict(self,ring_atom_indices):
        ring_dict={}
        for molecule in self.molecules:
            ring_dict[molecule.molecule_name]=molecule.get_ring_vibrations(ring_atom_indices)
        return ring_dict
    
    def get_dipole_dict(self,base_atoms):
        dipole_dict={}
        for molecule in self.molecules:
            dipole_dict[molecule.molecule_name]=molecule.get_dipole_gaussian_df(base_atoms)
        return dipole_dict
    
    def get_bond_angle_dict(self,atom_indices):
        bond_angle_dict={}
        for molecule in self.molecules:
            bond_angle_dict[molecule.molecule_name]=molecule.get_bond_angle(atom_indices)
        return bond_angle_dict
    
    def get_bond_length_dict(self,atom_pairs):
        bond_length_dict={}
        for molecule in self.molecules:
            bond_length_dict[molecule.molecule_name]=molecule.get_bond_length(atom_pairs)
        return bond_length_dict
    
    def get_stretch_vibration_dict(self,atom_pairs):
        stretch_vibration_dict={}
        for molecule in self.molecules:
            stretch_vibration_dict[molecule.molecule_name]=molecule.get_stretch_vibration(atom_pairs)
        return stretch_vibration_dict
    
    def get_nbo_dict(self,atoms_indices,diff_indices=None):
        nbo_dict={}
        for molecule in self.molecules:
            nbo_dict[molecule.molecule_name]=molecule.get_nbo_df(atoms_indices,diff_indices)
        return nbo_dict
    
    def get_bend_vibration_dict(self,atom_pairs,threshold=1300):
        bending_dict={}
        for molecule in self.molecules:
            bending_dict[molecule.molecule_name]=molecule.get_bend_vibration(atom_pairs,threshold)
        return bending_dict
    
    def visualize_molecules(self):
        for molecule in self.molecules:
            molecule.visualize_molecule()
        

    
    def get_molecules_comp_set_app(self,answers_dict: dict,
                                   dipole_mode = 'gaussian', radii = 'bondi', export_csv=False, answers_list_load=None):
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
        print(f'answers dict:{answers_dict}')
        for question, answer in answers_dict.items():
            if answer is not None:
                #TODO chagne so if its only one set of atoms it will be a list and not a list of lists for first parameter
                if question.split(' ')[0]=='NBO':
                    try:
                        if ' ' in answer:
                            single_numbers_str, pairs_str = answer.split(' ', 1)
                            # Convert the second part into pairs and convert each pair to a list of numbers
                            pairs = [[int(num) for num in pair.split(',')] for pair in pairs_str.split(' ')]
                        else:
                            single_numbers_str = answer

                        # Convert the first part to a list of numbers
                        single_numbers = [int(num) for num in single_numbers_str.split(',')]

                        # Depending on whether pairs exist, add to the dictionary
                        if ' ' in answer:
                            answers_dict[question] = [single_numbers, pairs]
                        else:
                            answers_dict[question] = [single_numbers]

                        answers_list.append(answers_dict[question])
                    except ValueError:
                        answers_list.append([])
                else:
                    answers_dict[question]=help_functions.convert_to_list_or_nested_list(answer)
                    answers_list.append(answers_dict[question])
            else:
                answers_list.append([])
            
        if answers_list_load is not None:
            answers_list=answers_list_load
        
        if len(answers_list) != 8:
            answers_list.insert(4, [])
            
        print(f'answers list:{answers_list}')
        res_df=pd.DataFrame()
        if answers_list[0] != []:
          
            res_df=help_functions.dict_to_horizontal_df(self.get_ring_vibration_dict(answers_list[0])) ### get_ring_vibration_dict change all of them later
        if answers_list[1] and answers_list[1]!= []:
            res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df(self.get_stretch_vibration_dict(answers_list[1]))],axis=1)
        if answers_list[2] and answers_list[2]!= []:
            try:
                res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df(self.get_bend_vibration_dict(answers_list[2]))],axis=1)
            except AttributeError:
                pass
        if answers_list[3] and answers_list[3]!= []:
            if dipole_mode=='compute':
                res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df(self.get_npa_dict(answers_list[3]))],axis=1)
            elif dipole_mode=='gaussian':
                print(f'params:{answers_list[3]}')
                res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df(self.get_dipole_dict(answers_list[3]))],axis=1)
        if answers_list[4] and answers_list[4]!= []:
            print(f'params:{answers_list[4]}')
            res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df(self.get_nbo_dict(answers_list[4][0],answers_list[4][1]))],axis=1)
        if answers_list[5] and answers_list[5]!= []:
            print(f'params sterimol:{answers_list[5]}')
            if radii=='CPK':
                res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df((self.get_sterimol_dict(answers_list[5])))],axis=1) ## add cpk and bondi
            else:
                res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df((self.get_sterimol_dict(answers_list[5])))],axis=1)
        if answers_list[6] and answers_list[6]!= []:
            res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df(self.get_bond_length_dict(answers_list[6]))],axis=1)
        if answers_list[7] and answers_list[7]!= []:
            res_df=pd.concat([res_df,help_functions.dict_to_horizontal_df(self.get_bond_angle_dict(answers_list[7]))],axis=1)
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
        # os.mkdir('xyz_files')
        # os.chdir('xyz_files')
        # for molecule in self.molecules:
        #     dataframe_to_xyz(molecule.xyz_df,(molecule.molecule_name+'.xyz'))

if __name__=='__main__':
    os.chdir(r'C:\Users\edens\Documents\GitHub\Automation_code-main\M2_data_extractor\gauss_log')
    molecules=Molecule('gauss_output.feather')