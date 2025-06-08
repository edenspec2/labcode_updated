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
    print("coef_mat shape:", np.shape(coef_mat), "coef_mat:", coef_mat)
    print("result_vector:", result_vector)
    print("angle_new_y_coplane:", angle_new_y_coplane)
    new_x, _, _, _ = np.linalg.lstsq(coef_mat, result_vector, rcond=None)
    new_basis = np_cross_and_vstack(new_x, y)
 
    return new_basis



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

    dipole_df=pd.DataFrame(array_dipole,index=XYZConstants.DIPOLE_COLUMNS.value).T
 
    return dipole_df

def calc_dipole_gaussian(coordinates_array, gauss_dipole_array, base_atoms_indices, origin):
    """
    a function that recives coordinates and gaussian dipole, transform the coordinates
    by the new base atoms and calculates the dipole in each axis
    """



    indices=adjust_indices(base_atoms_indices)
    if origin is None:
        new_origin = coordinates_array[indices[0]]
    else:
        orig_indices = adjust_indices(origin)
        new_origin = np.mean(coordinates_array[orig_indices], axis=0)

    # Recenter the coordinates array using the new origin
    recentered_coords = coordinates_array - new_origin
    basis_vector=calc_basis_vector(*calc_new_base_atoms(recentered_coords, indices))

    gauss_dipole_array = [np.concatenate((np.matmul(basis_vector, gauss_dipole_array[0, 0:3]), [gauss_dipole_array[0, 3]]))]

    if isinstance(gauss_dipole_array, pd.Series) or (isinstance(gauss_dipole_array, np.ndarray) and gauss_dipole_array.ndim == 1):
        gauss_dipole_array = np.expand_dims(gauss_dipole_array, axis=0)

    dipole_df=pd.DataFrame(gauss_dipole_array,columns=['dipole_x','dipole_y','dipole_z','total'])
    
    return dipole_df