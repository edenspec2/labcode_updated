"""
sterimol_utils.py
Utility functions for Sterimol parameter calculation and related geometry operations.
"""

import numpy as np
import pandas as pd 

# Helper functions

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

def generate_circle(center_x, center_y, radius, n_points=20):
    """
    Generate circle coordinates given a center and radius.
    Returns a DataFrame with columns 'x' and 'y'.
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return np.column_stack((x, y))

def direction_atoms_for_sterimol(bonds_df, base_atoms) -> list:
    base_atoms_copy = base_atoms[0:2]
    origin, direction = base_atoms[0], base_atoms[1]
    bonds_df = bonds_df[~((bonds_df[0] == origin) & (bonds_df[1] == direction)) & 
                        ~((bonds_df[0] == direction) & (bonds_df[1] == origin))]
    try:
        base_atoms[2] == origin
        if(any(bonds_df[0] == direction)):
            base_atoms_copy[2] = int(bonds_df[(bonds_df[0] == direction)][1].iloc[1])
        else:
            base_atoms_copy[2] = int(bonds_df[(bonds_df[1] == direction)][0].iloc[1])
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

def preform_coordination_transformation(xyz_df, indices=None, origin=None):
    xyz_copy = xyz_df.copy()
    coordinates = np.array(xyz_copy[['x', 'y', 'z']].values)
    if origin is not None:
        geometric_origin = np.mean(xyz_df.iloc[indices][['x', 'y', 'z']].values, axis=0)
    else:
        geometric_origin = None
    if indices is None:
        transformed = calc_coordinates_transformation(coordinates, [1, 2, 3], origin=geometric_origin)
    else:
        transformed = calc_coordinates_transformation(coordinates, indices, origin=geometric_origin)
    xyz_copy[['x', 'y', 'z']] = transformed
    return xyz_copy

def get_molecule_connections(bonds_df, source, direction):
    import igraph as ig
    from extractor_utils import help_functions
    graph = ig.Graph.DataFrame(edges=bonds_df, directed=False)
    paths = graph.get_all_simple_paths(v=source, mode='all')
    with_direction = [path for path in paths if (direction in path)]
    longest_path = np.unique(help_functions.flatten_list(with_direction))
    return longest_path

def get_specific_bonded_atoms_df(bonds_df, longest_path, coordinates_df):
    from extractor_utils import help_functions
    if longest_path is not None:
        mask = bonds_df.isin(longest_path)
        edited_bonds_df = bonds_df[mask.any(axis=1)].dropna().reset_index(drop=True)
    else:
        edited_bonds_df = bonds_df.copy()
    if edited_bonds_df.empty:
        return pd.DataFrame(columns=help_functions.XYZConstants.BONDED_COLUMNS.value)
    bonds_array = (edited_bonds_df.values - 1).astype(int)
    atom_bonds_list = []
    for i, bond in enumerate(bonds_array):
        coords = coordinates_df.iloc[bond]['atom'].values
        atom_bonds_list.append(coords)
    atom_bonds = np.vstack(atom_bonds_list).reshape(-1, 2)
    bonded_atoms_df = pd.concat([pd.DataFrame(atom_bonds), edited_bonds_df], axis=1)
    bonded_atoms_df.columns = help_functions.XYZConstants.BONDED_COLUMNS.value
    return bonded_atoms_df

def filter_atoms_for_sterimol(bonded_atoms_df, coordinates_df):
    allowed_bonds_indices = pd.concat([bonded_atoms_df['index_1'], bonded_atoms_df['index_2']], axis=1).reset_index(drop=True)
    atom_filter = adjust_indices(np.unique([atom for sublist in allowed_bonds_indices.values.tolist() for atom in sublist]))
    edited_coordinates_df = coordinates_df.loc[atom_filter].reset_index(drop=True)
    return edited_coordinates_df

def get_extended_df_for_sterimol(coordinates_df, bonds_df, radii='CPK'):
    from extractor_utils import help_functions
    bond_type_map_regular = GeneralConstants.REGULAR_BOND_TYPE.value
    bond_type_map = GeneralConstants.BOND_TYPE.value
    radii_map = GeneralConstants.CPK_RADII.value if radii == 'CPK' else GeneralConstants.BONDI_RADII.value
    df = coordinates_df.copy()
    if radii == 'bondi':
        df['atype'] = df['atom']
    else:
        df['atype'] = nob_atype(coordinates_df, bonds_df)
    df['magnitude'] = calc_magnitude_from_coordinates_array(df[['x', 'z']].astype(float))
    df['radius'] = df['atype'].map(radii_map)
    df['B5'] = df['radius'] + df['magnitude']
    df['L'] = df['y'] + df['radius']
    return df

def b1s_for_loop_function(extended_df, b1s, b1s_loc, degree_list, plane, b1_planes):
    results = []
    for degree in degree_list:
        transformed_plane = get_transfomed_plane_for_sterimol(plane, degree)
        max_x = np.max(transformed_plane[:, 0])
        min_x = np.min(transformed_plane[:, 0])
        max_y = np.max(transformed_plane[:, 1])
        min_y = np.min(transformed_plane[:, 1])
        avs = np.abs([max_x, min_x, max_y, min_y])
        min_val = np.min(avs)
        min_index = np.argmin(avs)
        if min_index == 0:
            b1_coords = (max_x, 0)
        elif min_index == 1:
            b1_coords = (min_x, 0)
        elif min_index == 2:
            b1_coords = (0, max_y)
        else:
            b1_coords = (0, min_y)
        norms_sq = np.sum(transformed_plane ** 2, axis=1)
        b5_index = np.argmax(norms_sq)
        b5_point = transformed_plane[b5_index]
        b5_value = np.linalg.norm(b5_point)
        angle_b1 = np.arctan2(b1_coords[1], b1_coords[0]) % (2 * np.pi)
        angle_b5 = np.arctan2(b5_point[1], b5_point[0]) % (2 * np.pi)
        angle_diff = abs(angle_b5 - angle_b1)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        B1_B5 = np.degrees(angle_diff)
        B1 = min_val
        b1_planes.append(transformed_plane)
        results.append({
            'degree': degree,
            'B1': B1,
            'B1_B5_angle': B1_B5,
            'b1_coords': b1_coords,
            'b5_value': b5_value,
            'plane': transformed_plane
        })
    sterimol_df = pd.DataFrame(results)
    return sterimol_df

def get_b1s_list(extended_df, scans=90 // 5, plot_result=False):
    b1s, b1s_loc, b1_planes = [], [], []
    degree_list = list(range(18, 108, scans))
    circles = []
    for idx, row in extended_df.iterrows():
        circle_points = generate_circle(row['x'], row['z'], row['radius'], n_points=100)
        circles.append(circle_points)
    plane = np.vstack(circles)
    sterimol_df = b1s_for_loop_function(extended_df, b1s, b1s_loc, degree_list, plane, b1_planes)
    b1s = sterimol_df['B1']
    try:
        back_ang = degree_list[np.where(b1s == min(b1s))[0][0]] - scans
        front_ang = degree_list[np.where(b1s == min(b1s))[0][0]] + scans
        new_degree_list = range(back_ang, front_ang + 1)
    except:
        back_ang = degree_list[np.where(np.isclose(b1s, min(b1s), atol=1e-8))[0][0]] - scans
        front_ang = degree_list[np.where(np.isclose(b1s, min(b1s), atol=1e-8))[0][0]] + scans
        new_degree_list = range(back_ang, front_ang + 1)
    b1s, b1s_loc, b1_planes = [], [], []
    sterimol_df = b1s_for_loop_function(extended_df, b1s, b1s_loc, list(new_degree_list), plane, b1_planes)
    b1s = sterimol_df['B1']
    b1_b5_angle = sterimol_df['B1_B5_angle']
    plane = sterimol_df['plane']
    return [b1s, b1_b5_angle, plane]

def calc_sterimol(bonded_atoms_df, extended_df, visualize=False):
    edited_coordinates_df = filter_atoms_for_sterimol(bonded_atoms_df, extended_df)
    b1s, b1_b5_angle, plane = get_b1s_list(edited_coordinates_df)
    valid_indices = np.where(b1s >= 0)[0]
    best_idx = valid_indices[np.argmin(b1s[valid_indices])]
    best_b1_plane = plane[best_idx]
    max_x = np.max(best_b1_plane[:, 0])
    min_x = np.min(best_b1_plane[:, 0])
    max_y = np.max(best_b1_plane[:, 1])
    min_y = np.min(best_b1_plane[:, 1])
    avs = np.abs([max_x, min_x, max_y, min_y])
    min_val = np.min(avs)
    B1 = min_val
    b1_index = np.where(b1s == B1)[0][0]
    angle = b1_b5_angle[b1_index]
    norms_sq = np.sum(best_b1_plane ** 2, axis=1)
    b5_index = np.argmax(norms_sq)
    b5_point = best_b1_plane[b5_index]
    b5_value = np.linalg.norm(b5_point)
    max_row = edited_coordinates_df['B5'].idxmax()
    max_row = int(max_row)
    L = max(edited_coordinates_df['L'].values)
    loc_B5 = edited_coordinates_df['y'].iloc[np.where(edited_coordinates_df['B5'] == max(edited_coordinates_df['B5']))[0][0]]
    sterimol_df = pd.DataFrame([B1, b5_value, L, loc_B5, angle], index=['B1', 'B5', 'L', 'loc_B5', 'B1_B5_angle'])
    if visualize:
        pass  # Visualization code can be added here
    return sterimol_df.T

def get_sterimol_base_atoms(center_atom, bonds_df):
    center_atom_id = int(center_atom.name) + 1
    base_atoms = [center_atom_id]
    if (any(bonds_df[0] == center_atom_id)):
        base_atoms.append(int(bonds_df[(bonds_df[0] == center_atom_id)][1].iloc[0]))
    else:
        base_atoms.append(int(bonds_df[(bonds_df[1] == center_atom_id)][0].iloc[0]))
    return base_atoms

def get_sterimol_indices(coordinates, bonds_df):
    center = get_center_of_mass(coordinates)
    center_atom = get_closest_atom_to_center(coordinates, center)
    base_atoms = get_sterimol_base_atoms(center_atom, bonds_df)
    return base_atoms

# You may need to add or import additional helper functions such as:
# - calc_coordinates_transformation
# - nob_atype
# - calc_magnitude_from_coordinates_array
# - get_center_of_mass
# - get_closest_atom_to_center
# - get_transfomed_plane_for_sterimol 