import numpy as np
import pandas as pd 

try:
    from ...utils import help_functions, visualize
except ImportError:
    from utils import help_functions, visualize

def get_center_of_mass(xyz_df):
    coordinates=np.array(xyz_df[['x','y','z']].values,dtype=float)
    atoms_symbol=np.array(xyz_df['atom'].values)
    masses=np.array([help_functions.GeneralConstants.ATOMIC_WEIGHTS.value[symbol] for symbol in atoms_symbol])
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
    atom_indices=help_functions.adjust_indices(atom_indices)
    substructure=coordinates_array[atom_indices]
    center_substructure=np.mean(substructure,axis=0)
    return center_substructure



def get_sterimol_indices(coordinates,bonds_df):
    center=get_center_of_mass(coordinates)
    center_atom=get_closest_atom_to_center(coordinates,center)
    base_atoms=get_sterimol_base_atoms(center_atom,bonds_df)
    return base_atoms

def filter_atoms_for_sterimol(bonded_atoms_df,coordinates_df):
    """
   
    """
    allowed_bonds_indices= pd.concat([bonded_atoms_df['index_1'],bonded_atoms_df['index_2']],axis=1).reset_index(drop=True)
    atom_filter=help_functions.adjust_indices(np.unique([atom for sublist in allowed_bonds_indices.values.tolist() for atom in sublist]))
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

    df['magnitude'] = calc_magnitude_from_coordinates_array(df[['x', 'z']].astype(float))
    
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
    graph=ig.Graph.DataFrame(edges=bonds_df,directed=False)
    paths=graph.get_all_simple_paths(v=source,mode='all')
   
    with_direction=[path for path in paths if (direction in path)]
    # print('with_direction:', with_direction)
    longest_path=np.unique(help_functions.flatten_list(with_direction))
    # take the longest path
    longest=max(with_direction, key=len)
    
    return longest_path

def find_longest_simple_path(bonds_df):
    """
    bonds_df : pandas.DataFrame with exactly two columns [u, v] (integer atom indices)
    
    Returns
    -------
    List[int]
        The sequence of atom indices along the longest simple path (graph diameter)
        in the undirected molecule graph.
    """
    # 1) Build an undirected igraph from your bond list
    g = ig.Graph.DataFrame(edges=bonds_df, directed=False)
    
    # 2) Compute the diameter: the furthest‐apart pair of nodes and the path between them
    diameter_path = g.get_diameter()  # returns a list of vertex indices
    
    # 3) Convert to plain Python ints and return
    return [int(v) for v in diameter_path]

    
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
    circles_y = []
    circles = []
    for idx, row in extended_df.iterrows():
        circle_points = generate_circle(row['x'], row['z'], row['radius'], n_points=100)
        circles.append(circle_points)
        circle_points_y = generate_circle(row['x'], row['z'], row['radius'], n_points=100)
        circles_y.append(circle_points_y)
    plane_xz = np.vstack(circles)  # All circle points combined.
    plane_yz = np.vstack(circles_y)  # All circle points combined.

    sterimol_df=b1s_for_loop_function(extended_df, b1s, b1s_loc, degree_list, plane_xz, b1_planes)

    b1s=sterimol_df['B1']
    
    try:
        back_ang=degree_list[np.where(b1s==min(b1s))[0][0]]-scans   
        front_ang=degree_list[np.where(b1s==min(b1s))[0][0]]+scans
        new_degree_list=range(back_ang,front_ang+1)
    except:
        
        back_ang=degree_list[np.where(np.isclose(b1s, min(b1s), atol=1e-8))[0][0]]-scans
        front_ang=degree_list[np.where(np.isclose(b1s, min(b1s), atol=1e-8))[0][0]]+scans
        new_degree_list=range(back_ang,front_ang+1)
    # plane=sterimol_df['plane']
    best_idx = np.where(sterimol_df['B1']==sterimol_df['B1'].min())[0][0]
    # 4) *you should* take that rotated plane:
    plane_xz = sterimol_df.loc[best_idx, 'plane']

    b1s, b1s_loc, b1_planes = [], [], []
    sterimol_df=b1s_for_loop_function(extended_df, b1s, b1s_loc, list(new_degree_list), plane_xz, b1_planes)

    b1s=sterimol_df['B1']

    b1_b5_angle=sterimol_df['B1_B5_angle']
    plane_xz=sterimol_df['plane']
    

    return [b1s, b1_b5_angle, plane_xz]




def calc_sterimol(bonded_atoms_df,extended_df,visualize_bool=False):
    edited_coordinates_df=filter_atoms_for_sterimol(bonded_atoms_df,extended_df)

    b1s,b1_b5_angle,plane=get_b1s_list(edited_coordinates_df)
   
    valid_indices = np.where(b1s >= 0)[0]
    best_idx = valid_indices[np.argmin(b1s[valid_indices])]
    best_b1_plane = plane[best_idx]

    max_x = np.max(best_b1_plane[:, 0])
    min_x = np.min(best_b1_plane[:, 0])
    max_y = np.max(best_b1_plane[:, 1])
    min_y = np.min(best_b1_plane[:, 1])
    avs = np.abs([max_x, min_x, max_y, min_y])
    min_val = np.min(avs)
    B1= min_val
    b1_index=np.where(b1s==B1)[0][0]
    angle=b1_b5_angle[b1_index]
    norms_sq = np.sum(best_b1_plane**2, axis=1)
    b5_index = np.argmax(norms_sq)
    b5_point = best_b1_plane[b5_index]
    b5_value = np.linalg.norm(b5_point)
    # loc_B1=max(b1s_loc[np.where(b1s[b1s>=0]==min(b1s[b1s>=0]))])
    # get the idx of the row with the  biggest b5 value from edited
    max_row=edited_coordinates_df['B5'].idxmax()
    max_row=int(max_row)
    L=max(edited_coordinates_df['L'].values)
    loc_B5 = edited_coordinates_df['y'].iloc[np.where(edited_coordinates_df['B5']==max(edited_coordinates_df['B5']))[0][0]]
    
    sterimol_df = pd.DataFrame([B1, b5_value, L ,loc_B5,angle], index=help_functions.XYZConstants.STERIMOL_INDEX.value)
    if visualize_bool:
        visualize.plot_b1_visualization(best_b1_plane, edited_coordinates_df)
        visualize.plot_L_B5_plane(edited_coordinates_df, sterimol_df.T)
        
    
    return sterimol_df.T 


def get_sterimol_df(coordinates_df, bonds_df, base_atoms,connected_from_direction, radii='bondi', sub_structure=True, drop_atoms=None,visualize_bool=False):

    if drop_atoms is not None:
        drop_atoms=help_functions.adjust_indices(drop_atoms)
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
    sterimol_df = calc_sterimol(bonded_atoms_df, extended_df,visualize_bool)
    sterimol_df= sterimol_df.rename(index={0: str(base_atoms[0]) + '-' + str(base_atoms[1])})
    sterimol_df = sterimol_df.round(4)
    
    return sterimol_df