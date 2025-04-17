import pandas as pd
import numpy as np
import os
import glob
import re
from enum import Enum
import tkinter as tk
from tkinter import filedialog
from typing import *
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import shutil
import fileinput

class FileExtensions(Enum):
    """
    Hold commonly used file extensions
    """
    SMI='.smi'
    XYZ='xyz'
    CSV='.csv'
    ZIP='.zip'
    PPT='.ppt'
    CIF='.cif'
    MOL='.mol'
    PDB='.pdb'

class XYZConstants(Enum):
    """
    Constants related to XYZ file processing
    """
    DF_COLUMNS=['atom','x','y','z']
    STERIMOL_INDEX = ['B1', 'B5', 'L','loc_B5','B1_B5_angle']
    DIPOLE_COLUMNS = ['dip_x', 'dip_y', 'dip_z', 'total_dipole']
    RING_VIBRATION_COLUMNS = ['cross', 'cross_angle', 'para', 'para_angle']
    RING_VIBRATION_INDEX=['Product','Frequency','Sin_angle']
    VIBRATION_INDEX = ['Frequency', 'Amplitude']
    BONDED_COLUMNS = ['atom_1', 'atom_2', 'index_1', 'index_2']
    NOF_ATOMS = ['N', 'O', 'F']
    STERIC_PARAMETERS = ['B1', 'B5', 'L', 'loc_B1', 'loc_B5','RMSD']
    ELECTROSTATIC_PARAMETERS = ['dip_x', 'dip_y', 'dip_z', 'total_dipole','energy']
    CHARGE_TYPE = ['nbo', 'hirshfeld', 'cm5']

class LinuxCommands(Enum):
    COPY='cp'
    OBABEL_PREFIX='/gpfs0/gaus/users/itamarwa/transfer_to_itamar/build/bin/obabel '
    OBABEL_XYZ_SETTINGS_1=' -O '
    OBABEL_XYZ_SETTINGS_2=' --gen3d'
    XTB_INPUT_PREFIX='/gpfs0/gaus/projects/xtb-6.4.1/bin/xtb --input ' # xtb_di.inp 
    XTB_PREFIX='/gpfs0/gaus/projects/xtb-6.4.1/bin/xtb '
    XTB_SUFIX=' --ohess --dipole --pop'
    CREST_INPUT_PREFIX='/gpfs0/gaus/projects/crest --input '
    CREST_PREFIX='/gpfs0/gaus/projects/crest '
    GAUSS_SUFIX='/gpfs0/gaus/users/kozuch/home/scripts/gaussian/g16'
    #GAUSS_SUFIX='/gpfs0/gaus/users/edenspec/g16'


def get_file_name_list(file_identifier):
    """
    The function gets a file identifier as input and returns a list of all files in the working 
    which contain the identifier in the files name
    ----------
    Parameters
    ----------
    identifier : str.
        The wanted file identifier like 'txt','info','nbo' contained in the filename
    -------
    Returns
    -------
    list
        A list of all files in the working directory with the chosen extension 
    --------
    Examples
    --------
    
    all_files_in_dir=listdir()
    print(all_files_in_dir)
        ['0_1106253-mod-mod.xyz', '0_1106253-mod.xyz', '1106253.cif', '1109098.cif', '1_1106253-mod.xyz', 'centered_0_BASCIH.xyz', 'cif_handler.py']
        
    xyz_files_in_dir=get_filename_list('.xyz')
    print(xyz_files_in_dir)
        ['0_1106253-mod-mod.xyz', '0_1106253-mod.xyz', '1_1106253-mod.xyz', 'centered_0_BASCIH.xyz']
  
    """
    return [filename for filename in os.listdir() if file_identifier in filename]

def split_strings(strings_list):
    split_list = []
    for string in strings_list:
        split_list.extend(string.split())
    return split_list

def get_df_from_file(filename,columns=['atom','x','y','z'],index=None):
    """
    Parameters
    ----------
    filename : str
        full file name to read.
    columns : str , optional
        list of column names for DataFrame. The default is None.
    splitter : str, optional
        input for [.split().] , for csv-',' for txt leave empty. The default is None.
    dtype : type, optional
        type of variables for dataframe. The default is None.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    with open(filename, 'r') as f:
        lines=f.readlines()[2:]
    splitted_lines=split_strings(lines)
    df=pd.DataFrame(np.array(splitted_lines).reshape(-1,4),columns=columns,index=index)
    df[['x','y','z']]=df[['x','y','z']].astype(float)
    return df

# def convert_file_to_xyz_df(filename,splitter=','):
#     df=get_df_from_csv(filename,columns=XYZConstants.DF_COLUMNS.value)
#     # df['atom'].replace(main.GeneralConstants.ATOMIC_NUMBERS.value, inplace=True)
#     return df

def data_to_xyz(dataframe, output_name, comment_line=''):
    """

     a function that recieves a dataframe, output name, and comment line and creates a xyz type file.
     
    parameters

    ---

    dataframe: an array that can contain different classes of data, needs to be 4 colums to run.

    output_name:str, the name for the file created.

    comment_line: str, the headline of the file .
    ---

    examples:
    ---
    """
    if type(dataframe) == pd.DataFrame :
        number_of_atoms=dataframe.shape[0]
        atoms_np_array=dataframe.to_numpy()
    else:
        number_of_atoms=len(dataframe)
        atoms_np_array=dataframe
    with open(output_name, 'w') as xyz_file:
        xyz_file.write("{}\n{}\n".format(number_of_atoms, comment_line))
        for atom_np_array in atoms_np_array:
            try:
                xyz_file.write("{:1} {:11.6} {:11.6} {:11.6} \n".format(*atom_np_array))
            except:
                xyz_file.write("{:1}".format(*atom_np_array))


def change_filename(old_name, new_name):
    # Get the file extension
    file_extension = os.path.splitext(old_name)[1]
    # Combine the new name with the original file extension
    new_filename = new_name + file_extension
    os.rename(old_name, new_filename)

def change_filetype (filename,new_type='xyz'):
    """
    a function that recieves a file name, and a new type, and changes the type-ending of the file's name to the new one.

    parameters
    ---
    filename: str, the file we want to change

    new_type:str, the new ending we want for the file

    returns
    ---
    the same file name with a new type ending

    examples
    ---
    filename='B_THR_127_5Angs_noHOH.pdb'
    new_filename=change_filetype(filename,'xyz')
    OUTPUT:'B_THR_127_5Angs_noHOH.xyz'
    
    """
    split_result=filename.split('.')
    if '.' in new_type:
        new_filename=split_result[0]+new_type
    else:
        new_filename=split_result[0]+'.'+new_type
    return new_filename

def xyz_string_to_df(lines):
    strip_lines=[line.strip().rstrip('\n') for line in lines]
    
def create_molecule_directories():
    list_of_dirs=[name.split('.')[0] for name in os.listdir()]
    for dir_name in list_of_dirs:
        os.mkdir(dir_name)
    return

def delete_file(filename):
    """
    a function that gets a file name and deletes it.
    """
    os.remove(os.path.abspath(filename))
    return

def delete_type_files(file_type='xyz'): ## my help function to delete xyz files
    """
    a function that gets a directory path and file type, and deletes all the files of said type.
    """
    list_of_molecules=[file for file in os.listdir() if file.endswith(file_type)]
    for molecule in list_of_molecules:
        os.remove(os.path.abspath(molecule))
        
        
def move_files_directory(file_type):#need edit
    """
    a function that moves xyz type files from one directory to another.
    help function for xyz_file_generator_library to move files to the new directory created
    A void function
    """
    list_of_dirs=[name for name in os.listdir() if os.path.isdir(os.path.abspath(name))]
    list_of_files=get_file_name_list(file_type)
    print(list_of_files,list_of_dirs)
    for file_name,dir_name in zip(list_of_files,list_of_dirs):
        new_path=os.path.join(os.path.abspath(dir_name),file_name)
        os.replace(os.path.abspath(file_name),new_path)
    return


def edit_filenames_in_directory(directory_path,old: str =None, new: str =None):
    # Check if the specified directory path exists and is a directory
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # Loop through all files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Check if it's a file (not a subdirectory)
            if os.path.isfile(file_path):
                # Replace spaces, parentheses, brackets, and commas with underscores
                new_filename = filename.replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(',', '')
                new_filename = filename.replace(old, new)
                new_file_path = os.path.join(directory_path, new_filename)

                # Check if the filename needs to be changed
                if file_path != new_file_path:
                    # Rename the file in place (replace the original)
                    os.rename(file_path, new_file_path)
                    print(f"Renamed '{filename}' to '{new_filename}'")

    else:
        print("The specified directory path does not exist or is not a directory.")


def sort_files_to_directories(file_type='xyz'):
    create_molecule_directories()
    move_files_directory(file_type)
    return


## write help functions for mol2 to dfs
def string_to_int_list(string,splitter=' '):
    string=string.split(splitter)
    return [int(value) for value in string]

def split_to_pairs(string,splitter=' '):
    list_a=string_to_int_list(string)
    chunck=(len(list_a)/2)
    splited=np.array_split(list_a,chunck)
    return [value.tolist() for value in splited]

def split_for_angles(string):
    string_list=string.split('  ')
    return [string_to_int_list(value,' ') for value in string_list]

def choose_filename():
    # Create a root window
    root = tk.Tk()
    # Ask the user to select a file
    file_path = filedialog.askopenfilename()
    root.withdraw()
    # Extract the file name from the file path
    file_name = file_path.split("/")[-1]
    # Print the file name
    return file_name, file_path.split(file_name)[0]

def choose_directory():
    # Create a root window
    root = tk.Tk()
    # Ask the user to select a directory
    directory_path = filedialog.askdirectory()
    root.withdraw()
    # Extract the directory name from the directory path
    directory_name = directory_path.split("/")[-1]
    # Print the directory name
    return directory_name, directory_path

def flatten_list(nested_list_arg: List[list]) -> List:
    """
    Flatten a nested list.
    turn [[1,2],[3,4]] to [1,2,3,4]
    """
    flat_list=[item for sublist in nested_list_arg for item in sublist]
    return flat_list



def submit_to_gaussian_calculation(queue='milo'):
    # Copy file from source to destination
    os.system(LinuxCommands.COPY.value +' '+ LinuxCommands.GAUSS_SUFIX.value+ ' .')
    # Loop over all files in current directory with .com extension
    for file in os.listdir():
        if file.endswith('.com'):
            # Run the command on each file
            output_name=file.split('.')[0]+'.log'
            os.system(f'./g16 {file} {output_name} -q {queue}')
    # Remove the g16 file
    os.remove('g16')
    # Change directory back to the previous directory
    return
                                  
def calculate_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """
    Calculate a distance matrix given an array of coordinates.
    coordinates: np.array of x y z coordinates

    """
    num_atoms = len(coordinates)
    distance_matrix = np.zeros((num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            atom_i = coordinates[i]
            atom_j = coordinates[j]

            distance = np.linalg.norm(atom_i - atom_j)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def validate_geometry(distance_matrix, threshold=0.5):
    num_atoms = distance_matrix.shape[0]
    invalid_atoms = []

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = distance_matrix[i, j]

            if distance < threshold:
                invalid_atoms.append((i, j))

    return invalid_atoms

def fix_coordinates(coordinates, invalid_atoms, displacement=0.3):
    fixed_coordinates = coordinates.copy()
    for atom_i, atom_j in invalid_atoms:
        # Compute the vector between atom_i and atom_j
        vector = fixed_coordinates[atom_j] - fixed_coordinates[atom_i]
        # Normalize the vector
        normalized_vector = vector / np.linalg.norm(vector)
        # Adjust the coordinates of atom_j beyond the threshold
        fixed_coordinates[atom_j] = fixed_coordinates[atom_i] + displacement * normalized_vector

    return fixed_coordinates

def generate_3d_coordinates(smiles):
    # Generate a molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    # Generate 3D coordinates using the ETKDG method
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    # Return the molecule with 3D coordinates and hydrogen atoms
    conformer = mol.GetConformer()
    symbols = ([atom.GetSymbol() for atom in mol.GetAtoms()])
    list_of_lists = np.array([[x] for x in symbols])
    array= np.concatenate((list_of_lists, conformer.GetPositions()), axis=1)
    return array


def smiles_to_xyz_files(smiles_list: List[str], molecule_names: List[str], new_dir: bool = False):
    """
    Convert a list of SMILES strings to XYZ files.

    Args:
        smiles_list (List[str]): List of SMILES strings.
        molecule_names (List[str]): List of corresponding molecule names.
        new_dir (bool, optional): Create a new directory for XYZ files. Defaults to False.
    """
    if new_dir:
        os.mkdir('xyz_files')
        os.chdir('xyz_files')
    for smiles, name in zip(smiles_list, molecule_names):
        coordinates = generate_3d_coordinates(smiles)
        data_to_xyz(coordinates, name + '.xyz')


def replace_path_in_sh_file(filename: str, new_path: str, sh_file_path: str = r'/gpfs0/gaus/users/edenspec/Conformers_code-main/M1_outward_sender'):
    """
    Replace a specific line containing 'path=' in a shell script file with a new path.

    Args:
        filename (str): The name of the shell script file to modify.
        new_path (str): The new path to replace with.
        sh_file_path (str, optional): The directory containing the shell script file. Defaults to the specified path.
    """
    # Change the current working directory to the shell script directory
    os.chdir(sh_file_path)

    # Open the file in place for editing
    with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
        for line in file:
            # Check if the line contains 'path='
            if '    path=' in line:
                # Replace the entire line with the new path
                line = f"    path='{new_path}'\n"
            # Print the modified line or the original line if no replacement was done
            print(line, end='')



def find_files_with_extension(
    directory_path: str,
    file_extension: str,
    min_size_bytes: Optional[int] = None,
    max_size_bytes: Optional[int] = None,
    return_non_matching: bool = False
) -> List[str]:
    """
    Find files with a specific extension and optional size range in a directory.

    Args:
        directory_path (str): The path to the directory to search.
        file_extension (str): The file extension to filter (e.g., '.log').
        min_size_bytes (int, optional): The minimum file size in bytes. Defaults to None.
        max_size_bytes (int, optional): The maximum file size in bytes. Defaults to None.
        return_non_matching (bool, optional): Whether to return files that do not meet the criteria. Defaults to False.

    Returns:
        List[str]: A list of file paths that meet the criteria if return_non_matching is False,
                   otherwise, a list of file paths that do not meet the criteria.
    """
    matching_files = []
    non_matching_files = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        file_size = os.path.getsize(file_path)

        if filename.endswith(file_extension):
            if (min_size_bytes is None or file_size >= min_size_bytes) and (max_size_bytes is None or file_size <= max_size_bytes):
                matching_files.append(file_path)
            elif return_non_matching:
                non_matching_files.append(file_path)

    return non_matching_files if return_non_matching else matching_files


def move_files_to_directory(file_list: List[str], destination_directory: str, create_directory: bool = False):
    """
    Move a list of files to a destination directory.

    Args:
        file_list (List[str]): List of file paths to move.
        destination_directory (str): The destination directory.
        create_directory (bool, optional): Create the destination directory if it doesn't exist. Defaults to False.
    """
    if create_directory and not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    for file_path in file_list:
        filename = os.path.basename(file_path)
        destination_path = os.path.join(destination_directory, filename)
        shutil.move(file_path, destination_path)


def delete_files_not_in_list(file_list: List[str]):
    """
    Delete all files that are not in the list.

    Args:
        file_list (List[str]): List of file paths to keep.
    """
    for filename in os.listdir(os.getcwd()):
        file_path = os.path.join(os.getcwd(), filename)
        if file_path not in file_list:
            #and if its not a directory
            if not os.path.isdir(file_path):
                os.remove(file_path)



def move_files_to_directories():
    """
    Move files to directories based on their names.

    This function looks at the current directory and finds files with names in the format "directory_name.log".
    It then moves each file to a subdirectory with the same name as the directory part of the file name.

    If a file with the same name already exists in the destination directory, the function appends a numeric
    suffix to the file name to avoid overwriting.

    Returns:
        None
    """
    # List of directories to create or use existing ones
    list_of_dirs = [name.split('.')[0] for name in os.listdir()]

    for dir_name in list_of_dirs:
        file_name = dir_name + '.log'
        destination_dir = dir_name

        # Check if the source file exists before moving
        if os.path.exists(file_name):
            # Check if the destination file already exists
            if os.path.exists(os.path.join(destination_dir, file_name)):
                # If it exists, rename the file being moved with a numeric suffix
                i = 1
                while True:
                    new_file_name = f"{dir_name}_{i}.log"
                    if not os.path.exists(os.path.join(destination_dir, new_file_name)):
                        file_name = new_file_name
                        break
                    i += 1

            # Move the file to the destination directory with the final file name
            shutil.move(file_name, os.path.join(destination_dir, file_name))

    
# def convert_to_list_or_nested_list(input_str):
#     split_by_space = input_str.split(' ')
    
#     # If there are no spaces, return a flat list
#     if len(split_by_space) == 1:
#         return list(map(int, filter(lambda x: x.strip(), split_by_space[0].split(','))))
    
#     # Otherwise, return a nested list
#     nested_list = []
#     for sublist_str in split_by_space:
#         sublist = list(map(int, sublist_str.split(',')))
#         nested_list.append(sublist)
#     return nested_list


def convert_to_custom_nested_list(input_str):
    """
    Converts a comma-separated string into a nested list based on the format:
    - "1,2,3,4,5,6" -> [[1,2,3,4], 5, 6]
    - "1,2,3,4,5,6 2,3,1,5,6,7" -> [[[1,2,3,4], 5, 6], [[2,3,1,5], 6, 7]]
    """
    print(f"Input string: {input_str}")
    split_by_space = input_str.split(' ')  # Split by space for multiple sections

    def process_sublist(sublist_str):
        """Convert a single comma-separated list to the required nested structure."""
        try:
            elements = list(map(int, sublist_str.split(',')))  # Convert to list of integers
        except:
            elements = list(map(int, sublist_str))
        if len(elements) > 2:  # Ensure we separate all but the last two elements
            return [elements[:-2]] + elements[-2:]
        return elements  # If fewer than 3 elements, return as-is

    # Process each segment and decide if it's a nested list or a single flat list
    if len(split_by_space) == 1:
        # Single segment, no spaces
        return process_sublist(split_by_space[0])
    else:
        # Multiple segments separated by spaces
        nested_list = []
        for sublist_str in split_by_space:
            nested_list.append(process_sublist(sublist_str))
        return nested_list
    

def convert_to_list_or_nested_list(input_str):
    # Remove trailing spaces
    input_str = input_str.rstrip()

    # Use regular expression to split by space, dash, or underscore
    split_by_delimiter = re.split(' |-|_', input_str)
    
    # Filter out empty strings
    split_by_delimiter = list(filter(None, split_by_delimiter))

    # If there's only one element, return a flat list
    if len(split_by_delimiter) == 1:
        return list(map(int, filter(None, re.split(',', split_by_delimiter[0]))))
    
    # Otherwise, return a nested list
    nested_list = []
    for sublist_str in split_by_delimiter:
        sublist = list(map(int, filter(None, re.split(',', sublist_str))))
        nested_list.append(sublist)
    return nested_list


def dict_to_horizontal_df(data_dict):
    # Initialize an empty DataFrame to store the transformed data
    df_transformed = pd.DataFrame()
    # Loop through each key-value pair in the original dictionary
    for mol, df in data_dict.items():
        transformed_data = {}
        # Loop through each row and column in the DataFrame
        for index, row in df.iterrows():
            
            index_words = set(index.split('_'))
            for col in df.columns:
                # Create a new key using the format: col_index
                try:
                    col_words = set(col.split('_'))
                except:
                    col_words = []
                 # Check if the index and the column have the same words and remove one
                common_words = index_words.intersection(col_words)
                if col != 0 and '0':
                    if common_words:
                        unique_col_words = col_words - common_words
                        unique_index_words = index_words - common_words
                        new_key_parts = ['_'.join(common_words)] if common_words else []
                        new_key_parts.extend([part for part in ['_'.join(unique_col_words), '_'.join(unique_index_words)] if part])
                        new_key = '_'.join(new_key_parts)
                    else:
                        new_key = f"{col}_{index}"
                else:
                    new_key = f"{index}"
                # Store the corresponding value in the transformed_data dictionary
                transformed_data[new_key] = row[col]
        # Convert the dictionary into a DataFrame row with the molecule name as the index
        df_row = pd.DataFrame([transformed_data], index=[mol])
        # Append the row to df_transformed
        df_transformed = pd.concat([df_transformed, df_row], ignore_index=False)
    return df_transformed


