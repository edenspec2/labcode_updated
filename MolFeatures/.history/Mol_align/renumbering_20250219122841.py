import pandas as pd

from RDKIT_utils import *
from MCS_optimization import *

from itertools import combinations
import math
import os

from datetime import datetime

def add_unumbered(MCSandCoords_map, mol2):

    mol2_atoms = mol2.GetAtoms()

    unnumbered = {}
    for idx2 in range (len(mol2_atoms)):
        if idx2 not in MCSandCoords_map:
            for idx in range (len(mol2_atoms)):
                if idx not in MCSandCoords_map.values() and idx not in unnumbered.values():
                    unnumbered[idx2] = idx
                    break

    return unnumbered

def optimize_based_on_current_mapping(df_1, df_2, temp_map, mol2, init_vars = None):

    optimize_atoms = list(temp_map.values())
    temp_map.update(add_unumbered(temp_map, mol2))

    final_temp_map = {temp_map[i]:i for i in range(len(temp_map))}
    final_temp_order = [final_temp_map[i] for i in range(len(final_temp_map))] 
    final_reversed_map = {i:temp_map[i] for i in range(len(temp_map))}
    final_reversed_order = [final_reversed_map[i] for i in range(len(final_reversed_map))] 

    df_2.iloc[2:, :] = df_2.iloc[2:, :].iloc[final_temp_order, :]
    df_2 = optimize_numbered (df_1, df_2, optimize_atoms, init_vars = None)  
    df_2.iloc[2:, :] = df_2.iloc[2:, :].iloc[final_reversed_order, :]

    return df_2, [torch.zeros(3)]*3 + list(init_vars)[3:]


def spatial_matching(MCSandCoords_map, mol2, mol1, df2, df1):
    
    unnumbered = {}
    mol1_atoms = len(mol1.GetAtoms())
    mol2_atoms = len(mol2.GetAtoms())

    ref_tensor = torch.tensor(df1.iloc[2:,1:].values.astype('float64')).to(torch.float64)
    query_tensor = torch.tensor(df2.iloc[2:,1:].values.astype('float64')).to(torch.float64)
    
    for threshold in torch.arange(0.3, 5.0, 0.005):

        target_unmactched = [i for i in range (mol1_atoms) if i not in MCSandCoords_map.values() and i not in unnumbered.values()]
        query_unmactched = [i for i in range (mol2_atoms) if i not in MCSandCoords_map and i not in unnumbered]
        
        if target_unmactched == []:
            break

        for i in query_unmactched:
            distances = torch.zeros(len(target_unmactched))
            for idx, j in enumerate(target_unmactched):
                distances[idx] = torch.pow(torch.sum(torch.pow (query_tensor[i] - ref_tensor[j], 2)),0.5)
            if torch.min(distances) < threshold and target_unmactched[torch.argmin(distances)] not in unnumbered.values():
                unnumbered[i] = target_unmactched[torch.argmin(distances)]
        
   
    return (unnumbered)

def gen_output_folder(files_list):

    run_name = f"{files_list[0].split('/')[-2]}_renumbered-{datetime.now().strftime('%D-%H-%M').replace('/', '-')}"
    os.mkdir(run_name)

    return run_name

def gen_out_path (path, comment = "", output_folder=None ):

    return  output_folder + "/" + path.split("/")[-1][:-4] + "_" + comment + ".xyz"

def gen_out_path1 (path, comment = ""):

    return   "/" + path.split("/")[-1][10:-4] + "_" + comment + ".xyz"


def batch_renumbering(folder_path, basic_path = "", anchors_file = None):
    
    files_list = []
    min_size = 0
    min_file = ""

    string_report= ""
    if folder_path[-1] != '/':
        folder_path += '/'

    if anchors_file != None:
        anchors_df = pd.read_excel(folder_path + "/" + anchors_file)
        anchors_dic = {folder_path+anchors_df['Mol'][i]+".xyz":anchors_df['idx'][i]-1 for i in range(anchors_df.shape[0])}

    if basic_path != "":

        if not os.path.isfile(basic_path):
            raise Exception("basic file is not exist")

        try:
            df = xyz_to_df(basic_path)
        except:
            raise Exception("basic file is not valid")

        mol = df_to_mol(df)

        if mol == False:
            raise Exception("basic file charge is not valid")

        basic_size = len(mol.GetAtoms())

    invalid_files = []

    for file_name in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file_name)
        try:
            df = xyz_to_df(file_path)
            print(file_name,df)
        except:
            print (file_name + " file is not valid")
            string_report += file_name + " file is not valid\n"
            invalid_files.append(file_name)
            continue

        mol = df_to_mol(df)

        if mol == False:
            raise Exception("basic file charge is not valid")
        
        if mol == False:
            print (file_name + " charge is not valid")
            string_report += file_name + " charge is not valid\n"
            invalid_files.append(file_name)
            continue
        
        if basic_path == "":
            if min_size == 0 or len(mol.GetAtoms()) < min_size:
                min_path = file_path
                min_size = len(mol.GetAtoms())
        
        else:
            if len(mol.GetAtoms()) < basic_size:
                raise Exception(file_name + " size is smaller than the basic file")

        files_list.append (file_path)
    
    if files_list == []:
        raise Exception("no valid files in the folder")
    
    if basic_path == "":
        basic_path = min_path

    output_folder = gen_output_folder(files_list)

    xyz_to_df(basic_path).to_csv(gen_out_path (basic_path, "target", output_folder), header=None, index=None, sep=' ', mode='w+')

    print (f"The basic structure path is {basic_path}")
    string_report += f"The basic structure path is {basic_path}\n"
    final_order_dict= {}
    error_files = []
    for idx, file_path in enumerate(files_list):
        print (f"optimizing {file_path}...({idx+1}/{len(files_list)})")
        string_report += f"optimizing {file_path}...({idx+1}/{len(files_list)})\n"
        if file_path == basic_path:
            print ("skips basic file")
            string_report += "skips basic file\n"
            continue
        else: 
            if anchors_file != None:
                anchors =[anchors_dic[basic_path], anchors_dic[file_path]]
                
                Error, final_order = renumbering (basic_path, file_path, output_folder, anchors = anchors, batch = True)
            else:
                Error, final_order = renumbering (basic_path, file_path, output_folder, batch = True)

            if type(Error) != str:
                error_files.append(file_path)
                print(f"{file_path} failed to optimize")
                continue
            
        final_order_dict[idx] = final_order
    string_report+=Error
    print ("Optimization Ended!")
    print (f"{len(invalid_files)} files were invalid:")
    print (invalid_files)
    print (f"{len(error_files)} failed to optimize:")
    print (error_files)
    string_report += f"Optimization Ended!\n {len(invalid_files)} files were invalid:\n {invalid_files}\n {len(error_files)} failed to optimize:\n {error_files}\n"
    return string_report, final_order_dict

def choose_init_map(init_vars, df_1, df_2, mol2):

    if len (init_vars[3]) == 1:
        return copy.deepcopy(init_vars[3][0])
    
    else:
        matches = torch.zeros(len(init_vars[3]))
        for idx, m in enumerate(init_vars[3]):
            temp_map = copy.deepcopy(m)
            df_2_temp = copy.deepcopy(df_2)
            df_2_temp, _ = optimize_based_on_current_mapping(df_1, df_2_temp, temp_map, mol2, init_vars)
            coords_mapping = Coords_Mapping(df_1, df_2_temp, mol2)
            matches[idx] = len (coords_mapping)

        return copy.deepcopy(init_vars[3][torch.argmax(matches)])

def complete_rings(corrds_mapping, temp_map, init_vars, mol1):
    
    update_map = {}
    for ring in init_vars[4]:
        alignment = 0
        for i in ring:
            if i >= len(mol1.GetAtoms()):
                alignment = 0
                break
            if i in corrds_mapping:
                if temp_map[i] == corrds_mapping[i]:
                    alignment += 1
        if alignment >= 3:
            update_map.update({i:temp_map[i] for i in ring})

            reversed_corrds_mapping = {corrds_mapping[i]:i for i in corrds_mapping}
            for i in update_map:
                if update_map[i] in reversed_corrds_mapping:
                    if i != reversed_corrds_mapping[update_map[i]]:
                        return {}

    return update_map
        

def renumbering(file1_path, file2_path, output_directory = "molecules/results", anchors = None, shuffle = False, batch = False):
    string_report = ""
    df_1 = xyz_to_df(file1_path)
    df_2 = xyz_to_df(file2_path)

    if df_1.shape[0] > df_2.shape[0]:
        df_1, df_2 = df_2, df_1
        file1_path, file2_path = file2_path, file1_path
        
    if shuffle:
        df_1 = shuffle_df(df_1)
    

    mol1 = df_to_mol(df_1)
    mol2 = df_to_mol(df_2)

    init_vars = find_init_vars(mol1, mol2, df_1, df_2, anchors = anchors)
    temp_map = choose_init_map(init_vars, df_1, df_2, mol2)

    # try:
    cycle_num = 1
    cycle = 0
    mcs_cycles = 0
    while cycle != cycle_num:  
        df_2, init_vars = optimize_based_on_current_mapping(df_1, df_2, temp_map, mol2, init_vars)
        coords_mapping = Coords_Mapping(df_1, df_2, mol2)
        coords_mapping.update(complete_rings(coords_mapping, temp_map, init_vars, mol1))
        print (coords_mapping)
        string_report += str(coords_mapping) + "\n"
        if coords_mapping == {} and mcs_cycles < 2:
            init_vars = find_init_vars(mol1, mol2, df_1, df_2, Frag_mcs = mcs_cycles, anchors = anchors)
            temp_map = choose_init_map(init_vars, df_1, df_2, mol2)
            mcs_cycles += 1
            continue

        mcs_cycles = 0
        temp_map = apply_MCS(coords_mapping, mol1, mol2, df_1, df_2)
   
        while type(temp_map) is not dict:
            temp_map = apply_MCS(coords_mapping, mol1, mol2, df_1, df_2)

        cycle += 1
    
    MCSandCoords_map = copy.deepcopy(temp_map)
    MCSandCoords_map.update(spatial_corrections(coords_mapping, MCSandCoords_map, mol2, df_1, df_2))
    MCSandCoords_map.update(spatial_matching(MCSandCoords_map, mol2, mol1, df_2, df_1))
    MCSandCoords_map.update(add_unumbered(MCSandCoords_map, mol2))

    print (MCSandCoords_map)
    string_report += str(MCSandCoords_map) + "\n"
    final_map = {MCSandCoords_map[i]:i for i in range(len(MCSandCoords_map))}
    final_order = [final_map[i] for i in range(len(final_map))]
    print("Shape of df_2.iloc[2:, :]:", df_2.iloc[2:, :].shape)
    print("Length of final_order:", len(final_order))
    # Check if the lengths are equal
    if len(final_order) != df_2.iloc[2:, :].shape[0]:
        print("Mismatch in lengths detected.", file1_path)
        return None, None
    else:
        df_2.iloc[2:, :] = df_2.iloc[2:, :].iloc[final_order, :]
        print("Lengths match.")


    if batch == False:

        if shuffle:
            df_1.to_csv(gen_out_path (file1_path, "shuffled_target", output_directory), header=None, index=None, sep=' ', mode='w+')

        else:
            df_1.to_csv(gen_out_path (file1_path, "target", output_directory), header=None, index=None, sep=' ', mode='w+')

    df_2.to_csv(gen_out_path (file2_path, "optimized", output_directory), header=None, index=None, sep=' ', mode='w+')  

    return string_report, final_order
    
    # except:
    #     print (f"{file2_path} wasn't optimized")
    #     return file2_path
    

def mapping_by_bond():
             
    print("Enter the bonds (use - to separate the atoms): ")
    bond_1 = input('Bond from molecule #1: ')
    bond_2 = input('Bond from molecule #2: ')
    atoms1 = bond_1.split("-")
    atoms2 = bond_2.split("-")
    mapping = {int(atoms1[0]) -1 : int(atoms2[0]) - 1, int(atoms1[1]) - 1 : int(atoms2[1]) - 1}
    return mapping

def check_mul_ring(map, ring1):
    for ring in ring1:
        if check_ring(map, ring):
            return True
    return False

def check_bond(map, ring1, ring2):
    if check_mul_ring(list(map.keys()), ring1) and check_mul_ring(list(map.values()), ring2):
        flag = 0
        keys = list(map.keys())
        vals = list(map.values())
        for itm in ring1:
            if (keys[0] in itm and keys[1] in itm) or (vals[0] in itm and vals[1] in itm) :
                flag +=1
                for itm in ring2:
                    if (keys[0] in itm and keys[1] in itm) or (vals[0] in itm and vals[1] in itm) :
                        flag +=1
        if flag == 2:
            return True
        elif flag >= 2:
            # molecule #1 will be the one with more atoms
            atom1 = input('Please enter a different atom on the ring from molecule #1: ')
            atom2 = input('Please enter a different atom on the ring from molecule #2: ')
            if atom1.isnumeric() and atom2.isnumeric():
                return [int(atom1) - 1, int(atom2) - 1]
            else:
                return False
        else:
            return False


def renumbering_ring(file1_path, file2_path, output_directory = "molecules/results", shuffle = False, batch = False, temp_map = {0:0, 1:1}, by_bond = False):

    df_1 = xyz_to_df(file1_path)
    df_2 = xyz_to_df(file2_path)
    
    flag = False

    if df_1.shape[0] > df_2.shape[0]:
        flag = True
        df_1, df_2 = df_2, df_1
        file1_path, file2_path = file2_path, file1_path
        keys = list(temp_map.keys())
        vals = list(temp_map.values())
        temp_map = {vals[0]:keys[0], vals[1]:keys[1]}
        

    if shuffle:
        df_1 = shuffle_df(df_1)

    mol1 = df_to_mol(df_1)
    mol2 = df_to_mol(df_2)

    ring1 = [[i for i in ring] for ring in Chem.GetSSSR(mol1)]
    ring2 = [[i for i in ring] for ring in Chem.GetSSSR(mol2)]
    
    # if not by_bond:
    #     temp_map = {}
    # else:
    #     temp_map = mapping_by_bond()

    checkbond = check_bond(temp_map, ring1, ring2)

    if not checkbond:
        print('Error, please choose a different bond')
        return 0

    if flag:
        keys = list(temp_map.keys())
        vals = list(temp_map.values())
        temp_map = {vals[0]:keys[0], vals[1]:keys[1]}
   
    init_vars = find_init_vars_ring(df_1, df_2, temp_map,checkbond)
    # try:
    tmap = copy.deepcopy(temp_map)
    df_2, init_vars = optimize_based_on_current_mapping(df_1, df_2, temp_map, mol2, init_vars)
    temp_map = ring_mapping(tmap, mol1, mol2, checkbond)
    map = copy.deepcopy(temp_map)
    print(map)
    map.update(spatial_corrections(temp_map, map, mol2, df_1, df_2))
    map.update(spatial_matching(map, mol2, mol1, df_2, df_1))
    map.update(add_unumbered(map, mol2))
    print (map)

    final_map = {map[i]:i for i in range(len(map))}
    final_order = [final_map[i] for i in range(len(final_map))]
    for i in range(len(final_order)):
        if i not in final_order:
            final_order.append(i)
    df_t = df_2.iloc[:2, :]
    df_2 = df_2.iloc[2:, :].iloc[final_order, :]
    dfs = [df_t, df_2]
    df_2 = pd.concat(dfs)

    if batch == False:

        if shuffle:
            df_1.to_csv(gen_out_path1 (file1_path, "shuffled_target", output_directory), header=None, index=None, sep=' ', mode='w+')

        else:
            df_1.to_csv(gen_out_path1 (file1_path, "target", output_directory), header=None, index=None, sep=' ', mode='w+')
    
    df_2.to_csv(gen_out_path1 (file2_path, "optimized", output_directory), header=None, index=None, sep=' ', mode='w+')  

    return



if __name__ == "__main__": 

    # file1_path = 'molecules/Pyrollydine/b20_opt_ligand.xyz'
    # file2_path = 'molecules/Pyrollydine/b13_opt_ligand.xyz'
    # renumbering(file1_path, file2_path, 1, 1)

    # file1_path = 'molecules/Medium/basic.xyz'
    # file2_path = 'molecules/Medium/d_one.xyz'
    # renumbering(file1_path, file2_path, 0, 0)

        
    # file1_path = 'molecules/conformers/1.txt'
    # file2_path = 'molecules/conformers/3.txt'
    # renumbering(file1_path, file2_path, 0, 0)

    # file1_path = 'molecules/conformers/xbefore.xyz'
    # file2_path = 'molecules/conformers/ybefore.xyz'
    # renumbering(file1_path, file2_path)

    # file1_path = 'molecules/nature_full/ML112_target.xyz'
    # file2_path = 'molecules/nature_full/ML3_optimized.xyz'
    # anchors = [5, 0]
    # renumbering(file1_path, file2_path, anchors = anchors)


    # folder_path = "molecules/nature"
    #batch_renumbering(folder_path)
    file1_path = r'molecules\5_0.xyz'
    file2_path = r'molecules\21_0.xyz'

    renumbering_ring(file1_path, file2_path)