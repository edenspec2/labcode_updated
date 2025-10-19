import pandas as pd
import numpy as np
import torch
import random
import os
from .df_funs import *
from torch.autograd import Variable
from .rotations import *
from .loss import *
from .renumbering import *

# df_1 = read_big_xyz_file(r'C:\Users\hagga\Desktop\Mol_align\molecules\crest_conformers_21.xyz')
# print(df_1)
# for i in range(len(df_1)):
#     df_1[i].to_csv(r'molecules/21_'+str(i)+".xyz", header=None, index=None, sep=' ', mode='a')

# pairs = []


# for root, dirs, files in os.walk(r"C:\Users\hagga\Desktop\Mol_align\molecules\mol_pairs", topdown=False):
#    for name in files:
#       print(os.path.join(root, name))
#    for name in dirs:
#       print(os.path.join(root, name))

# for root, dirs, files in os.walk(r"C:\Users\hagga\Desktop\Mol_align\molecules\mol_pairs", topdown=False):
#    for name in dirs:
#       pairs.append(os.path.join(root, name))

# files_list = []

# for pair in pairs:
#    for root, dirs, files in os.walk(pair, topdown=False):
#       for name in files:
#          files_list.append((pair+"\\"+name)[33::])

# print(len(pairs))
# renumbering(r'molecules\81_0.xyz',r'molecules\81_1.xyz')

# file_path1 = r'molecules\21_0.xyz'
# file_path2 = r'molecules\21_1.xyz'
# df_1 = xyz_to_df(file_path1)
# df_2 = xyz_to_df(file_path2)
# var1, var2 = find_init_vars_ring(df_1, df_2)
# df1 = df_w_symbols(df_1, var1)
# df2 = df_w_symbols(df_2, var2)
# df1.to_csv(r'C:\Users\hagga\Desktop\Mol_align\molecules\centered at COM\ '+file_path1[10:], header=None, index=None, sep=' ', mode='w+')
# df2.to_csv(r'C:\Users\hagga\Desktop\Mol_align\molecules\centered at COM\ '+file_path2[10:], header=None, index=None, sep=' ', mode='w+')
