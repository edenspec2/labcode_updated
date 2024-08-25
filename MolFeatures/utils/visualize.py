import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import sys
import pandas as pd
from typing import *
from enum import Enum
import os
import numpy as np

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils  import help_functions as hf
# Now you can import from the parent directory
from Mol_align import renumbering


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

def plot_interactions(xyz_df,color):
    """Creates a 3D plot of the molecule"""

    atomic_radii = GeneralConstants.COVALENT_RADII.value
    cpk_colors = dict(C='black', F='green', H='white', N='blue', O='red', P='orange', S='yellow', Cl='green', Br='brown', I='purple', Ni='blue', Fe='red', Cu='orange', Zn='yellow', Ag='grey', Au='gold',Si='grey')

    # if molecule_name not in train_df.molecule_name.unique():
    #     print(f'Molecule "{molecule_name}" is not in the training set!')
    #     return
    #

    coordinates = np.array(xyz_df[['x', 'y', 'z']].values,dtype=float)
    x_coordinates = coordinates[:, 0]
    y_coordinates = coordinates[:, 1]
    z_coordinates = coordinates[:, 2]
    atoms = ((xyz_df['atom'].astype(str)).values.tolist())
    atom_ids = xyz_df.index.tolist()
    try:
        radii = [atomic_radii[atom] for atom in atoms]
    except TypeError:
        atoms = flatten_list(atoms)
        radii = [atomic_radii[atom] for atom in atoms]


    def get_bonds():
        """Generates a set of bonds from atomic cartesian coordinates"""
        ids = np.arange(coordinates.shape[0])
        bonds = dict()
        coordinates_compare, radii_compare, ids_compare = coordinates, radii, ids

        for _ in range(len(ids)):
            coordinates_compare = np.roll(coordinates_compare, -1, axis=0)
            radii_compare = np.roll(radii_compare, -1, axis=0)
            ids_compare = np.roll(ids_compare, -1, axis=0)
            distances = np.linalg.norm(coordinates - coordinates_compare, axis=1)
            bond_distances = (radii + radii_compare) * 1.3
            mask = np.logical_and(distances > 0.1, distances < bond_distances)
            distances = distances.round(2)
            new_bonds = {frozenset([i, j]): dist for i, j, dist in zip(ids[mask], ids_compare[mask], distances[mask])}
            bonds.update(new_bonds)
        return bonds

    def atom_trace():
        """Creates an atom trace for the plot"""
        colors = [cpk_colors[atom] for atom in atoms]
        markers = dict(color=colors, line=dict(color='lightgray', width=2), size=5, symbol='circle', opacity=0.8)
        trace = go.Scatter3d(x=x_coordinates, y=y_coordinates, z=z_coordinates, mode='markers', marker=markers,
                             text=atoms, name='')
        return trace

    def bond_trace(color):
        """"Creates a bond trace for the plot"""
        trace = go.Scatter3d(x=[], y=[], z=[], hoverinfo='none', mode='lines',
                             marker=dict(color=color , size=7, opacity=1), line=dict(width=3))
        for i, j in bonds.keys():
            trace['x'] += (x_coordinates[i], x_coordinates[j], None)
            trace['y'] += (y_coordinates[i], y_coordinates[j], None)
            trace['z'] += (z_coordinates[i], z_coordinates[j], None)
        return trace

    bonds = get_bonds()
    print(bonds)
    zipped = zip(atom_ids, x_coordinates, y_coordinates, z_coordinates)
    annotations_id = [dict(text=num+1, x=x, y=y, z=z, showarrow=False, yshift=15, font=dict(color="blue"))
                      for num, x, y, z in zipped]

    annotations_length = []
    for (i, j), dist in bonds.items():
        x_middle, y_middle, z_middle = (coordinates[i] + coordinates[j]) / 2
        annotation = dict(text=dist, x=x_middle, y=y_middle, z=z_middle, showarrow=False, yshift=10)
        annotations_length.append(annotation)

    updatemenus = list([
        dict(buttons=list([
                 dict(label = 'Atom indices',
                      method = 'relayout',
                      args = [{'scene.annotations': annotations_id}]),
                 dict(label = 'Bond lengths',
                      method = 'relayout',
                      args = [{'scene.annotations': annotations_length}]),
                 dict(label = 'Atom indices & Bond lengths',
                      method = 'relayout',
                      args = [{'scene.annotations': annotations_id + annotations_length}]),
                 dict(label = 'Hide all',
                      method = 'relayout',
                      args = [{'scene.annotations': []}])
                 ]),
                 direction='down',
                 xanchor = 'left',
                 yanchor = 'top'
            ),
    ])
    data = [atom_trace(), bond_trace(color)]
    return data, annotations_id, updatemenus

def choose_conformers_input():
    string=input('Enter the conformers numbers: ')
    conformer_numbers=string.split(' ')
    conformer_numbers=[int(i) for i in conformer_numbers]
    return conformer_numbers


def unite_buttons(buttons_list, ref_index=0):
    buttons_keys=buttons_list[0].keys()
    united_button=dict.fromkeys(buttons_keys)
    for key in buttons_keys:
        if key=='args':
            all_annotations=[buttons[key][0]['scene.annotations'] for buttons in buttons_list]
            united_annotations=list(zip(*all_annotations))
            united_button[key]=[{'scene.annotations': united_annotations}]
        else:
            united_button[key]=buttons_list[ref_index][key]
    return united_button

def unite_updatemenus(updatemenus_list, ref_index=0):
    menus_keys=updatemenus_list[ref_index][0].keys()
    united_updatemenus_list=dict.fromkeys(menus_keys)
    for key in menus_keys:
        if key=='buttons':
            buttons_list=[updatemenus[0].get(key) for updatemenus in updatemenus_list]
            buttons_num=len(buttons_list[0])   
            segregated_buttons=[]
            for i in range(buttons_num):
                type_buttons=[buttons[i] for buttons in buttons_list]
                segregated_buttons.append(type_buttons)
            buttons=[unite_buttons(buttons) for buttons in segregated_buttons]
            united_updatemenus_list[key]=buttons
        else:
            united_updatemenus_list[key]=updatemenus_list[ref_index][0][key]
    return [united_updatemenus_list]


def compare_molecules(coordinates_df_list: List[pd.DataFrame],conformer_numbers:List[int]=None):
    if conformer_numbers is None:
        conformer_numbers=choose_conformers_input()
    # Create a subplot with 3D scatter plot
    colors_list=['red','purple','blue','green','yellow','orange','brown','black','pink','cyan','magenta']
    new_coodinates_df_list=[coordinates_df_list[i] for i in conformer_numbers]
    new_coodinates_df_list=renumbering_df_list(new_coodinates_df_list)
    # coordinates_df_list=renumber_xyz_by_mcs(coordinates_df_list)  ##needs fixing , renumbering not working.
    xyz_df=(coordinates_df_list[conformer_numbers[0]])
    data_main, annotations_id_main, updatemenus = plot_interactions(xyz_df,'grey')
    updatemenus_list=[updatemenus]
    # Iterate through the conformer numbers and create traces for each conformer
    for  conformer_number,color in zip((conformer_numbers[1:]),colors_list):
        xyz_df = coordinates_df_list[conformer_number]
        data, annotations_id, updatemenus_main = plot_interactions(xyz_df,color)
        data_main += data
        annotations_id_main += annotations_id
        updatemenus_list.append(updatemenus_main)
    # Set axis parameters
    updatemenus=unite_updatemenus(updatemenus_list)
    axis_params = dict(showgrid=False, showbackground=False, showticklabels=False, zeroline=False,
                       titlefont=dict(color='white'))
    # Set layout
    layout = dict(scene=dict(xaxis=axis_params, yaxis=axis_params, zaxis=axis_params, annotations=annotations_id_main),
                  margin=dict(r=0, l=0, b=0, t=0), showlegend=False, updatemenus=updatemenus)
    fig = go.Figure(data=data_main, layout=layout)
    fig.show()
    return fig

import dash
from dash import html, dcc, Output, Input, State
import plotly.graph_objects as go
import pandas as pd

def show_single_molecule(molecule_name,xyz_df=None, color='black'):
    if xyz_df is None:
        xyz_df=get_df_from_file(choose_filename()[0])

    data_main, annotations_id_main, updatemenus = plot_interactions(xyz_df,color)
    # Set axis parameters
    axis_params = dict(showgrid=False, showbackground=False, showticklabels=False, zeroline=False,
                       titlefont=dict(color='white'))
    # Set layout
    layout = dict(title=dict(text=molecule_name, x=0.5, y=0.9, xanchor='center', yanchor='top'),scene=dict(xaxis=axis_params, yaxis=axis_params, zaxis=axis_params, annotations=annotations_id_main),
                  margin=dict(r=0, l=0, b=0, t=0), showlegend=False, updatemenus=updatemenus)
    fig = go.Figure(data=data_main, layout=layout)
    html=fig.show()
    run_app(fig)

    
    return html


def run_app(figure):
        # Create a Dash app
    app = dash.Dash(__name__)

    # App layout
    app.layout = html.Div([
        dcc.Graph(id='molecule-plot', figure=figure), # Replace "Water" with your molecule
        html.Div(id='clicked-data', children=[]),
        html.Button('Save Clicked Atom', id='save-button', n_clicks=0),
        html.Div(id='saved-atoms', children=[])
    ])

    # Callback to display clicked data
    @app.callback(
        Output('clicked-data', 'children'),
        Input('molecule-plot', 'clickData'),
        prevent_initial_call=True
    )
    def display_click_data(clickData):
        if clickData:
            return f"Clicked Point: {clickData['points'][0]['pointIndex']}"
        return "Click on an atom."

    # Callback to save clicked atom index
    @app.callback(
        Output('saved-atoms', 'children'),
        Input('save-button', 'n_clicks'),
        State('molecule-plot', 'clickData'),
        State('saved-atoms', 'children'),
        prevent_initial_call=True
    )
    def save_clicked_atom(n_clicks, clickData, saved_atoms):
        if clickData:
            saved_atoms.append(clickData['points'][0]['pointIndex'])
        return f"Saved Atom Indices: {saved_atoms}"
        



if __name__ == '__main__':
    
    pass
