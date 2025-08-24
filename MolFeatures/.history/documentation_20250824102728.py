The Molecules class manages collections of Molecule objects for batch analysis of molecular properties.
This class provides functionality to load multiple molecular structure files, perform
batch calculations of various molecular features (sterimol parameters, NPA charges, 
vibrations, bond angles, etc.), and visualize results.
Attributes:
    molecules_path (str): Absolute path to the directory containing molecule files
    molecules (List[Molecule]): List of loaded Molecule objects
    failed_molecules (List[str]): List of filenames that could not be loaded
    success_molecules (List[str]): List of successfully loaded filenames
    molecules_names (List[str]): List of names of loaded molecules
    old_molecules (List[Molecule]): Backup of original molecule list
    old_molecules_names (List[str]): Backup of original molecule names
Methods:
    export_all_xyz(): Exports all molecules to XYZ files in a 'xyz_files' subdirectory
    filter_molecules(indices): Keeps only the molecules at specified indices
    get_sterimol_dict(atom_indices, radii, ...): Calculates Sterimol parameters for all molecules
    get_npa_dict(atom_indices, sub_atoms): Calculates NPA charges for all molecules
    get_ring_vibration_dict(ring_atom_indices, threshold): Gets ring vibration data for all molecules
    get_dipole_dict(atom_indices, origin, visualize_bool): Calculates dipole moments for all molecules
    get_bond_angle_dict(atom_indices): Calculates bond angles for all molecules
    get_bond_length_dict(atom_pairs): Calculates bond lengths for all molecules
    get_stretch_vibration_dict(atom_pairs, threshold): Calculates stretching vibrations for all molecules
    get_charge_df_dict(atom_indices): Extracts charge data for all molecules
    get_charge_diff_df_dict(atom_indices, type): Calculates charge differences for all molecules
    get_bend_vibration_dict(atom_pairs, threshold): Calculates bending vibrations for all molecules
    visualize_molecules(indices): Visualizes specified molecules (or all if indices=None)
    visualize_smallest_molecule(): Visualizes the molecule with the fewest atoms
    visualize_smallest_molecule_morfeus(indices): Visualizes Sterimol parameters for smallest molecule
    get_molecules_features_set(entry_widgets, parameters, ...): Comprehensive method to extract
                                                               multiple features based on GUI inputs
    extract_all_dfs(): Exports all dataframes to CSV files for each molecule
    extract_all_xyz(): Exports all molecules to XYZ files in a batch