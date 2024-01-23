try:
    from tkinter import Tk, Frame, Label, Button, Entry, StringVar, OptionMenu, Toplevel, filedialog, Text, Scrollbar, Checkbutton, IntVar, Canvas
    import customtkinter  # Assuming you have this library
    import sys
    import os
    from tkinter.simpledialog import askstring
    import csv
    import pandas as pd
    import re
    import shutil
    import subprocess
    from .utils import help_functions, file_handlers
    from .M2_data_extractor.data_extractor import Molecules
    from .M1_pre_calculations.main import Module1Handler
    from .Mol_align.renumbering import batch_renumbering
    from .M2_data_extractor.feather_extractor import logs_to_feather
    import warnings
    from .M3_modeler.single_model_processing import Model
    from .M3_modeler.model_info_app import ModelInfoTkinter
    from tkinter import filedialog, messagebox
    import warnings
    from typing import Dict
    from .QuantuMol.DGLmol import QuantuMolGraph
    from tkinterweb import HtmlFrame
    
except ImportError or ModuleNotFoundError as e:
    print(f"An error occurred: {e}")
    import os
    import subprocess
    import sys
    # List of packages to install
    packages = [
        "pandas",
        "rdkit",
        "python-igraph",
        "XlsxWriter",
        "dgl",
        "pyarrow",
        "plotly",
        "customtkinter",
        "chardet",
        "torch",
        "matplotlib",
        "rmsd",
        "networkx"
    ]
    def install(package):
            # Replace 'your_python_path' with the path of the Python executable used in CMD
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], capture_output=True, text=True)

        if result.returncode == 0:
            print("Package installed successfully.")
        else:
            print("Error:", result.stderr)

    [install(package) for package in packages]
    print(f'Installed the Following Packages : {packages}\n')

    from tkinter import Tk, Frame, Label, Button, Entry, StringVar, OptionMenu, Toplevel, filedialog, Text, Scrollbar, Checkbutton, IntVar, Canvas
    import customtkinter  # Assuming you have this library
    import sys
    import os
    from tkinter.simpledialog import askstring
    import csv
    import pandas as pd
    import re
    import shutil
    import subprocess
    from .utils import help_functions, file_handlers
    from .M2_data_extractor.data_extractor import Molecules
    from .M1_pre_calculations.main import Module1Handler
    from .Mol_align.renumbering import batch_renumbering
    from .M2_data_extractor.feather_extractor import logs_to_feather
    import warnings
    from .M3_modeler.single_model_processing import Model
    from .M3_modeler.model_info_app import ModelInfoTkinter
    from tkinter import filedialog, messagebox
    import warnings
    from typing import Dict
    from .QuantuMol.DGLmol import QuantuMolGraph
    from tkinterweb import HtmlFrame


# Assuming the 'Model' and 'Model_info' classes are defined elsewhere in your code


    
def convert_to_list_or_nested_list(input_str):
    split_by_space = input_str.split(' ')
    
    # If there are no spaces, return a flat list
    if len(split_by_space) == 1:
        return list(map(int, split_by_space[0].split(',')))
    
    # Otherwise, return a nested list
    nested_list = []
    for sublist_str in split_by_space:
        sublist = list(map(int, sublist_str.split(',')))
        nested_list.append(sublist)
    return nested_list

class MoleculeApp:
    def __init__(self, master):
        
        self.master = master
        master.title("Molecule Data Extractor")
        self.current_file_path = os.path.abspath(__file__)
        # Get the directory of the current file
        self.current_directory = os.path.dirname(self.current_file_path)
        os.chdir(self.current_directory)
        
        self.sidebar_frame_left = customtkinter.CTkFrame(master, width=140, corner_radius=0)
        self.sidebar_frame_left.grid(row=0, column=0, rowspan=4, sticky="nsew")

        self.sidebar_frame_right = customtkinter.CTkFrame(master, width=140, corner_radius=0)
        self.sidebar_frame_right.grid(row=0, column=3, rowspan=4, sticky="nsew")

        self.output_text = Text(master, wrap='word', height=20, width=100)
        self.scrollbar = Scrollbar(master, command=self.output_text.yview)
        self.output_text.config(yscrollcommand=self.scrollbar.set)
        self.output_text.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.scrollbar.grid(row=0, column=2, rowspan=4, sticky='ns')
        self.scrollbar.bind("<MouseWheel>", lambda event: self.output_text.yview_scroll(int(-1 * (event.delta / 120)), "units"))
        self.show_result(f"Current directory: {self.current_directory}\n List of files: {os.listdir()}\n")
        self.print_description()

        #label for parameters
        self.param_description = Label(master, text="")
        self.param_description.grid(row=4, column=1, sticky='w')

        # choose working directory
        self.choose_dir_button = customtkinter.CTkButton(self.sidebar_frame_left, text="Choose Working Directory", command=self.choose_directory)
        self.choose_dir_button.grid(row=2, column=0, padx=20, pady=10)  # Adjust row and column as needed

        #create new directory
        self.create_dir_button = customtkinter.CTkButton(self.sidebar_frame_left, text="Create New Directory", command=self.create_new_directory)
        self.create_dir_button.grid(row=2, column=1, padx=20, pady=10) 

        # Entry for parameters
        self.param_entry = Entry(master,width=50)
        self.param_entry.grid(row=5, column=1, sticky='w')

        # Submit button
        self.submit_button = Button(master, text="Submit", command=self.activate_method)
        self.submit_button.grid(row=5, column=0, sticky='e')

        self.label = customtkinter.CTkLabel(self.sidebar_frame_left, text="Choose Directory to Load Feather files:")
        self.label.grid(row=0, column=0, padx=20, pady=10)

        self.folder_path = StringVar()
        self.browse_button = customtkinter.CTkButton(self.sidebar_frame_left, text="Browse for Feather Files Directory", command=self.browse_directory)
        self.browse_button.grid(row=1, column=0, padx=20, pady=10)
        
        # self.method_var = StringVar(master)
        # self.method_var.set("Choose a method")  # Default value
        # self.method_menu = OptionMenu(self.sidebar_frame_left, self.method_var, "get_sterimol_dict", "get_npa_dict", "get_stretch_dict", "get_ring_dict", "get_dipole_dict", "get_bond_angle_dict", "get_bond_length_dict", "get_nbo_dict", "get_bending_dict")
        # self.method_menu.grid(row=3, column=0, padx=20, pady=10)
        # self.method_menu.bind("<ButtonRelease-1>", self.open_param_window)

        self.method_var = StringVar(master)
        self.method_var.set("Choose a method")  # Default value
        self.method_var.trace_add("write", lambda *args: self.open_param_window())

        self.method_menu = OptionMenu(self.sidebar_frame_left, self.method_var, "Windows Command",
                                    "get_sterimol_dict", "get_npa_dict", "get_stretch_dict", "get_ring_dict",
                                    "get_dipole_dict", "get_bond_angle_dict", "get_bond_length_dict",
                                    "get_nbo_dict", "get_bending_dict")
        self.method_menu.grid(row=3, column=0, padx=20, pady=10)
        

        
        # StringVar for dropdown menu selection
        self.file_handler_var = StringVar(master)
        self.file_handler_var.set("File Handler")  # Default value
        

        # Dropdown menu for file handling options
        self.file_handler_menu = OptionMenu(self.sidebar_frame_left, self.file_handler_var, "Smiles to XYZ", "Create com Files", "Log to Feather")
        self.file_handler_menu.grid(row=3, column=1, padx=20, pady=10)
        self.file_handler_var.trace_add("write", lambda *args: self.handle_file_action())

        
        

        # Callback for dropdown menu
        
    

        # Separate button for Visualization
        self.visualize_button = customtkinter.CTkButton(self.sidebar_frame_left, text="Visualize Molecules", command=self.visualize_molecules)
        self.visualize_button.grid(row=5, column=0, padx=20, pady=10)
        
        self.model_button = customtkinter.CTkButton(self.sidebar_frame_left, text="Model Data", command=self.run_model_in_directory)
        self.model_button.grid(row=5, column=1, padx=20, pady=10)

        # Separate button for Export Data
        self.export_button = customtkinter.CTkButton(self.sidebar_frame_left, text="Extract DataFrames", command=self.export_data)
        self.export_button.grid(row=6, column=0, padx=20, pady=10)
        self.molecules = None  # Placeholder for Molecules object

        self.filter_molecules_button = customtkinter.CTkButton(self.sidebar_frame_left, text="Filter Molecules", command=self.filter_molecules)
        self.filter_molecules_button.grid(row=1, column=1, padx=20, pady=10)

        self.comp_set_button = customtkinter.CTkButton(self.sidebar_frame_left, text="Produce Comp Set", command=self.open_question_window)
        self.comp_set_button.grid(row=4, column=0, padx=20, pady=10)
         

        if self.molecules is not None:
            self.check_vars = [IntVar(value=1) for _ in self.molecules_names]
        else:
            self.check_vars = []
        
        # save text button
        self.save_txt_button = Button(master, text="Save Output Text", command=self.save_text)
        self.save_txt_button.grid(row=5, column=1, sticky='e')

        

        self.renumber_button = customtkinter.CTkButton(self.sidebar_frame_left, text="Renumber xyz Directory", command=self.renumber_directory)
        self.renumber_button.grid(row=4, column=1, padx=20, pady=10)

        self.create_heterograph_button = customtkinter.CTkButton(self.sidebar_frame_right, text="Create Heterograph", command=self.create_heterograph)
        self.create_heterograph_button.grid(row=0, column=0, padx=20, pady=10)

    def print_description(self):
        # Path to README.md file from the MolFeatures directory
        txt_path =  'description.txt'
        try:
            with open(txt_path, 'r') as txt_file:
                string=txt_file.read()
                self.show_result(string)
        except FileNotFoundError:
            self.show_result("Description file not found.")

    def create_heterograph(self):
        feather_filename=filedialog.askopenfilename(defaultextension=".feather",
                                       filetypes=[("Feather files", "*.feather"),
                                                  ("All files", "*.*")])
        if feather_filename:
            quantumgraph=QuantuMolGraph(feather_filename)
            self.show_result(f"QuantuMolGraph created from {feather_filename}\n")
            self.show_result(f"QuantuMolGraph nodes: {quantumgraph.graph_report_string}")

    def run_model_in_directory(self, min_features_num=2, max_features_num=4, target_csv_filepath= '' ) -> None:
        """
        Runs a model in a specified directory using provided CSV filepaths.

        :param directory: The directory to change to.
        :param csv_filepaths: A dictionary with filepaths for features and target CSV files.
        :param min_features_num: Minimum number of features for the model.
        :param max_features_num: Maximum number of features for the model.
        """
        directory=filedialog.askdirectory()
        output_csv_filepath=filedialog.askopenfilename(defaultextension=".csv",
                                       filetypes=[("Excel files", "*.csv"),
                                                  ("All files", "*.*")])
        os.chdir(directory)
        csv_filepaths = {'features_csv_filepath': output_csv_filepath,
                        'target_csv_filepath': target_csv_filepath}
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:    
                self.model = Model(csv_filepaths, min_features_num=min_features_num, max_features_num=max_features_num)  # leave_out=['o_Cl']
            except:
                messagebox.showinfo('Error', 'Failed to initialize model. Check that the CSV includes output column.')
            self.model_info = ModelInfoTkinter(self.master, self.model)
            self.model_info.present_model()

            return 

    def handle_file_action(self, *args):
        selected_action = self.file_handler_var.get()
        if selected_action == "Smiles to XYZ":
            self.smiles_to_xyz_files()
        elif selected_action == "Create com Files":
            self.open_com_window()
        elif selected_action == "Log to Feather":
            self.log_to_feather()

    def log_to_feather(self):
        directory = filedialog.askdirectory()
        string_report=logs_to_feather(directory)
        self.show_result(f"Log to Feather Report: {string_report}")

    def renumber_directory(self):
        # Ask user if they want to create a new directory
        create_new_dir = messagebox.askyesno("Choose Directory", "Do you want to create a new directory for XYZ files?")
        
        if create_new_dir:
            # Let the user choose a location and name for the new directory
            new_dir_path = filedialog.asksaveasfilename(title="Select location for new directory",
                                                        filetypes=[('All Files', '*.*')])
            if new_dir_path:
                os.makedirs(new_dir_path, exist_ok=True)
                directory = new_dir_path
                os.chdir(directory)
                try:
                    [mol.write_xyz_file() for mol in self.molecules.molecules]
                except AttributeError:
                    self.show_result(f"Failed to write XYZ files to {directory} ...")
                
            else:
                return  # User cancelled the action
        else:
            # Let the user select an existing directory
            directory = filedialog.askdirectory()
            os.chdir(directory)
            if not directory:
                return  # User cancelled the action

        
        string_report = batch_renumbering(directory)
        self.show_result(f"Renumbering Report: {string_report}")

    
    def smiles_to_xyz_files(self):
        # Initialize a Module1Handler object
        file_path = filedialog.askopenfilename(defaultextension=".csv",
                                       filetypes=[("Excel files", "*.csv"),
                                                  ("All files", "*.*")])

        module_handler = Module1Handler(file_path)
        os.chdir(module_handler.working_dir)
        help_functions.smiles_to_xyz_files(module_handler.smiles_list, module_handler.names_list, new_dir=True)
        
        
    def save_text(self):
        # dir_path = filedialog.askdirectory()
        text_name = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=[("Text files", "*.txt"),
                                                            ("All files", "*.*")])
        dir_path = text_name.replace(text_name.split('/')[-1], '')
        if dir_path:
            os.chdir(dir_path)
            self.show_result(f" text saved at {dir_path}")
            with open(text_name, 'w') as f:
                f.write(self.output_text.get(1.0, "end-1c")) 
                f.close()
                                                   
    
    def open_question_window(self):
        self.parameters={'dipole_mode': 'gaussian', 'radii': 'bondi'}

        def load_answers():
            file_path = filedialog.askopenfilename(defaultextension=".txt",
                                                filetypes=[("Text files", "*.txt"),
                                                            ("All files", "*.*")])
            if file_path:
                with open(file_path, 'r') as f:
                    lines = f.read()
                    # Define a regex pattern for identifying lists and lists of lists in the text.
                    pattern = r'(\[[\d, ]*\])|(\[\[[\d, \[\]]*\]\])'
                    # Find all matches of the pattern in the text.
                    matches = re.findall(pattern, lines)
                    # Extract the non-empty matches and initialize a list to store the final strings.
                    # non_empty_matches = [match[0] or match[1] for match in matches]
                    final_strings = []
                    for match in matches:
                        match_str = match[0] or match[1]
                        if not match_str:
                            final_strings.append(None)  # Add None for empty matches
                        elif match_str.startswith("[["):  # List of lists
                            inner_lists = match_str[1:-1].split("], [")
                            joined_string = " ".join([inner_list.replace(", ", ",") for inner_list in inner_lists])
                            final_string = joined_string.strip('[]')
                            final_strings.append(final_string if final_string else None)  # Add None for empty lists
                        else:  # Single list
                            final_string = match_str[1:-1].replace(", ", ",")
                            final_strings.append(final_string if final_string else None )  # Add None for empty lists
                # Create a dictionary to store the transformed lists
                dict_of_ints = {f"list_{i}": lst.replace('[', '').replace(']', '') for i, lst in enumerate(final_strings) if lst is not None}
                for i in range(8):
                    key = f'list_{i}'
                    if key not in dict_of_ints:
                        dict_of_ints[key] = ''
                f.close()
                print(dict_of_ints)
            return dict_of_ints

        
        # Function to open a new window with the parameters of the given function
        def open_parameter_window():
            window = Toplevel(root)
            window.title("Parameters")
            window.grab_set()
            frame = Frame(window)
            var1 = StringVar(frame)
            var1.set("Dipole")
            var1.trace_add("write", lambda *args: apply_parameters())
            frame.pack(pady=5)
            dipole_mode=OptionMenu(frame,var1, 'Gaussian', 'NBO')
            dipole_mode.grid(row=0, column=0, padx=5)
            
            var2 = StringVar(frame)
            var2.set("Radii")
            var2.trace_add("write", lambda *args: apply_parameters())
            radii_mode=OptionMenu(frame, var2 ,'Bondi', 'CPK','Covalent')
            radii_mode.grid(row=0, column=1, padx=5)
            
            
            def apply_parameters():
                self.parameters['dipole_mode']=var1.get()   
                self.parameters['radii']=var2.get()
                chosen_parameters.config(text=f"Chosen Parameters: {self.parameters}")
                
                return 
            
            # Create an entry widget for the answer
            apply_button = Button(frame, text="Apply", command=new_window.destroy)
            apply_button.grid(row=0, column=2, padx=5)


        def submit_answers(entry_widgets, parameters ,save_as=False, load=False):
            answers = {}
            for question, entry in entry_widgets.items():
                answers[question] = entry.get()
            if load:
                answers=load_answers()
            
            dipole = self.parameters['dipole_mode'] if 'dipole' in self.parameters else 'Gaussian'
            radii = self.parameters['radii'] if 'radii' in self.parameters else 'Bondi'
            comp_set=self.molecules.get_molecules_comp_set_app(answers, dipole_mode=dipole, radii=radii)  # For demonstration purposes; replace this with your desired action

            self.show_result(f"Comp Set: {comp_set}")
            if save_as and not load:
                file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=[("Text files", "*.txt"),
                                                            ("All files", "*.*")])
                if file_path :
                    with open(file_path, 'w') as f:
                        for question, answer in answers.items():
                            f.write(f"{question}\n{answer}\n\n")

            
            if save_as:
                file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("csv files", "*.csv"),
                                                        ("All files", "*.*")])
                if file_path:
                    # Save the DataFrame to a CSV file
                    comp_set.to_csv(file_path, index=True)

        # Create a new window
        new_window = Toplevel(root)
        new_window.title("Questions")
        new_window.grab_set()

        button = Button(new_window, text="Choose Parameters", command=lambda : open_parameter_window())
        button.pack(pady=10)
        chosen_parameters = Label(new_window, text=f"Chosen Parameters: {self.parameters}")
        chosen_parameters.pack(pady=10)
        

        questions = [
            "Ring Vibration atoms - by order -> primary axis (para first), ortho atoms and meta atoms: \n example: 1,2 5,7 3,6",
            "Strech Vibration atoms- enter atom pairs that have a common atom: \n example: 1,2 4,5",
            "Bending Vibration atoms - Atom pairs: \n example: 1,2 4,5",
            "Dipole atoms - indices for coordination transformation: \n example: 4,5,6",
            "NBO Difference - Insert atoms to show NBO, Insert atom pairs to calculate differences: \n example: 1,2,3,4 1,2 3,4",
            "Sterimol atoms - Primary axis along: \n example: 7,8 2,3",
            "Bond lenght - Atom pairs to calculate difference: \n example: 1,2 4,5",
            "Bond Angle - Insert a list of atom triads/quartets for which you wish to have angles/dihedrals:"
        ]

        # Dictionary to store Entry widgets
        entry_widgets = {}

        for question in questions:
            # Create a frame for each question and entry pair
            frame = Frame(new_window)
            frame.pack(pady=5)

            # Create a label for the question
            label = Label(frame, text=question, wraplength=400)
            label.pack(side="left", padx=5)

            # Create an entry widget for the answer
            entry = Entry(frame, width=50)
            entry.pack(side="left", padx=5)
            # choose parameters button
            visualize_button = Button(frame, text="Visualize", command=lambda: self.visualize_smallest_molecule())
            visualize_button.pack(side="left", padx=5)

            # Store the entry widget in the dictionary
            entry_widgets[question] = entry

            # Add Submit button
        submit_button = Button(new_window, text="Submit", command=lambda: submit_answers(entry_widgets, parameters=self.parameters))
        submit_button.pack(pady=20)

        # save as 
        save_as_button = Button(new_window, text="Save input/output", command=lambda: submit_answers(entry_widgets, parameters=self.parameters,save_as=True))
        save_as_button.pack(pady=10)

        load_answers_file = Button(new_window, text="Load input", command=lambda: submit_answers(entry_widgets,parameters=self.parameters, save_as=True, load=True))
        load_answers_file.pack(pady=10)

        

    def open_com_window(self):
        options_window = Toplevel(self.master)
        options_window.title("Conversion Options")
        options_window.grab_set()  # Make the window modal
        gaussian_options = {
            'functionals': ['HF', 'B3LYP', 'PBE', 'M06-2X', 'CAM-B3LYP', 'MP2', 'CCSD'],
            'basis_sets': ['STO-3G', '3-21G', '6-31G', '6-31G(d)', '6-31G(d,p)', '6-31+G(d,p)', '6-311G(d,p)', '6-311+G(d,p)', '6-311++G(d,p)', '6-311++G(2d,p)', '6-311++G(3df,2p)'],
            'tasks': ['sp', 'opt']
        }

        # Create OptionMenus for functional, basis_set, and task
        self.functional_var = StringVar(value='HF')
        Label(options_window, text='Functional:').pack()
        OptionMenu(options_window, self.functional_var, *gaussian_options['functionals']).pack()

        self.basisset_var = StringVar(value='6-31G(d)')
        Label(options_window, text='Basis Set:').pack()
        OptionMenu(options_window, self.basisset_var, *gaussian_options['basis_sets']).pack()

        self.task_var = StringVar(value='sp')
        Label(options_window, text='Task:').pack()
        OptionMenu(options_window, self.task_var, *gaussian_options['tasks']).pack()

        # Parameters to be entered by the user

        self.charge_var = StringVar(value='0 1')
        self.nbo_var = StringVar(value='n')
        self.title_var = StringVar(value='title')
        # Create labels and entry widgets for each parameter
        Label(options_window, text='Spin & Charge:').pack()
        Entry(options_window, textvariable=self.charge_var).pack()
        Label(options_window, text='Title:').pack()
        Entry(options_window, textvariable=self.title_var).pack()

        # Add button to select directory and execute conversion
        customtkinter.CTkButton(options_window, text="Select Directory and Convert", command=self.convert_xyz_to_com).pack()
        customtkinter.CTkButton(options_window, text="Create New Directory", command=self.create_new_directory).pack()

    def convert_xyz_to_com(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            os.chdir(folder_selected)
            # Loop over each .xyz file in the selected directory
            for filename in os.listdir(folder_selected):
                if filename.endswith('.xyz'):
                    # Your xyz_to_gaussian_file function goes here
                    file_handlers.write_gaussian_file(filename,
                                         self.functional_var.get(),
                                         self.basisset_var.get(),
                                         self.charge_var.get(),
                                         self.title_var.get(),
                                         self.task_var.get())

                    # self.show_result(f"Converting {filename} with {self.functional_var.get()} / {self.basisset_var.get()} ...")

                    try: 
                        com_filename = filename.replace('.xyz', '.com')
                        shutil.move(com_filename, self.new_directory_path)
                        self.show_result(f"Moving {com_filename} to {self.new_directory_path} ...")
                    except AttributeError:
                        self.show_result(f"Failed to move {com_filename} to {self.new_directory_path} ...")
                    
            # move all com files to a new directory called com.
            


    def get_answers(self):
        for question, entry in self.answers.items():
            self.show_result(f"{question}: {entry.get()}")

    def filter_molecules(self):
            self.new_window = Toplevel(self.master)
            self.new_window.title("Filter Molecules")

            canvas = Canvas(self.new_window)
            scrollbar = Scrollbar(self.new_window, orient='vertical', command=canvas.yview)
            scrollbar.pack(side='right', fill='y')
            scrollbar.bind("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))
            canvas.pack(side='left', fill='both', expand=True)
            canvas.configure(yscrollcommand=scrollbar.set)

            frame = Frame(canvas)
            canvas_frame = canvas.create_window((0, 0), window=frame, anchor='nw')

            self.check_vars = [IntVar(value=1) for _ in self.molecules.old_molecules_names]
            for index, molecule in enumerate(self.molecules.old_molecules_names):
                Checkbutton(frame, text=molecule, variable=self.check_vars[index]).pack(anchor='w')

            Button(frame, text="Submit", command=self.get_selected_molecules).pack()
            Button(frame, text="Uncheck", command=self.uncheck_all_boxes).pack()
            Button(frame, text="Check", command=self.check_all_boxes).pack()

            frame.update_idletasks()
            canvas.config(scrollregion=canvas.bbox('all'))
            # allow scrooling with scrollwheel
            canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

    def check_all_boxes(self):
        for var in self.check_vars:
            var.set(1)

    def uncheck_all_boxes(self):
        for var in self.check_vars:
            var.set(0)

    def get_selected_molecules(self):
        self.molecules.molecules_names = self.molecules.old_molecules_names
        self.molecules.molecules = self.molecules.old_molecules

        selected_indices = [i for i, var in enumerate(self.check_vars) if var.get() == 1]
        self.show_result(f"Selected indices: {selected_indices}")
        self.new_window.destroy()
        self.molecules.filter_molecules(selected_indices)
        self.show_result(f"Initializing Molecules: {self.molecules.molecules_names}")

    def choose_directory(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            os.chdir(folder_selected)
            self.show_result(f"Working directory changed to {folder_selected}")


    def create_new_directory(self):
        folder_selected = filedialog.askdirectory()
        os.chdir(folder_selected)
        folder_name = filedialog.asksaveasfilename(title="Enter a Name")
        self.new_directory_path = os.path.join(folder_selected, folder_name)
        if folder_name:
            try:
                os.makedirs(folder_name)
                self.show_result(f"Directory {folder_name} created.")
            except FileExistsError:
                self.show_result(f"Directory {folder_name} already exists.")
            except Exception as e:
                self.show_result(f"An error occurred: {e}")
        os.chdir(folder_name)
        self.show_result(f"Working directory changed to {folder_name}")

    def browse_directory(self):
        
        print("Inside browse_directory()...")  # Debugging
        folder_selected = filedialog.askdirectory(initialdir=self.current_directory)
        print(f"folder_selected: {folder_selected}")  # Debugging
        if folder_selected:
            self.folder_path.set(folder_selected)
            self.initialize_molecules()

    def initialize_molecules(self):
        
        directory = self.folder_path.get()
        os.chdir(directory)
        files_list = os.listdir(directory)
        feather_files = [file for file in files_list if file.endswith('.feather')]
        if directory:
            if len(feather_files) == 0:
                dir_list=os.listdir()
                try:
                    os.mkdir('feather_files')
                except FileExistsError:
                    pass
                for dir in dir_list:
                    os.chdir(dir)     
                    feather_file = [file for file in os.listdir() if (file.endswith('.feather') and file.split('-')[0]=='xyz_files')][0]
                    try:
                        shutil.copy(feather_file, directory + '/feather_files')
                    except shutil.SameFileError:    
                        pass
                    os.chdir('..')
                
                self.molecules = Molecules(directory+'/feather_files') # , renumber=True
                self.show_result(f"Molecules initialized with directory: {self.molecules.molecules_names}\n")
                self.show_result(f'Failed to load Molecules: {self.molecules.failed_molecules}\n')
                self.show_result(f"Initializing Molecules with directory: {directory}\n")  # Debugging
                os.chdir('..')
            else:
                print(f"Initializing Molecules with directory: {directory}")  # Debugging
                self.show_result(f"Initializing Molecules with directory: {directory}\n")  
                self.molecules = Molecules(directory) # , renumber=True
                self.show_result(f"Molecules initialized : {self.molecules.molecules_names}\n")
                self.show_result(f'Failed to load Molecules: {self.molecules.failed_molecules}\n')
                self.show_result(f"Initializing Molecules with directory: {directory}\n") 
                
            

    def open_param_window(self):
        
        selected_method = self.method_var.get()
        description_text = f"Enter parameters for {selected_method}:"
        self.param_description.config(text=description_text)

        if selected_method == "Windows Command":
            self.show_result(f"Use as Command Line:\n")
        elif selected_method == "get_sterimol_dict":
            self.show_result(f"Method: {(self.molecules.molecules[0].get_sterimol.__doc__)}\n)")
        elif selected_method == "get_npa_dict":
            self.show_result(f"Method: {(self.molecules.molecules[0].get_npa_df.__doc__)}\n)")
        elif selected_method == "get_stretch_dict":
            self.show_result(f"Method: {self.molecules.molecules[0].get_stretch_vibration.__doc__}\n)")
        elif selected_method == "get_ring_dict":
            self.show_result(f"Method: {self.molecules.molecules[0].get_ring_vibrations.__doc__}\n)")
        elif selected_method == "get_dipole_dict":
            self.show_result(f"Method: {self.molecules.molecules[0].get_dipole_gaussian_df.__doc__}\n)")
        elif selected_method == "get_bond_angle_dict":
            self.show_result(f"Method: {self.molecules.molecules[0].get_bond_angle.__doc__}\n)")
        elif selected_method == "get_bond_length_dict":
            self.show_result(f"Method: {self.molecules.molecules[0].get_bond_length.__doc__}\n)")
        elif selected_method == "get_nbo_dict":
            self.show_result(f"Method: {self.molecules.molecules[0].get_nbo_df.__doc__}\n)")
        elif selected_method == "get_bending_dict":
            self.show_result(f"Method: {self.molecules.molecules[0].get_bend_vibration.__doc__}\n)")
        
    def show_result(self, result):
        # Update Text widget instead of creating a new Toplevel window
        self.output_text.insert('end', str(result) + '\n')
        self.output_text.see('end')  # Auto-scroll to the end

    def activate_method(self):

        method = self.method_var.get()
        params = self.param_entry.get()

        # Now you have the method and parameters, you can activate the method
        if method == "Windows Command":
            self.use_command_line(params)
            
        elif method == "get_sterimol_dict":
            self.get_sterimol(params)
        elif method == "get_npa_dict":
            self.get_npa(params)
        elif method == "get_stretch_dict":
            self.get_stretch(params)
        elif method == "get_ring_dict":
            self.get_ring(params)
        elif method == "get_dipole_dict":
            self.get_dipole(params)
        elif method == "get_bond_angle_dict":
            self.get_bond_angle(params) 
        elif method == "get_bond_length_dict":
            self.get_bond_length(params)
        elif method == "get_nbo_dict":
            self.get_nbo(params)
        elif method == "get_bending_dict":
            self.get_bending(params)
        elif method == "get_molecules_comp_set":
            self.get_molecules_comp_set_app()

    def use_command_line(self, params):
        try:
            # Execute the command and capture the output
            result = subprocess.run(params, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Print the standard output of the command
            self.show_result(f"Output: {result.stdout}\n")
            

            # Optionally, print the standard error if there is any
            if result.stderr:
                self.show_result(f"Errors: {result.stderr}\n")
                
        except subprocess.CalledProcessError as e:
            # This block will run if the command exits with a non-zero status
            self.show_result(f"An error occurred: {e}")

    def get_sterimol(self,base_atoms_str):
        base_atoms = convert_to_list_or_nested_list(base_atoms_str)
        sterimol_data = self.molecules.get_sterimol_dict(base_atoms)
        self.show_result(f"Sterimol values:\n {sterimol_data}\n")

    def get_npa(self,base_atoms_str):
        if base_atoms_str:
            base_atoms = convert_to_list_or_nested_list(base_atoms_str)
            npa_data = self.molecules.get_npa_dict(base_atoms)
            self.show_result(f"NPA Charges:\n {npa_data}\n")

    def get_stretch(self,base_atoms_str):
        if base_atoms_str:
            base_atoms = convert_to_list_or_nested_list(base_atoms_str)
            stretch_data = self.molecules.get_stretch_vibration_dict(base_atoms)
            self.show_result(f"Stretch Vibration:\n {stretch_data}\n")

    def get_ring(self,base_atoms_str):
        if base_atoms_str:
            base_atoms = convert_to_list_or_nested_list(base_atoms_str)
            ring_data = self.molecules.get_ring_vibration_dict(base_atoms)
            self.show_result(f"Ring Vibrations:\n {ring_data}\n")

    def get_dipole(self,base_atoms_str):
        if base_atoms_str:
            base_atoms = convert_to_list_or_nested_list(base_atoms_str)
            dipole_data = self.molecules.get_dipole_dict(base_atoms)
            self.show_result(f"Dipole Moment:\n {dipole_data}\n")
    
    def get_bond_angle(self,base_atoms_str):
        if base_atoms_str:
            base_atoms = convert_to_list_or_nested_list(base_atoms_str)
            bond_angle_data = self.molecules.get_bond_angle_dict(base_atoms)
            self.show_result(f"Bond Angles:\n {bond_angle_data}\n")

    def get_bond_length(self,base_atoms_str):
        if base_atoms_str:
            base_atoms = convert_to_list_or_nested_list(base_atoms_str)
            bond_length_data = self.molecules.get_bond_length_dict(base_atoms)
            self.show_result(f"Bond Lengths:\n {bond_length_data}\n")

    def get_nbo(self,base_atoms_str):
        if base_atoms_str:
            # Split the string into two parts
            single_numbers_str, pairs_str = base_atoms_str.split(' ', 1)
            # Convert the first part to a list of numbers
            single_numbers = [int(num) for num in single_numbers_str.split(',')]
            # Split the second part into pairs and convert each pair to a list of numbers
            pairs = [[int(num) for num in pair.split(',')] for pair in pairs_str.split(' ')]
            # base_atoms = convert_to_list_or_nested_list(base_atoms_str)
            nbo_data = self.molecules.get_nbo_dict(single_numbers, pairs)
            self.show_result(f"NBO Analysis:\n {nbo_data}\n")

    def get_bending(self,base_atoms_str):
        if base_atoms_str:
            base_atoms = convert_to_list_or_nested_list(base_atoms_str)
            bending_data = self.molecules.get_bend_vibration_dict(base_atoms)
            self.show_result(f"Bending Vibrations:\n {bending_data}\n")

    def visualize_molecules(self):
        if self.molecules:
            self.molecules.visualize_molecules()

    def visualize_smallest_molecule(self):
        if self.molecules:
            self.molecules.visualize_smallest_molecule()
            html='my_plot.html'
            frame = HtmlFrame(root, horizontal_scrollbar="auto") #create the HTML browser
            frame.load_file(html) #load a website
            frame.pack(fill="both", expand=True) #attach the HtmlFrame widget to the parent window

    def export_data(self):
        self.molecules.extract_all_dfs()
        self.show_result(f"DataFrames extracted.")

    


    

root = Tk()
app = MoleculeApp(root)
root.mainloop()

