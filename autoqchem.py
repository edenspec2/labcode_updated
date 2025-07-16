import autoqchem
# from autoqchem import AutoQChem

def run_autoqchem():
    # Initialize AutoQChem with default parameters
    aqc = autoqchem.AutoQChem()
    # Set up the molecule (example: water)
    aqc.set_molecule("H2O")
    # Set the calculation parameters
    aqc.set_parameters({
        "method": "DFT",
        "basis": "6-31G",
        "functional": "B3LYP"
    })
    # Run the calculation
    results = aqc.run_calculation()
    # Process and print the results
    print("Energy:", results['energy'])
    print("Geometry:", results['geometry'])
    print("Vibrational Frequencies:", results['frequencies'])
if __name__ == "__main__":
    run_autoqchem()