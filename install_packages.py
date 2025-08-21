#!/usr/bin/env python3

import subprocess
import sys

# List of packages to check/install
packages = [
    "pandas",
    "rdkit",
    "python-igraph",
    "XlsxWriter",
    "ipywidgets",
    "pyarrow",
    "plotly",
    "customtkinter",
    "chardet",
    "matplotlib",
    "rmsd",
    "networkx",
    "dash",
    "pyvista",
    "pyvistaqt",
    "morfeus-ml",
    "scikit-learn",
    "seaborn",
    "PIL",  # Note: PIL is often installed as Pillow
    "scipy",
    "tqdm",
    "statsmodels",
    "adjustText",
    "multiprocess",
    'random',
    'shap',
    'pymc'
]

# (Optional) Map any package name to its import name if they differ
# For example, "PIL" -> "PIL" or "python-igraph" -> "igraph"
import_name_map = {
    "PIL": "PIL",
    "python-igraph": "igraph",
    "morfeus-ml": "morfeus"
    # Add more if needed
}

def install_package(package_name):
    """
    Installs the given package using pip.
    """
    print(f"\nAttempting to install '{package_name}' ...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", package_name], 
                            capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Package installed successfully: {package_name}")
    else:
        print(f"Error installing {package_name}: {result.stderr}")

def check_import(package_name):
    """
    Attempts to import a package. If unsuccessful, installs the package and re-attempts.
    """
    # Determine the import name (if it differs from pip package name)
    mod_name = import_name_map.get(package_name, package_name)

    try:
        __import__(mod_name)
        print(f"Imported successfully: {package_name} (import as '{mod_name}')")
    except ImportError:
        install_package(package_name)
        # Retry import after installation
        try:
            __import__(mod_name)
            print(f"Imported successfully after install: {package_name} (import as '{mod_name}')")
        except ImportError as e:
            print(f"Failed to import {package_name} even after installation. Error: {e}")

def main():
    """
    Checks (and if necessary installs) all packages, then tries re-importing them.
    """
    print("Checking required packages...\n")
    for pkg in packages:
        check_import(pkg)
    print("\nAll done. If no errors were reported above, your environment is ready.")

if __name__ == "__main__":
    main()
