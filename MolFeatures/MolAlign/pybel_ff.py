#!/usr/bin/env python3
import os
import glob
import argparse
from openbabel import openbabel, pybel

def optimize_xyz_file(in_path, out_path, ff_name='MMFF94', steps=500):
    """
    Read a single XYZ file, run a force‐field optimization, and write out the result.
    """
    mol = next(pybel.readfile("xyz", in_path))
    ff = openbabel.OBForceField.FindForceField(ff_name)
    if ff is None:
        raise ValueError(f"Force field '{ff_name}' not found")
    ff.Setup(mol.OBMol)
    ff.SteepestDescent(steps)
    ff.GetCoordinates(mol.OBMol)
    mol.write("xyz", out_path, overwrite=True)

def optimize_directory(input_dir, output_dir, ff_name='MMFF94', steps=500):
    """
    Optimize all .xyz files in input_dir and save them to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(input_dir, "*.xyz")
    for in_path in glob.glob(pattern):
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_path = os.path.join(output_dir, f"{base}_opt.xyz")
        print(f"Optimizing {in_path} → {out_path}")
        optimize_xyz_file(in_path, out_path, ff_name=ff_name, steps=steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch force‐field optimize XYZ files with Open Babel"
    )
    parser.add_argument("input_dir", help="Directory containing .xyz files")
    parser.add_argument("output_dir", help="Directory to write optimized .xyz files")
    parser.add_argument(
        "--forcefield", "-f",
        default="MMFF94",
        help="Which force field to use (e.g. MMFF94, UFF)"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=500,
        help="Maximum number of optimization steps"
    )
    args = parser.parse_args()

    optimize_directory(
        args.input_dir,
        args.output_dir,
        ff_name=args.forcefield,
        steps=args.steps
    )
