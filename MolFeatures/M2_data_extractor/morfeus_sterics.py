from __future__ import annotations
from typing import Iterable, Sequence, Optional, Dict, Any, Union
import numpy as np


def wrapper(
    mols,
    *,
    # You can pass explicit 1-indexed atoms; if None, they’ll be inferred per molecule
    metal_index: int | None = None,
    cone_center_index: int | None = None,
    z_axis_atoms: list[int] | None = None,   # [center, neighbor]
    xz_plane_atoms: list[int] | None = None, # [center, second neighbor]
    excluded_atoms=None,
    include_hs: bool = True,
    radius_first: float = 3.5,
    radius_second: float = 5.5,
    radii_type: str = "crc",
    radii_scale: float = 1.17,
    density: float = 0.001,
    cone_method: str = "libconeangle",
):
    """
    Build morfeus steric descriptors for each molecule in `mols.molecules`.

    Notes:
    - morfeus expects 1-INDEXED atom indices.
    - If any of metal_index / cone_center_index / z_axis_atoms / xz_plane_atoms are None
      or out of range for a given molecule, they are inferred automatically:
        - center = first heavy (non-H) atom, or atom #1 if all H
        - z-axis neighbor = nearest heavy neighbor to center (by distance)
        - xz-plane neighbor = second nearest heavy neighbor (distinct from z-axis neighbor)
    """
    results_list = []

    for idx, mol in enumerate(mols.molecules, start=1):
        try:
            elements = mol.xyz_df["atom"].astype(str).tolist()
            coords = mol.xyz_df[["x", "y", "z"]].to_numpy(dtype=float)
            n_atoms = len(elements)

            def in_range(i: int | None) -> bool:
                return i is not None and 1 <= int(i) <= n_atoms

            # --- auto-select center and neighbors when needed ---
            # choose a center: first heavy atom (non-H), else atom 1
            heavy = [i for i, a in enumerate(elements) if a.upper() != "H"]
            center0 = heavy[0] if heavy else 0  # 0-based
            auto_center = center0 + 1           # 1-based

            m_idx = metal_index if in_range(metal_index) else auto_center
            c_idx = cone_center_index if in_range(cone_center_index) else m_idx
            c0 = int(c_idx) - 1  # 0-based for distance work

            def nearest_neighbors(center_zero_based: int, k: int = 3):
                # returns a list of 0-based neighbor indices (excluding center), sorted by distance
                d2 = np.sum((coords - coords[center_zero_based])**2, axis=1)
                order = np.argsort(d2).tolist()
                order = [i for i in order if i != center_zero_based]
                return order[:k]

            # infer z-axis pair [center, neighbor]
            if (
                z_axis_atoms is None
                or len(z_axis_atoms) != 2
                or not all(in_range(i) for i in z_axis_atoms)
                or z_axis_atoms[0] == z_axis_atoms[1]
            ):
                nn = nearest_neighbors(c0, k=1)
                z_pair = [c_idx, (nn[0] + 1) if nn else c_idx]
            else:
                z_pair = [int(z_axis_atoms[0]), int(z_axis_atoms[1])]

            # infer xz-plane pair [center, second neighbor distinct from z-axis neighbor]
            if (
                xz_plane_atoms is None
                or len(xz_plane_atoms) != 2
                or not all(in_range(i) for i in xz_plane_atoms)
                or len(set(xz_plane_atoms)) < 2
            ):
                nn2 = nearest_neighbors(c0, k=2)
                # choose a neighbor different from z_pair[1]
                z_nb_0 = int(z_pair[1]) - 1
                alt0 = next((j for j in nn2 if j != z_nb_0), (nn2[0] if nn2 else c0))
                xz_pair = [c_idx, alt0 + 1]
            else:
                xz_pair = [int(xz_plane_atoms[0]), int(xz_plane_atoms[1])]

            # --- call morfeus ---
            res = morfeus_sterics(
                elements=elements,
                coordinates=coords,
                metal_index=int(m_idx),
                cone_center_index=int(c_idx),
                excluded_atoms=excluded_atoms,
                include_hs=include_hs,
                radius_first=float(radius_first),
                radius_second=float(radius_second),
                radii_type=str(radii_type),
                radii_scale=float(radii_scale),
                density=float(density),
                z_axis_atoms=[int(z_pair[0]), int(z_pair[1])],
                xz_plane_atoms=[int(xz_pair[0]), int(xz_pair[1])],
                cone_method=str(cone_method),
            )

        except Exception as e:
            name = (
                getattr(mol, "molecule_name", None)
                or getattr(mol, "name", None)
                or f"mol_{idx}"
            )
            print(f"[WARN] wrapper: failed for {name}: {e}")
            # Return a minimal dict so downstream code keeps working
            res = {
                "cone_angle_deg": np.nan,
                "vbur_first_percent": np.nan,
                "vbur_second_percent": np.nan,
                "vbur_first_volume_A3": np.nan,
                "free_volume_first_A3": np.nan,
                "vbur_second_volume_A3": np.nan,
                "free_volume_second_A3": np.nan,
                "donor_index_used": np.nan,
                "donor_guess_used": False,
            }

        results_list.append(res)

    return results_list

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any

def morfeus_results_to_df(
    mols,
    *,
    sort_indices: bool = True,
    # --- optional Sterimol integration ---
    add_sterimol: bool = False,
    sterimol_atoms=None,           # e.g., [12, 5, 11] (1-indexed) if you want to force the axis
    sterimol_radii: str = "CPK",
    sterimol_drop_atoms=None,
    sterimol_kwargs=None,
) -> pd.DataFrame:
    """
    Build a tidy DataFrame from morfeus_sterics results (wrapper(mols)).
    Index = molecule name (if available), else mol_#.

    If add_sterimol=True, also computes Sterimol (B1, B5, L) per molecule via
    Molecule.get_sterimol(...). If sterimol_atoms is None, tries to infer or falls back.
    """
    if sterimol_kwargs is None:
        sterimol_kwargs = {}
    if sterimol_drop_atoms is None:
        sterimol_drop_atoms = []

    rows = {}
    # Use ASCII minus to avoid encoding issues
    oct_labels = ["+x+y+z", "+x+y-z", "+x-y+z", "+x-y-z",
                  "-x+y+z", "-x+y-z", "-x-y+z", "-x-y-z"]

    # `wrapper(mols)` must exist in your module and return list of dicts
    results_list = wrapper(mols)

    for i, (mol, res) in enumerate(zip(mols.molecules, results_list), start=1):
        name = (
            getattr(mol, "name", None)
            or getattr(mol, "molecule_name", None)
            or getattr(mol, "id", None)
            or f"mol_{i}"
        )

        row = {
            "cone_angle_deg":          res.get("cone_angle_deg", np.nan),
            "%Vbur_first":             res.get("vbur_first_percent", np.nan),
            "%Vbur_second":            res.get("vbur_second_percent", np.nan),
            "Δ%Vbur(second-first)":    (
                res.get("vbur_second_percent", np.nan) - res.get("vbur_first_percent", np.nan)
            ),
            "Vbur1_volume_A3":         res.get("vbur_first_volume_A3", np.nan),
            "Free1_volume_A3":         res.get("free_volume_first_A3", np.nan),
            "Vbur2_volume_A3":         res.get("vbur_second_volume_A3", np.nan),
            "Free2_volume_A3":         res.get("free_volume_second_A3", np.nan),
            "donor_index_used":        res.get("donor_index_used", np.nan),
            "donor_guess_used":        bool(res.get("donor_guess_used", False)),
        }

        # Quadrants (Q1..Q4) if present
        q = res.get("quadrants_first", None)
        if q is not None:
            if isinstance(q, dict):
                for k in ["Q1","Q2","Q3","Q4"]:
                    row[f"{k}_percent"] = float(q.get(k, np.nan))
            else:
                q_list = list(q)
                for idx, k in enumerate(["Q1","Q2","Q3","Q4"]):
                    row[f"{k}_percent"] = float(q_list[idx]) if idx < len(q_list) else np.nan

        # Octants (8 bins) if present
        o = res.get("octants_first", None)
        if o is not None:
            if isinstance(o, dict):
                for lab in oct_labels:
                    row[f"oct_{lab}_percent"] = float(o.get(lab, np.nan))
            else:
                o_list = list(o)
                for idx, lab in enumerate(oct_labels):
                    row[f"oct_{lab}_percent"] = float(o_list[idx]) if idx < len(o_list) else np.nan

        # -------- Sterimol (optional) --------
        if add_sterimol:
            try:
                base_atoms = None

                # 1) If provided explicitly, use them (1-indexed expected)
                if sterimol_atoms is not None:
                    base_atoms = list(sterimol_atoms)

                # 2) Else try to infer from your utilities, if available
                if base_atoms is None:
                    try:
                        # optional: only if your project exposes it
                        from MolFeatures.M2_data_extractor.extractor_utils.sterimol_utils import get_sterimol_indices
                        base_atoms = get_sterimol_indices(mol.xyz_df, mol.bonds_df)
                    except Exception:
                        base_atoms = None

                # 3) Fallback heuristic: pick a heavy atom + its nearest heavy neighbor
                if base_atoms is None:
                    try:
                        df_xyz = mol.xyz_df
                        coords = df_xyz[["x", "y", "z"]].to_numpy(float)
                        atoms  = df_xyz["atom"].astype(str).tolist()
                        heavy  = [i for i, a in enumerate(atoms) if a.upper() != "H"]
                        if len(heavy) >= 2:
                            i0 = heavy[0]
                            d2 = np.sum((coords[heavy] - coords[i0])**2, axis=1)
                            j0 = heavy[int(np.argsort(d2)[1])]  # 2nd is nearest neighbor
                            base_atoms = [i0 + 1, j0 + 1]       # 1-indexed
                        else:
                            base_atoms = [1, 2]
                    except Exception:
                        base_atoms = [1, 2]

                st_df = mol.get_sterimol(
                    base_atoms,
                    radii=sterimol_radii,
                    sub_structure=True,
                    drop_atoms=sterimol_drop_atoms or [],
                    visualize_bool=None,
                    mode="all",
                    **(sterimol_kwargs or {})
                )
                row.update(_extract_sterimol_metrics(st_df))
            except Exception as e:
                row.update({"sterimol_B1": np.nan, "sterimol_B5": np.nan, "sterimol_L": np.nan})
                print(f"[WARN] Sterimol failed for {name}: {e}")

        rows[name] = row

    df = pd.DataFrame.from_dict(rows, orient="index")

    # Optional: numeric-aware sort for IDs like "LS2002"
    if sort_indices:
        try:
            df = df.sort_index(
                key=lambda idx: idx.map(
                    lambda x: int(re.search(r"\d+", str(x)).group())
                    if re.search(r"\d+", str(x)) else float("inf")
                )
            )
        except Exception:
            pass

    return df





def _extract_sterimol_metrics(st_df) -> Dict[str, float]:
    """
    Accepts the DataFrame returned by your Molecule.get_sterimol(...)
    and returns a dict with B1, B5, L (robust to multiple rows/NaNs).
    Policy:
      - B1: min over rows
      - B5: max over rows
      - L : max over rows
    """
    import numpy as np
    import pandas as pd

    out = {"sterimol_B1": np.nan, "sterimol_B5": np.nan, "sterimol_L": np.nan}
    if st_df is None or not hasattr(st_df, "columns"):
        return out

    # Some implementations return different shapes; standardize.
    cols = set(map(str, st_df.columns))
    get = lambda c: st_df[str(c)] if str(c) in cols else None

    b1 = get("B1")
    b5 = get("B5")
    L  = get("L")

    # Reduce to scalars if series exist
    if b1 is not None and len(b1):
        out["sterimol_B1"] = float(np.nanmin(pd.to_numeric(b1, errors="coerce").values))
    if b5 is not None and len(b5):
        out["sterimol_B5"] = float(np.nanmax(pd.to_numeric(b5, errors="coerce").values))
    if L is not None and len(L):
        out["sterimol_L"]  = float(np.nanmax(pd.to_numeric(L,  errors="coerce").values))

    return out
