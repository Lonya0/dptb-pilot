"""
Generate CNT supercell and apply uniaxial strain
consistent with Yang & Han PRL 1999 definition:

ε = (L - L0) / L0
"""

from pathlib import Path

import numpy as np
from ase.data import atomic_numbers, atomic_masses
from ase.io import read, write

from dptb_pilot.tools.modules.deeptb.submodules.supercell import build_supercell


def apply_uniaxial_strain(atoms, axis, strain_percent):
    """
    Apply affine uniaxial strain along axis
    """
    eps = strain_percent / 100.0

    cell = atoms.get_cell()
    new_cell = cell.copy()

    new_cell[axis] *= (1.0 + eps)

    atoms.set_cell(new_cell, scale_atoms=True)

    return atoms


def write_lammps_input(task_dir, symbols, relax=False, model_name='model.pb'):
    """
    symbols: unique chemical symbols in specorder order
    """

    with open(task_dir / "in.lammps", "w") as f:

        f.write(f"""
units           metal
atom_style      atomic
boundary        p p p

read_data       lammps.data

""")

        # ----- MASS SECTION -----
        for i, sym in enumerate(symbols, start=1):
            Z = atomic_numbers[sym]
            mass = atomic_masses[Z]
            f.write(f"mass {i} {mass:.6f}  # {sym}\n")

        f.write(f"""
pair_style      deepmd {model_name}
pair_coeff      * * {' '.join(symbols)}

neighbor        2.0 bin
neigh_modify    every 10 delay 0 check yes

thermo          100

""")

        if relax:
            f.write("""
min_style       cg
minimize        1e-6 1e-8 1000 10000
""")
        else:
            f.write("run 0\n")

        f.write("""
write_data      relaxed.data
""")

def _generate_uniaxial_strain_input_file(output_root,
                                         poscar_file,
                                         strain_list,
                                         axis='auto',
                                         target_length=100,
                                         model_name='model.pb',
                                         relax_after_strain=True):
    OUTPUT_ROOT_PATH = Path(output_root)
    OUTPUT_ROOT_PATH.mkdir(exist_ok=True)

    atoms = read(poscar_file)

    supercell, axis, _ = build_supercell(atoms,
                                         axis=axis,
                                         target_length=target_length)

    L0 = np.linalg.norm(supercell.get_cell()[axis])
    print(f"Reference length L0 = {L0:.3f} Å")

    # 获取元素顺序（用于 specorder）
    symbols = []
    for s in supercell.get_chemical_symbols():
        if s not in symbols:
            symbols.append(s)

    in_lammps_file_paths = []
    lammps_data_file_paths = []

    for strain in strain_list:
        task_dir = OUTPUT_ROOT_PATH / f"strain_{strain:+.2f}pct"
        task_dir.mkdir(exist_ok=True)

        strained_atoms = supercell.copy()
        strained_atoms = apply_uniaxial_strain(
            strained_atoms, axis, strain
        )

        # Write POSCAR
        write(task_dir / "POSCAR", strained_atoms)

        # Write LAMMPS data
        write(task_dir / "lammps.data",
              strained_atoms,
              format="lammps-data",
              specorder=symbols)
        lammps_data_file_paths.append(Path(task_dir / "lammps.data"))

        # Write LAMMPS input
        write_lammps_input(task_dir,
                           symbols=symbols,
                           relax=relax_after_strain,
                           model_name=model_name)
        in_lammps_file_paths.append(Path(task_dir / "in.lammps"))

        print(f"Generated strain {strain:+.2f}%")

    print("All strain structures generated.")
    return {"in_lammps_file_paths":in_lammps_file_paths, "lammps_data_file_paths":lammps_data_file_paths}