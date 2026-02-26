import os
from pathlib import Path

import dpdata
from ase.io import read, write


def convert_from_lammps_data(
    input: Path,
    output_name: str = "auto",
    format: str = "vasp",
    pp_filename=None,
    orb_filename=None
) -> Path:
    """
    Convert LAMMPS data file to VASP (POSCAR) or ABACUS (STRU).

    Parameters
    ----------
    input : Path
        Input LAMMPS data file.
    output_name : str
        Output file name. If "auto", use default naming.
    format : str
        Target format: "vasp" or "abacus".

    Returns
    -------
    Path
        Path to written structure file.
    """

    input = Path(input).resolve()

    if not input.exists():
        raise FileNotFoundError(f"Input file not found: {input}")

    if format not in ("vasp", "abacus"):
        raise ValueError("format must be 'vasp' or 'abacus'")

    # 读取 LAMMPS data 文件
    atoms = read(input, format="lammps-data")

    # ===== 自动命名逻辑 =====
    if output_name == "auto":
        if format == "vasp":
            filename = "POSCAR"
        elif format == "abacus":
            filename = "STRU"
    else:
        filename = output_name

    output_path = input.parent / filename

    # ===== 避免覆盖 =====
    counter = 1
    base = output_path
    while output_path.exists():
        output_path = base.with_name(f"{base.name}_{counter}")
        counter += 1

    # ===== 写出 =====
    if format == "vasp":
        write(output_path, atoms, format='vasp')
    if format == "abacus":
        write(input.parent / 'POSCAR', atoms, format='vasp')
        (dpdata.System(input.parent / 'POSCAR', fmt="vasp/poscar")
         .to("abacus/stru", output_path,
             pp_file=[f"../../{pp_filename}"],
             numerical_orbital=[f"../../{orb_filename}"]))

    print(f"Written {format.upper()} file: {output_path}")

    return output_path