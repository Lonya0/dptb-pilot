from pathlib import Path
from typing import TypedDict, List

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.submodules.uniaxial_strain import _generate_uniaxial_strain_input_file

class GenerateUniaxialStrainInputResult(TypedDict):
    in_lammps_file_paths: List[Path]
    lammps_data_file_paths: List[Path]

@mcp.tool()
def generate_uniaxial_strain_input_file(
        struture_file_path:Path,
        strain_list:List[float],
        axis='auto',
        target_length=20,
        model_name='model.pb',
        relax_after_strain=True
) -> GenerateUniaxialStrainInputResult:
    """
    Generate LAMMPS input files for uniaxial strain calculations along a specified axis.

    This tool creates a series of strained structures by applying uniaxial strain to a
    given crystal structure. For each strain value in the provided list, it:
    1. Builds a supercell along the specified axis to achieve a target length
    2. Applies the specified uniaxial strain
    3. Creates a separate directory with:
       - POSCAR file of the strained structure
       - LAMMPS data file (lammps.data) with atomic positions and cell information
       - LAMMPS input script (in.lammps) configured for relaxation with a machine learning potential

    The generated files are organized in subdirectories named by strain percentage
    (e.g., 'strain_+1.00pct/', 'strain_-0.50pct/') for easy management of multiple
    calculations.

    Parameters
    ----------
    struture_file_path : Path
        Path to the input structure file (POSCAR format). The file should contain
        the crystal structure to be strained.

    strain_list : List[float]
        List of strain values to apply, given as percentages. Positive values
        indicate tensile strain, negative values indicate compressive strain.
        Example: [1.0, 0.5, 0.0, -0.5, -1.0] for ±1% strain.

    axis : str or int, optional (default='auto')
        Crystallographic axis along which to apply strain. Options:
        - 'auto': Automatically select the axis that gives the desired target length
        - 'x', 'y', 'z': Manually specify the axis index ('x' for x-axis, 'y' for y-axis, 'z' for z-axis)

    target_length : float, optional (default=20)
        Target length in Angstroms for the supercell along the strain axis.
        The tool will create a supercell that approximates this length to ensure
        sufficient periodic length for accurate strain calculations.

    model_name : str, optional (default='model.pb')
        Name of the machine learning potential file (in PB format) to be used
        in the LAMMPS calculations. This file should be present in the working
        directory or specified with full path.

    relax_after_strain : bool, optional (default=True)
        Whether to include relaxation commands in the generated LAMMPS input script.
        If True, the LAMMPS script will perform structural relaxation after applying
        strain. If False, only a single-point calculation will be performed.

    Returns
    -------
    GenerateUniaxialStrainInputResult
        A dictionary containing:
        - in_lammps_file_paths: List of paths to generated LAMMPS input files (in.lammps)
          for each strain value
        - lammps_data_file_paths: List of paths to generated LAMMPS data files (lammps.data)
          for each strain value

    Notes
    -----
    The tool performs the following steps:
    1. Reads the input structure from the POSCAR file
    2. Builds an appropriate supercell to achieve the target length along the strain axis
    3. For each strain value in strain_list:
       a. Creates a subdirectory named 'strain_{value:+.2f}pct'
       b. Applies the uniaxial strain by scaling the lattice vectors
       c. Writes the strained structure to POSCAR format
       d. Converts to LAMMPS data format with proper element ordering
       e. Generates a LAMMPS input script with the specified potential

    The generated LAMMPS input scripts are configured to use the specified machine
    learning potential and include appropriate settings for accurate structure
    relaxation.

    Example
    -------
    >>> from pathlib import Path
    >>> result = generate_uniaxial_strain_input_file(
    ...     struture_file_path=Path("./POSCAR"),
    ...     strain_list=[-1.0, -0.5, 0.0, 0.5, 1.0],
    ...     axis='z',  # strain along z-axis
    ...     target_length=25.0,
    ...     model_name="my_potential.pb",
    ...     relax_after_strain=True
    ... )
    >>> print(f"Generated {len(result['in_lammps_file_paths'])} input files")
    """

    return _generate_uniaxial_strain_input_file(output_root='.',
                                                poscar_file=struture_file_path,
                                                strain_list=strain_list,
                                                axis=axis,
                                                target_length=target_length,
                                                model_name=model_name,
                                                relax_after_strain=relax_after_strain)


