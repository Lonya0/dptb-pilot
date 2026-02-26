from pathlib import Path
from typing import TypedDict, List

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.submodules.uniaxial_strain import _generate_uniaxial_strain_input_file


class GenerateUniaxialStrainInputResult(TypedDict):
    in_lammps_file_paths: List[Path]
    lammps_data_file_paths: List[Path]

@mcp.tool()
def generate_uniaxial_strain_input_file(
        output_root,
        poscar_file,
        strain_list,
        axis='auto',
        target_length=100,
        model_name='model.pb',
        relax_after_strain=True
) -> GenerateUniaxialStrainInputResult:
    """

    """

    return _generate_uniaxial_strain_input_file(output_root=output_root,
                                                poscar_file=poscar_file,
                                                strain_list=strain_list,
                                                axis=axis,
                                                target_length=target_length,
                                                model_name=model_name,
                                                relax_after_strain=relax_after_strain)


