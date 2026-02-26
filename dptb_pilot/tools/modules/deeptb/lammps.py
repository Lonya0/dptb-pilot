from pathlib import Path
from typing import TypedDict

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.submodules.lammps import _run_lammps

class RunLammpsResult(TypedDict):
    relaxed_system_file_path: Path

@mcp.tool()
def run_lammps(
        in_lammps_file_path: Path,
        lammps_data_file_path: Path,
        deepmd_model_file_path: Path = None,
        lmp_command: str = 'lmp'
) -> RunLammpsResult:
    """

    """

    return _run_lammps(in_lammps_file_path=in_lammps_file_path,
                       lammps_data_file_path=lammps_data_file_path,
                       deepmd_model_file_path=deepmd_model_file_path,
                       lmp_command=lmp_command)