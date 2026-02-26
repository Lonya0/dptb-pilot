from pathlib import Path

from dptb_pilot.tools.modules.deeptb.submodules.convert import convert_from_lammps_data

def test_convert_from_lammps_data_to_vasp():
    convert_from_lammps_data(Path('resources/relaxed+3.data'),
                             output_name='test_generated_vasp_poscar_file',
                             format='vasp')

def test_convert_from_lammps_data_to_abacus():
    convert_from_lammps_data(Path('resources/relaxed-3.data'),
                             output_name='test_generated_abacus_stru_file',
                             format='abacus')