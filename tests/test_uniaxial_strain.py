from dptb_pilot.tools.modules.deeptb.submodules.uniaxial_strain import _generate_uniaxial_strain_input_file

STRUCTURE_FILE = "resources/10_0.vasp"
OUTPUT_ROOT = "test_generated_uniaxial_strain"

TARGET_LENGTH = 20.0  # Å, target supercell axial length
AXIS = 'z'          # x,y,z or auto

# strain values in percent (same meaning as PRL 1999)
STRAIN_LIST = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

MODEL_NAME = "model.pb"

RELAX_AFTER_STRAIN = True  # True -> LAMMPS minimize

def test_uniaxial_strain():
    _generate_uniaxial_strain_input_file(OUTPUT_ROOT,
                                         STRUCTURE_FILE,
                                         strain_list=STRAIN_LIST,
                                         axis=AXIS,
                                         target_length=TARGET_LENGTH,
                                         model_name=MODEL_NAME,
                                         relax_after_strain=RELAX_AFTER_STRAIN)