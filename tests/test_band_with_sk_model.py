from pathlib import Path

from dptb_pilot.tools.modules.deeptb.submodules.band import _band_with_sk_model, _band_gap

MODEL_FILE_PATH = Path('resources/env_corrected.pth')
STRUCTURE_FILE_PATH, FERMI_LEVEL, N_ATOMS = Path('resources/POSCAR_-3'), -9.22, 200
# STRUCTURE_FILE_PATH, FERMI_LEVEL, N_ATOMS = Path('resources/10_0.vasp'), -9.2, 40

def test_band_with_sk_model():
    OUTPUT_ROOT_PATH = Path('test_generated_band_with_sk_model')
    OUTPUT_ROOT_PATH.mkdir(exist_ok=True)
    band_structure_file_path, _ = _band_with_sk_model(model_file_path=MODEL_FILE_PATH,
                                                      structure_file_path=STRUCTURE_FILE_PATH,
                                                      work_path=str(OUTPUT_ROOT_PATH)).values()
    band_gap = _band_gap(band_structure_file_path=band_structure_file_path,
                         fermi_level=FERMI_LEVEL,
                         n_atoms=N_ATOMS).values()
    print(band_gap)
