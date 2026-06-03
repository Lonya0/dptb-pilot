from pathlib import Path
from typing import TypedDict


class ModelResult(TypedDict):
    model_path: Path

class BandResult(TypedDict):
    band_structure_file_path: Path
    image_file_path: Path
    fermi_level: float

class HamiltonianResult(TypedDict):
    hamiltonian_file_path: Path
    overlap_file_path: Path

class RunNegfResult(TypedDict):
    negf_result_file_path: Path
    log_file_path: Path

class RunLammpsResult(TypedDict):
    relaxed_system_file_path: Path

class ConfigResult(TypedDict):
    config_path: Path