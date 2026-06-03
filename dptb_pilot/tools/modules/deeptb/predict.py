from pathlib import Path
from typing import TypedDict, Dict

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.results_unified import BandResult, HamiltonianResult
from dptb_pilot.tools.modules.deeptb.submodules.band import _band_predict
from dptb_pilot.tools.modules.deeptb.submodules.hamiltonian import _hamiltonian_predict

@mcp.tool()
def band_predict(
    model_file_path: Path,
    structure_file_path: Path,
    kpath: str,
    nel_atom: Dict[str, int],
    kmesh: str = None,
    work_path: str = "."
) -> BandResult:
    """
    使用输入的SK模型预测结构能带。

    参数:
        model_file_name: 使用的model路径。
        structure_file_name: 输入的结构文件路径。结构文件应为vasp的格式
        kpath: K-Path，格式如"[[0.0,0.0,0.0,50,G],[0.5,0.0,0.5,1,X]]"
        nel_atom: Dictionary mapping element symbols to number of valence electrons. Example: {'Si': 4, 'H': 1}
        kmesh: 用于计算费米能级的k点网格，默认为[5,5,5]，格式如"[5,5,5]"
        work_path: 能带信息的保存路径。注意应该是文件夹而不是文件。

    返回:
        包含能带文件路径的字典。

    抛出:
        AssumptionError: 某些数据输入不合规。
        RuntimeError: 写入配置文件失败。
    """

    return _band_predict(model_file_path=model_file_path,
                       structure_file_path=structure_file_path,
                       kpath=kpath,
                       nel_atom=nel_atom,
                       kmesh=kmesh,
                       work_path=work_path)


@mcp.tool()
def hamiltonian_predict(
        model_file_path: Path,
        structure_file_path: Path,
        k_points: str,
        override_overlap: Path = None,
        work_path: str = "."
) -> HamiltonianResult:
    """
    使用输入的SK模型预测结构哈密顿量。

    参数:
        model_file_name: 使用的model路径。
        structure_file_name: 输入的结构文件路径。结构文件应为vasp的格式
        k_points: 想要计算的k点，格式如"[[0,0,0],[0,0,0.5]]"
        override_overlap: 覆盖的overlap文件，使用后覆盖模型产生的overlap
        work_path: 哈密顿量信息的保存路径。注意应该是文件夹而不是文件。

    返回:
        包含能带文件路径的字典。

    抛出:
        AssumptionError: 某些数据输入不合规。
        RuntimeError: 写入配置文件失败。
    """

    return _hamiltonian_predict(model_file_path=model_file_path,
                              structure_file_path=structure_file_path,
                              k_points=k_points,
                              override_overlap=override_overlap,
                              work_path=work_path)