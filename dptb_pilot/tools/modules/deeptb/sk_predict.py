from pathlib import Path
from typing import TypedDict

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.submodules.band import _band_with_sk_model

class BandResult(TypedDict):
    band_structure_file_path: Path
    image_file_path: Path

@mcp.tool()
def band_with_sk_model(
    model_file_path: Path,
    structure_file_path: Path,
    work_path: str = "."
) -> BandResult:
    """
    使用输入的SK模型预测结构能带。

    参数:
        model_file_name: 使用的model路径。
        structure_file_name: 输入的结构文件路径。结构文件应为vasp的格式
        work_path: 能带信息的保存路径。注意应该是文件夹而不是文件。

    返回:
        包含能带文件路径的字典。

    抛出:
        AssumptionError: 某些数据输入不合规。
        RuntimeError: 写入配置文件失败。
    """

    return _band_with_sk_model(model_file_path=model_file_path,
                               structure_file_path=structure_file_path,
                               work_path=work_path)