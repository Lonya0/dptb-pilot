import os
import json
from typing import Optional, Literal, Dict, Any, TypedDict, Union, List
from pathlib import Path
import subprocess as sp

from dptb.entrypoints.emp_sk import EmpSK
from matplotlib import image as mpimg, pyplot as plt

from dptb_agent_tools.init_mcp import mcp

class ModelResult(TypedDict):
    model_path: str

class BandResult(TypedDict):
    band_path: str

@mcp.tool()
def band_with_sk_model(
    model_file_name: str = "your_model_file_name",
    structure_file_name: str = "your_structure_file_name",
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

    assert model_file_name != "your_model_file_name", "SK模型必须输入"
    assert structure_file_name != "your_structure_file_name", "输入的结构必须输入"

    os.makedirs("/tmp/running", exist_ok=True)
    os.chdir("/tmp/running")

    sp.run(['dptb', 'run', 'band', '-i', model_file_name, '-stu', structure_file_name, '-o', 'band_running'])

    plt.figure(figsize=(10, 8))
    img = mpimg.imread('./band_running/results/band.png')
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(os.path.join(work_path, "band.png"))

    return {"band_path": str(Path(work_path).absolute())}
