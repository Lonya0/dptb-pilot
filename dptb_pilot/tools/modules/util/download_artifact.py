import os
from pathlib import Path
from typing import TypedDict
from dptb_pilot.tools.init import mcp
from dp.agent.server.storage import BohriumStorage

class FilePathResult(TypedDict):
    file_path: str

@mcp.tool()
def download_artifact(
        artifact_url: str,
        work_path: str = "."
) -> FilePathResult:
    """
    将Artifact从Bohrium存储下载到本地工作目录。

    支持两种URL格式:
    1. bohrium:// 协议的URL (例如: bohrium://path/to/file.txt)
    2. 普通路径 (例如: /remote/path/file.txt)

    参数:
        artifact_url: Artifact的URL或路径，支持 bohrium:// 协议
        work_path: 本地工作路径，文件将下载到此目录（由程序自动传入）

    返回:
        包含本地文件路径的字典。

    抛出:
        ValueError: URL格式错误或为空
        RuntimeError: 下载失败
    """

    if not artifact_url or artifact_url.strip() == "":
        raise ValueError("artifact_url不能为空")

    # 解析URL
    artifact_url = artifact_url.strip()

    # 检查是否是bohrium://协议
    if artifact_url.startswith("bohrium://"):
        # 移除协议前缀，获取实际的存储路径
        storage_key = artifact_url[len("bohrium://"):]
    else:
        # 如果不是bohrium://协议，直接使用路径
        storage_key = artifact_url

    # 确保工作目录存在
    work_dir = Path(work_path).absolute()
    work_dir.mkdir(parents=True, exist_ok=True)

    # 获取文件名（从路径中提取）
    file_name = os.path.basename(storage_key.split("?")[0])
    if not file_name:
        file_name = "downloaded_file"

    # 本地目标路径
    local_path = work_dir / file_name

    try:
        # 初始化Bohrium存储客户端
        storage = BohriumStorage()

        # 下载文件
        # download方法会自动处理目录和压缩文件
        downloaded_path = storage.download(key=storage_key, path=str(work_dir))

        return {"file_path": str(Path(downloaded_path).absolute())}

    except Exception as e:
        raise RuntimeError(f"下载artifact失败: {str(e)}")
