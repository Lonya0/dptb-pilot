import shutil
import tempfile
from pathlib import Path
import subprocess as sp
import tempfile
import os
import shutil
from dftio.io.abacus.abacus_parser import AbacusParser
from tqdm import tqdm

from dptb_pilot.tools.modules.util.comm import generate_work_path

def _dftio_parse_abacus(
        work_root: Path,
        prefix: str = "abacus",
        output_dir_name: str = "parse_result",
        out_hamiltonian: bool = False,
        out_overlap: bool = False,
        out_density_matrix: bool = False,
        out_eigenvalue: bool = False
    ):
    work_root = work_root.absolute()

    parser = AbacusParser(
        root=str(work_root),
        prefix=prefix,
    )
    num_entries = len(parser.raw_datas)

    for idx in tqdm(range(num_entries)):
        parser.write_hdf5(
            idx=int(idx),
            hamiltonian=out_hamiltonian,
            overlap=out_overlap,
            outroot=output_dir_name,
            eigenvalue=out_eigenvalue,
            density_matrix=out_density_matrix,
            band_index_min=0
        )

def _dftio_parse(
        work_root: Path,
        mode: str = "abacus",
        prefix: str = "abacus",
        output_dir_name: str = "parse_result",
        out_hamiltonian: bool = False,
        out_overlap: bool = False,
        out_density_matrix: bool = False,
        out_eigenvalue: bool = False
    ):
    """
    使用输入的模型预测结构哈密顿量。

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
    work_path = Path(generate_work_path()).absolute()

    assert work_root, "根路径必须输入"

    with tempfile.TemporaryDirectory(dir=work_path) as temp_dir:
        temp_path = Path(temp_dir)

        cmd = ["dftio", "parse",
               "-m", mode,
               "-r", work_root,
               "-p", prefix,
               "-o", temp_path]

        if out_hamiltonian:
            cmd.append("-ham")
        if out_overlap:
            cmd.append("-ovp")
        if out_density_matrix:
            cmd.append("-dm")
        if out_eigenvalue:
            cmd.append("-eig")

        result = sp.run(cmd, cwd=temp_dir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"dftio execution failed:\n{result.stderr}")

        output_path = work_path / output_dir_name
        shutil.copytree(temp_path,
                        output_path,
                        dirs_exist_ok=True)

    return {"output_path": output_path}




