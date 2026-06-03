import shutil
from pathlib import Path
from typing import Optional, List
import subprocess as sp
import tempfile

from dptb_pilot.tools.modules.util.comm import generate_work_path, temporary_chdir


def _run_abacus(
        stru_path: Path,
        input_path: Path,
        kpt_path: Optional[Path] = None,
        command: str = "abacus",
        pp_orb_paths: List[Path] = None,
        other_file_path: Optional[Path] = None
    ):
    """
    运行Abacus

    参数:
        stru_path: Path,
        input_path: Path,
        kpt_path: Optional[Path] = None,
        command: str = "abacus",
        pp_orb_paths: List[Path] = None,

    返回:
        输出文件夹的路径

    抛出:
        AssumptionError: 某些数据输入不合规。
        RuntimeError: 写入配置文件失败。
    """
    work_path = Path(generate_work_path()).absolute()

    assert stru_path, "结构必须输入"
    assert input_path, "运行配置必须输入"

    with tempfile.TemporaryDirectory(dir=work_path) as temp_dir:
        temp_path = Path(temp_dir)

        shutil.copy(stru_path, temp_path / "STRU")
        shutil.copy(input_path, temp_path / "INPUT")
        if other_file_path:
            shutil.copytree(other_file_path, temp_path / other_file_path.name)
        if kpt_path:
            shutil.copy(kpt_path, temp_path / "KPT")
        for p in pp_orb_paths:
            shutil.copy(p, temp_path / p.name)

        cmd = command.split(" ")
        print(f"running abacus at {temp_path}")
        result = sp.run(cmd, cwd=temp_path, capture_output=True, text=True)

        if result.returncode != 0:
            full_output = "\n".join([
                "===== STDOUT =====",
                result.stdout,
                "===== STDERR =====",
                result.stderr
            ])
            raise RuntimeError(f"abacus execution failed:\n{full_output}")

        abacus_output_path = work_path / "OUT.ABACUS"
        shutil.copytree(temp_path / "OUT.ABACUS",
                        abacus_output_path,
                        dirs_exist_ok=True)

    return {"abacus_output_path": str(abacus_output_path)}

import shutil
from pathlib import Path
from typing import Optional, List
import subprocess as sp
import tempfile

from dptb_pilot.tools.modules.util.comm import generate_work_path

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import glob


from pathlib import Path
import re


def _abacus_get_efermi(abacus_out_path: Path):
    """
    从Abacus运行结果中提取费米能级
    优先使用 running_nscf.log，其次 fallback 到 running_*.log
    """

    # --- 1. 优先选择 running_nscf.log ---
    nscf_log = abacus_out_path / "running_nscf.log"

    if nscf_log.exists():
        log_file = nscf_log
    else:
        # --- 2. fallback: 查找 running_*.log ---
        log_files = list(abacus_out_path.glob("running_*.log"))
        if not log_files:
            raise FileNotFoundError("未找到 running_nscf.log 或 running_*.log 文件")

        # 可选：优先 scf（更稳定）
        log_files_sorted = sorted(log_files, key=lambda x: ("scf" not in x.name, x.name))
        log_file = log_files_sorted[0]

    # --- 3. 正则提取 EFERMI ---
    pattern = re.compile(r"EFERMI\s*=\s*([-\d\.Ee+]+)")

    efermi = None
    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                efermi = float(match.group(1))
                break

    if efermi is None:
        raise RuntimeError(f"未在 {log_file.name} 中找到 EFERMI")

    return {"efermi": efermi}


def _abacus_band_plot(
        abacus_out_path: Path,
        emin: float = None,
        emax: float = None
    ):
    """
    Abacus 的 out_band=1 结果能带绘制
    """

    work_path = Path(generate_work_path()).absolute()

    # --- 1. 获取费米能级 ---
    efermi_dict = _abacus_get_efermi(abacus_out_path=abacus_out_path)
    efermi = efermi_dict["efermi"]

    # --- 2. 读取 BANDS_1.dat ---
    band_file = abacus_out_path / "BANDS_1.dat"
    if not band_file.exists():
        raise FileNotFoundError("未找到 BANDS_1.dat")

    data = []
    with open(band_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            # 第一列: index
            # 第二列: k-path
            # 后面: 各个能带
            values = list(map(float, parts))
            data.append(values)

    data = np.array(data)

    kpoints = data[:, 1]
    bands = data[:, 2:]

    # --- 3. 相对费米能级 ---
    bands = bands - efermi

    # --- 4. 绘图 ---
    plt.figure(figsize=(6, 5))

    for i in range(bands.shape[1]):
        plt.plot(kpoints, bands[:, i], color="black", linewidth=1)

    # 费米能级线
    plt.axhline(0, linestyle="--")

    if not emin:
        emin = bands.min()
    if not emax:
        emax = bands.max()
    plt.ylim(emin, emax)
    plt.xlabel("k-path")
    plt.ylabel("Energy (eV)")
    plt.title("Band Structure")

    # --- 5. 保存 ---
    import time
    timestamp = int(time.time())
    band_img_filename = f"band_{timestamp}.png"
    band_img_path = work_path / band_img_filename

    plt.tight_layout()
    plt.savefig(band_img_path, dpi=300)
    plt.close()

    return {"abacus_output_path": str(band_img_path)}

def _abacus_band_gap(
        abacus_out_path: Path
    ):
    """
    从 Abacus 的 BANDS_1.dat 计算带隙

    返回:
        {
            "band_gap": float,
            "vbm": float,
            "cbm": float,
            "is_metal": bool
        }
    """

    # --- 1. 获取费米能级 ---
    efermi_dict = _abacus_get_efermi(abacus_out_path=abacus_out_path)
    efermi = efermi_dict["efermi"]

    # --- 2. 读取能带文件 ---
    band_file = abacus_out_path / "BANDS_1.dat"
    if not band_file.exists():
        raise FileNotFoundError("未找到 BANDS_1.dat")

    import numpy as np

    data = []
    with open(band_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            values = list(map(float, parts))
            data.append(values)

    data = np.array(data)

    # 能带数据
    bands = data[:, 2:]

    # --- 3. 相对费米能级 ---
    bands_rel = bands - efermi

    # --- 4. 展平成一维（全k点扫描） ---
    energies = bands_rel.flatten()

    # --- 5. 分离导带/价带 ---
    above = energies[energies > 0]   # 导带
    below = energies[energies <= 0]  # 价带

    # --- 6. 判断是否为金属 ---
    if len(above) == 0 or len(below) == 0:
        # 极端情况（全在一侧），直接视为金属
        return {
            "band_gap": 0.0,
            "vbm": None,
            "cbm": None
        }

    cbm = np.min(above)
    vbm = np.max(below)

    band_gap = cbm - vbm

    return {
        "band_gap": float(band_gap),
        "vbm": float(vbm),
        "cbm": float(cbm)
    }