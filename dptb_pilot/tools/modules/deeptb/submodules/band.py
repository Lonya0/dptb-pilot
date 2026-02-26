import subprocess as sp
from pathlib import Path

import numpy as np
from matplotlib import image as mpimg, pyplot as plt



def find_fermi_level(eigenvalues, n_atoms, valence_electrons=4):
    """
    根据电子数守恒自动计算费米能级（0K）
    """

    # 展平并排序
    energies = eigenvalues.ravel()
    energies_sorted = np.sort(energies)

    # 总电子数
    N_e = valence_electrons * n_atoms

    # 占据能级数（自旋简并）
    N_occ = int(N_e / 2)

    if N_occ <= 0 or N_occ >= len(energies_sorted):
        raise ValueError("电子数与能级数不匹配")

    # 费米能级取 HOMO 与 LUMO 中点
    E_F = 0.5 * (
        energies_sorted[N_occ - 1] + energies_sorted[N_occ]
    )

    return E_F

def _band_gap(band_structure_file_path: Path, fermi_level=None, n_atoms=None):
    data = np.load(band_structure_file_path, allow_pickle=True).item()
    eig = data["eigenvalues"]

    if fermi_level is None:
        fermi_level = data.get("E_fermi", None)
    if fermi_level is None:
        if n_atoms is None:
            raise ValueError("必须提供 n_atoms 才能自动计算费米能级")
        print("自动查找费米能级——")
        fermi_level = find_fermi_level(eig, n_atoms)
        print(f"费米能级为：{fermi_level}")

    energies = eig.ravel()

    valence = energies[energies <= fermi_level]
    conduction = energies[energies > fermi_level]

    if valence.size == 0 or conduction.size == 0:
        return 0.0

    vbm = valence.max()
    cbm = conduction.min()

    return {"band_gap": cbm - vbm}


def _band_with_sk_model(
        model_file_path: Path,
        structure_file_path: Path,
        work_path: str = "."
    ):
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

    assert model_file_path, "SK模型必须输入"
    assert structure_file_path, "输入的结构必须输入"

    work_dir = Path(work_path).absolute()

    if not model_file_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    if not structure_file_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_file_path}")

    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy input files to temp dir
        shutil.copy(model_file_path, temp_path / model_file_path.name)
        shutil.copy(structure_file_path, temp_path / structure_file_path.name)

        cmd = ['dptb', 'run', 'band', '-i', model_file_path.name, '-stu', structure_file_path.name, '-o',
               'band_running']
        result = sp.run(cmd, cwd=temp_dir, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"dptb execution failed:\n{result.stderr}")

        # Check if result image exists
        bandstructure_path = temp_path / 'band_running' / 'results' / 'bandstructure.npy'
        if not bandstructure_path.exists():
            raise RuntimeError("Band calculation failed.")
        img_path = temp_path / 'band_running' / 'results' / 'band.png'
        if not img_path.exists():
            raise RuntimeError("Band calculation finished but no image generated.")

        # Copy result back to work_path
        import time
        timestamp = int(time.time())
        bandstructure_filename = f"bandstructure_{timestamp}.npy"
        output_bandstructure_path = work_dir / bandstructure_filename
        shutil.copy(bandstructure_path, output_bandstructure_path)

        band_img_filename = f"band_{timestamp}.png"
        output_band_img_path = work_dir / band_img_filename

        plt.figure(figsize=(10, 8))
        img = mpimg.imread(str(img_path))
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(output_band_img_path, bbox_inches='tight', pad_inches=0.0)
        plt.close()  # Close figure to free memory

        print(f"Band structure saved to {output_band_img_path}")

    return {"band_structure_file_path": output_bandstructure_path, "image_file_path": output_band_img_path}