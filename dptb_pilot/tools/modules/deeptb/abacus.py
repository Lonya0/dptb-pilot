from pathlib import Path
from typing import Dict, List, Optional

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.results_unified import (
    AbacusRunResult,
    BandGapResult,
    BandPlotResult,
    EfermiResult,
)
from dptb_pilot.tools.modules.deeptb.submodules.abacus import (
    _abacus_band_gap,
    _abacus_band_plot,
    _abacus_get_efermi,
    _run_abacus,
)


@mcp.tool()
def run_abacus(
        stru_path: Path,
        input_path: Path,
        kpt_path: Optional[Path] = None,
        command: str = "abacus",
        pp_orb_paths: List[Path] = None,
        other_file_path: Optional[Path] = None
) -> AbacusRunResult:
    """
    Run an ABACUS calculation from STRU/INPUT/KPT files and collect OUT.ABACUS.

    Parameters
    ----------
    stru_path : Path
        Path to the ABACUS ``STRU`` structure file.
    input_path : Path
        Path to the ABACUS ``INPUT`` control file.
    kpt_path : Path, optional
        Optional path to a ``KPT`` file.
    command : str, optional
        ABACUS executable command, for example ``abacus`` or ``mpirun -np 4 abacus``.
    pp_orb_paths : List[Path], optional
        Pseudopotential and numerical-orbital files that must be copied into the
        run directory.
    other_file_path : Path, optional
        Optional extra directory to copy into the temporary run directory.

    Returns
    -------
    AbacusRunResult
        Dictionary with ``abacus_output_path`` pointing to the copied output folder.
    """
    if pp_orb_paths is None:
        pp_orb_paths = []
    return _run_abacus(
        stru_path=stru_path,
        input_path=input_path,
        kpt_path=kpt_path,
        command=command,
        pp_orb_paths=pp_orb_paths,
        other_file_path=other_file_path,
    )


@mcp.tool()
def abacus_get_efermi(abacus_out_path: Path) -> EfermiResult:
    """
    Extract the Fermi level from an ABACUS output directory.

    The tool first checks ``running_nscf.log`` and then falls back to
    ``running_*.log`` files inside the ABACUS output directory.

    Parameters
    ----------
    abacus_out_path : Path
        Path to an ``OUT.ABACUS`` directory.

    Returns
    -------
    EfermiResult
        Dictionary containing the extracted ``efermi`` value.
    """
    return _abacus_get_efermi(abacus_out_path=abacus_out_path)


@mcp.tool()
def abacus_band_plot(
        abacus_out_path: Path,
        emin: float = None,
        emax: float = None
) -> BandPlotResult:
    """
    Plot an ABACUS band structure from ``BANDS_1.dat``.

    Parameters
    ----------
    abacus_out_path : Path
        Path to an ABACUS output directory containing ``BANDS_1.dat`` and a log
        file with ``EFERMI``.
    emin : float, optional
        Lower y-axis limit relative to the Fermi level.
    emax : float, optional
        Upper y-axis limit relative to the Fermi level.

    Returns
    -------
    BandPlotResult
        Dictionary with the generated band image path.
    """
    return _abacus_band_plot(abacus_out_path=abacus_out_path, emin=emin, emax=emax)


@mcp.tool()
def abacus_band_gap(abacus_out_path: Path) -> BandGapResult:
    """
    Calculate an ABACUS band gap from ``BANDS_1.dat`` relative to the Fermi level.

    Parameters
    ----------
    abacus_out_path : Path
        Path to an ABACUS output directory containing ``BANDS_1.dat`` and a log
        file with ``EFERMI``.

    Returns
    -------
    BandGapResult
        Dictionary with ``band_gap``, ``vbm``, and ``cbm`` values in eV.
    """
    return _abacus_band_gap(abacus_out_path=abacus_out_path)
