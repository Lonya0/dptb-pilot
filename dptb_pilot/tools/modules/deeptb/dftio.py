from pathlib import Path

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.results_unified import DftioParseResult
from dptb_pilot.tools.modules.deeptb.submodules.dftio import _dftio_parse


@mcp.tool()
def dftio_parse(
        work_root: Path,
        mode: str = "abacus",
        prefix: str = "abacus",
        output_dir_name: str = "parse_result",
        out_hamiltonian: bool = False,
        out_overlap: bool = False,
        out_density_matrix: bool = False,
        out_eigenvalue: bool = False
) -> DftioParseResult:
    """
    Parse raw DFT outputs into DeePTB/dftio dataset files.

    Parameters
    ----------
    work_root : Path
        Root directory containing raw DFT calculation folders.
    mode : str, optional
        Parser backend passed to ``dftio parse``, usually ``abacus``.
    prefix : str, optional
        Prefix of calculation folders to parse.
    output_dir_name : str, optional
        Name of the output directory copied into the generated work path.
    out_hamiltonian : bool, optional
        Export Hamiltonian matrices.
    out_overlap : bool, optional
        Export overlap matrices.
    out_density_matrix : bool, optional
        Export density matrices.
    out_eigenvalue : bool, optional
        Export eigenvalues.

    Returns
    -------
    DftioParseResult
        Dictionary with ``output_path`` pointing to the parse result directory.
    """
    return _dftio_parse(
        work_root=work_root,
        mode=mode,
        prefix=prefix,
        output_dir_name=output_dir_name,
        out_hamiltonian=out_hamiltonian,
        out_overlap=out_overlap,
        out_density_matrix=out_density_matrix,
        out_eigenvalue=out_eigenvalue,
    )
