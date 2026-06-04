# Installation

{mod}`cellrank` requires Python ≥ 3.12.
If you encounter any problems, feel free to open an [issue].

## pip / uv (recommended)

Install CellRank from [PyPI](https://pypi.org/project/cellrank/):

```bash
pip install cellrank
```

Or, using [uv](https://docs.astral.sh/uv/), a fast, modern pip replacement:

```bash
uv pip install cellrank
```

### Optional extras

CellRank provides optional dependency groups for common workflows:

```bash
pip install "cellrank[moscot]"  # moscot for optimal transport (RealTimeKernel)
pip install "cellrank[plot]"    # additional plotting dependencies
```

## conda / mamba / pixi

CellRank is available on [conda-forge](https://anaconda.org/conda-forge/cellrank).
This installation method is recommended if you need [PETSc] and [SLEPc],
libraries for large-scale linear algebra that CellRank uses when computing
macrostates or fate probabilities on large datasets:

```bash
mamba install -c conda-forge cellrank
```

Or, using [pixi](https://pixi.sh/), a fast, modern conda replacement:

```bash
pixi add cellrank
```

To also install PETSc and SLEPc (conda-only packages):

```bash
mamba install -c conda-forge cellrank petsc4py slepc4py
```

```{note}
**Windows users:** [PETSc] and [SLEPc] are not built for Windows on conda-forge
(only Linux and macOS). Because the conda-forge `cellrank` package depends on
`pygpcca`, which in turn requires `petsc`, `mamba install -c conda-forge cellrank`
cannot be solved on Windows and fails with an error such as
`petsc [...] does not exist (perhaps a missing channel)`.

On Windows, install CellRank with `pip install cellrank` instead. PETSc and SLEPc
are optional — without them CellRank falls back to a slower [SciPy]-based
eigensolver for computing macrostates and fate probabilities, which is fine for
small to medium datasets. To use PETSc and SLEPc on Windows, run CellRank inside
[WSL] and install via conda there.
```

## Development version

To install the latest development version from GitHub:

```bash
pip install git+https://github.com/theislab/cellrank.git@main
```

See {doc}`contributing` for setting up a full development environment.

[issue]: https://github.com/theislab/cellrank/issues/new
[PETSc]: https://petsc.org/
[SLEPc]: https://slepc.upv.es/
[SciPy]: https://scipy.org/
[WSL]: https://learn.microsoft.com/windows/wsl/install
