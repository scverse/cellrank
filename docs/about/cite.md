# Citing CellRank

If you find CellRank useful for your research, please consider citing our work as follows: If you are using
CellRank's {class}`~cellrank.kernels.VelocityKernel` with classical RNA velocity, cite {cite}`lange:22` as:

```bibtex
@article{lange:22,
    author    = {Lange, Marius and Bergen, Volker and Klein, Michal and Setty, Manu and Reuter, Bernhard and Bakhti, Mostafa and Lickert, Heiko and Ansari, Meshal and Schniering, Janine and Schiller, Herbert B. and Pe'er, Dana and Theis, Fabian J.},
    publisher = {Nature Publishing Group},
    doi       = {10.1038/s41592-021-01346-6},
    journal   = {Nat. Methods},
    title     = {CellRank for directed single-cell fate mapping},
    year      = {2022},
}
```

If you are using the {class}`~cellrank.kernels.PseudotimeKernel`, {class}`~cellrank.kernels.CytoTRACEKernel`, {class}`~cellrank.kernels.RealTimeKernel`, or the {class}`~cellrank.kernels.VelocityKernel` with velocities inferred
from metabolic labeling data using the CellRank 2 approach, cite {cite}`weiler:24` as:

```bibtex
@article{weiler:24,
    author    = {Weiler, Philipp and Lange, Marius and Klein, Michal and Pe'er, Dana and Theis, Fabian},
    publisher = {Springer Science and Business Media LLC},
    url       = {https://doi.org/10.1038/s41592-024-02303-9},
    doi       = {10.1038/s41592-024-02303-9},
    issn      = {1548-7105},
    journal   = {Nature Methods},
    month     = jun,
    number    = {7},
    pages     = {1196--1205},
    title     = {CellRank 2: unified fate mapping in multiview single-cell data},
    volume    = {21},
    year      = {2024},
}
```

In addition, if you use the {class}`~cellrank.estimators.GPCCA` estimator to compute initial, terminal or intermediate
states, you are using the [pyGPCCA package](https://github.com/msmdev/pyGPCCA) {cite}`reuter:22` under the hood,
which implements the Generalized Perron Cluster Cluster Analysis (GPCCA) algorithm. Thus, additionally to CellRank,
please cite GPCCA {cite}`reuter:19` as:

```bibtex
@article{reuter:19,
  author  = {Reuter, Bernhard and Fackeldey, Konstantin and Weber, Marcus},
  doi     = {10.1063/1.5064530},
  journal = {The Journal of Chemical Physics},
  number  = {17},
  pages   = {174103},
  title   = {Generalized Markov modeling of nonreversible molecular kinetics},
  volume  = {150},
  year    = {2019},
}
```

## CellRank papers

### CellRank 1 — *Nature Methods*, 2022

The original CellRank publication introduced a framework for directed single-cell fate mapping
based on RNA velocity and gene expression similarity. It combines these signals via a Markov chain
formulation and uses the GPCCA algorithm to identify initial and terminal states, compute fate
probabilities, and infer driver genes.

Lange, M. *et al.* [CellRank for directed single-cell fate mapping.](https://doi.org/10.1038/s41592-021-01346-6) *Nat. Methods* **19**, 159–170 (2022).

See the {doc}`RNA velocity tutorial <../notebooks/tutorials/kernels/200_rna_velocity>` to get started
with the VelocityKernel.

### CellRank 2 — *Nature Methods*, 2024

CellRank 2 generalizes the framework beyond RNA velocity, unifying multiple biological signals
— pseudotime, developmental potential (CytoTRACE), real-time experimental time points via optimal
transport, and more — into a single coherent API. It introduces the PseudotimeKernel,
CytoTRACEKernel, and RealTimeKernel alongside the original VelocityKernel.

Weiler, P. *et al.* [CellRank 2: unified fate mapping in multiview single-cell data.](https://doi.org/10.1038/s41592-024-02303-9) *Nat. Methods* **21**, 1196–1205 (2024).

See {doc}`../notebooks/tutorials/kernels/300_pseudotime`,
{doc}`../notebooks/tutorials/kernels/400_cytotrace`, and
{doc}`../notebooks/tutorials/kernels/500_real_time` for kernel-specific tutorials.

### CellRank Protocol — *Nature Protocols*, 2025

A step-by-step practical guide for applying CellRank to single-cell data. The protocol covers
the full workflow from data preprocessing to fate probability computation and downstream analysis,
making it the ideal starting point for new users.

Weiler, P. *et al.* [Unified fate mapping in multiview single-cell data.](https://doi.org/10.1038/s41596-024-01108-4) *Nat. Protoc.* (2025).

See {doc}`../notebooks/tutorials/general/100_getting_started` for the introductory tutorial.
