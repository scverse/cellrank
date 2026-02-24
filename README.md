[![PyPI](https://img.shields.io/pypi/v/cellrank.svg)](https://pypi.org/project/cellrank)
[![Downloads](https://static.pepy.tech/badge/cellrank)](https://pepy.tech/project/cellrank)
[![CI](https://img.shields.io/github/actions/workflow/status/theislab/cellrank/test.yaml?branch=main)](https://github.com/theislab/cellrank/actions)
[![Docs](https://img.shields.io/readthedocs/cellrank)](https://cellrank.readthedocs.io/)
[![Codecov](https://codecov.io/gh/theislab/cellrank/branch/main/graph/badge.svg)](https://codecov.io/gh/theislab/cellrank)
[![Discourse](https://img.shields.io/discourse/posts?color=yellow&logo=discourse&server=https%3A%2F%2Fdiscourse.scverse.org)](https://discourse.scverse.org/c/ecosystem/cellrank/)
[![Python Version](https://img.shields.io/pypi/pyversions/cellrank)](https://pypi.org/project/cellrank)

# CellRank 2: Unified fate mapping in multiview single-cell data

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/img/dark_mode_overview.png">
  <img alt="CellRank overview" src="docs/_static/img/light_mode_overview.png" width="600px" align="center">
</picture>

**CellRank** is a modular framework to study cellular dynamics based on Markov state modeling of
multi-view single-cell data. See our [documentation], and the [CellRank 1] and [CellRank 2] manuscripts to learn more.

> [!IMPORTANT]
> Please refer to [our citation guide](https://cellrank.readthedocs.io/en/latest/about/cite.html) to cite our software correctly.

CellRank scales to large cell numbers, is fully compatible with the [scverse] ecosystem, and is easy to use.
In the backend, it is powered by [pyGPCCA] ([Reuter et al. (2018)]). Feel
free to open an [issue] if you encounter a bug, need our help, or just want to make a comment/suggestion.

## CellRank's key applications

- Estimate differentiation direction based on a varied number of biological priors, including RNA velocity
  ([La Manno et al. (2018)], [Bergen et al. (2020)]), any pseudotime or developmental potential,
  experimental time points, metabolic labels, and more.
- Compute initial, terminal and intermediate macrostates.
- Infer fate probabilities and driver genes.
- Visualize and cluster gene expression trends.
- ... and much more, check out our [documentation].

## Installation

```bash
pip install cellrank
```

See the [installation guide](https://cellrank.readthedocs.io/en/latest/installation.html) for more options.

[La Manno et al. (2018)]: https://doi.org/10.1038/s41586-018-0414-6
[Bergen et al. (2020)]: https://doi.org/10.1038/s41587-020-0591-3
[Reuter et al. (2018)]: https://doi.org/10.1021/acs.jctc.8b00079
[scverse]: https://scverse.org/
[pyGPCCA]: https://github.com/msmdev/pyGPCCA
[CellRank 1]: https://www.nature.com/articles/s41592-021-01346-6
[CellRank 2]: https://doi.org/10.1038/s41592-024-02303-9
[documentation]: https://cellrank.org
[issue]: https://github.com/theislab/cellrank/issues/new/choose
