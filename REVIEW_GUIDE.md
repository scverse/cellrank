# CellRank Review Guide

Agent-neutral PR review playbook. Written for **review agents running on GitHub** — use the imperative voice.

**Scope: review only.** Produce comments and suggestions. Do **not** push commits, modify files, or apply fixes. Flag issues and suggest diffs in comments; leave the edits to the author.

Architecture, invariants, and commands live in `AGENTS.md`. Do not restate them here — link.

## Workflow

1. Read the PR body.
2. Check CI (`gh pr checks <num>`, `gh run view <run-id> --log-failed`) and investigate failures before commenting.
3. Map changed paths to tests (see below) and check whether the change touches an invariant from `AGENTS.md`.
4. Prioritize behavioral regressions, serialization compatibility, and numerical correctness over style.

## High-Risk Areas

Pointers only — see `AGENTS.md` for the actual invariants.

- **Kernel composition** (`src/cellrank/kernels/_base_kernel.py`): weight normalization in `KernelAdd`, direction flipping in bidirectional kernels. Silent regressions possible.
- **AnnData serialization** (`write_to_adata` / `from_adata`, estimator shadow AnnData): breaks saved analyses and downstream notebooks if the round-trip changes.
- **`RealTimeKernel`** (`src/cellrank/kernels/_real_time_kernel.py`): assembling per-timepoint couplings (`from_moscot` / `from_wot`) into a global block transition matrix, with configurable self-transitions — the most complex path.
- **Spectral estimators** (`src/cellrank/estimators/mixins/`): Schur, eigen, fate-probability, and lineage-driver mixins are correctness-sensitive.
- **`Lineage`** (`src/cellrank/_utils/_lineage.py`): ndarray subclass with public slicing/coloring semantics.
- **Optional-dep guards**: new `jax` / `moscot` / `petsc4py` / `slepc4py` / `rpy2` / `wot` / `scvelo` / `adjusttext` usage must route through the existing guards and not leak into top-level imports.
- **Public API surface**: new re-exports in `src/cellrank/__init__.py` or in `kernels` / `estimators` / `models` / `pl` / `datasets`.

## Changed-Path Test Lookup

Flat layout — `tests/test_X.py` keyed by component, not a mirror of `src/`.

| Changed path | Primary tests |
|--------------|---------------|
| `src/cellrank/kernels/` | `tests/test_kernels.py`; add `tests/test_pipeline.py` for composition |
| `src/cellrank/estimators/` | `tests/test_gpcca.py`, `tests/test_cflare.py`, `tests/test_lineage_drivers.py`, `tests/test_pipeline.py` |
| `src/cellrank/models/` | `tests/test_model.py` |
| `src/cellrank/pl/` | `tests/test_plotting.py` |
| `src/cellrank/_utils/_lineage.py` | `tests/test_lineage.py` |
| `src/cellrank/_utils/_linear_solver.py` | `tests/test_linear_solver.py` |
| Fixture changes | `tests/conftest.py` |

## Testing

- **New code** should be covered. Reuse fixtures from `tests/conftest.py`; prefer `pytest.mark.parametrize`; favor few meaningful tests over many redundant ones.
- **Failing CI** is not to be waved through. Distinguish critical regressions from flakes or expected skips (PETSc/SLEPc and R-backed tests skip when the extras are missing); escalate critical failures.
- **Modified tests** — scrutinize *how*. Relaxed tolerances, removed assertions, deleted cases, or loosened matrices are red flags. Require explicit justification in the PR body.

## Documentation Impact

Behavior or API changes often touch docs in multiple places. Point at the **owning file**, don't duplicate content in the review.

- Public symbol / API changes → `docs/api/` + autosummary, `README.md` quickstart.
- Design-principle or architectural changes → `docs/about/index.md`.
- Contributor workflow or env changes → `docs/contributing.md`, `docs/installation.md`.
- Tutorials under `docs/notebooks/tutorials/` → flag stale imports or outputs.
- Release-affecting changes → `docs/release_notes.md`.
- Invariants / commands / reference table → `AGENTS.md`.
- Review workflow / risk areas / test lookup → this file.
- `CLAUDE.md` and `.github/copilot-instructions.md` should stay thin pointers — flag any PR that re-adds content here.

## Checklist

- Invariants in `AGENTS.md` preserved?
- CI green (or failures investigated)?
- Test coverage adequate and not silently weakened?
- AnnData serialization round-trip (`write_to_adata` / `from_adata`, estimator `to_adata`) intact — or explicitly called out in the PR body?
- Public API surface unchanged — or intentional and called out?
- Affected human- and agent-facing docs updated?
- PR scope tight, no unrelated bundling?

## PR Metadata

This repo uses `.github/PULL_REQUEST_TEMPLATE.md`. Treat its sections (Changes / Bug fixes / New / Related issues) as the preferred summary surface.
