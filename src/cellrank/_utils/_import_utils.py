"""Utilities for optional dependency checking."""

import importlib

__all__ = ["_check_module_importable"]


def _check_module_importable(module: str, *, extra: str | None = None) -> None:
    """Raise :class:`ModuleNotFoundError` with install instructions if *module* is missing.

    Parameters
    ----------
    module
        Fully qualified module name, e.g. ``"moscot"``.
    extra
        pip extra to suggest, e.g. ``"moscot"`` →
        ``pip install cellrank[moscot]``.  If :obj:`None`, suggests
        ``pip install <module>`` instead.
    """
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        install = f"pip install cellrank[{extra}]" if extra else f"pip install {module}"
        raise ModuleNotFoundError(f"Unable to import `{module}`. Install it with: `{install}`") from None
