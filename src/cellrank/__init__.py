from importlib import metadata

from cellrank import datasets, estimators, kernels, models, pl
from cellrank._settings import CellRankConfig, _setup_logger, settings
from cellrank._utils._lineage import Lineage

_setup_logger()

__all__ = ["datasets", "estimators", "kernels", "models", "pl", "CellRankConfig", "Lineage", "settings"]

try:
    md = metadata.metadata(__name__)
    __version__ = md.get("version", "")
    __author__ = md.get("Author", "")
    __maintainer__ = md.get("Maintainer-email", "")
except ImportError:
    md = None

del metadata, md
