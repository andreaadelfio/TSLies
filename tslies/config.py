"""Central configuration for TSLies filesystem layout.

This module centralises the computation of frequently used directories and
timestamps so that other modules can simply import the resolved values instead of
calling :func:`tslies.paths.PathManager.get_base_dir` repeatedly.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings

from .paths import PathManager


RUN_TIMESTAMP: datetime = datetime.now()
DATE_FOLDER: str = RUN_TIMESTAMP.strftime("%Y-%m-%d")
TIME_FOLDER: str = RUN_TIMESTAMP.strftime("%H%M")

BASE_DIR: Optional[Path] = None
LOGS_DIR: Optional[Path] = None
DATA_DIR: Optional[Path] = None
RESULTS_DIR: Optional[Path] = None
BACKGROUND_PREDICTION_DIR: Optional[Path] = None
ANOMALIES_DIR: Optional[Path] = None
ANOMALIES_PLOTS_DIR: Optional[Path] = None
CATALOGS_DIR: Optional[Path] = None

_FALLBACK_WARNING_EMITTED = False


def _warn_if_fallback(base_dir: Optional[Path]) -> None:
    """Warn once when the home-directory fallback is being used."""
    if globals().get("_FALLBACK_WARNING_EMITTED"):
        return
    if base_dir is None:
        return
    if PathManager.get_base_dir(allow_home_fallback=False) is None:
        warnings.warn(
            "TSLies base directory is not configured. Outputs will be stored under ~/.tslies.",
            RuntimeWarning,
            stacklevel=3,
        )
        globals()["_FALLBACK_WARNING_EMITTED"] = True


def _cache_paths(base_dir: Optional[Path], *, ensure_exists: bool) -> None:
    module_globals = globals()
    module_globals["BASE_DIR"] = base_dir

    def _prepare(path: Optional[Path]) -> Optional[Path]:
        if ensure_exists and path is not None:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _join(*parts: str) -> Optional[Path]:
        if base_dir is None:
            return None
        return base_dir.joinpath(*parts)

    results_dir = _prepare(_join("results", DATE_FOLDER))
    module_globals["RESULTS_DIR"] = results_dir
    module_globals["LOGS_DIR"] = _prepare(_join("logs"))
    module_globals["DATA_DIR"] = _prepare(_join("data"))
    module_globals["BACKGROUND_PREDICTION_DIR"] = _prepare(
        results_dir.joinpath("background_prediction") if results_dir else None
    )
    module_globals["ANOMALIES_DIR"] = _prepare(
        results_dir.joinpath("anomalies") if results_dir else None
    )
    module_globals["ANOMALIES_PLOTS_DIR"] = _prepare(
        results_dir.joinpath("anomalies", TIME_FOLDER, "plots") if results_dir else None
    )
    module_globals["CATALOGS_DIR"] = _prepare(_join("catalogs"))


def _resolve_and_cache(*, resolver, allow_home_fallback: bool, ensure_exists: bool) -> Optional[Path]:
    base_dir = resolver(allow_home_fallback=allow_home_fallback, ensure_exists=ensure_exists)
    if base_dir is not None:
        base_dir = base_dir.resolve()
    _warn_if_fallback(base_dir)
    _cache_paths(base_dir, ensure_exists=ensure_exists)
    return base_dir


def refresh_paths(*, ensure_exists: bool = False) -> Optional[Path]:
    """Recompute cached paths from the current PathManager state."""
    return _resolve_and_cache(
        resolver=PathManager.get_base_dir,
        allow_home_fallback=True,
        ensure_exists=ensure_exists,
    )


def set_base_dir(root_dir: str | Path, *, ensure_exists: bool = True) -> Path:
    """Persist the directory that should be used for TSLies outputs and refresh caches."""
    path = PathManager.set_base_dir(root_dir).resolve()
    _warn_if_fallback(path)
    _cache_paths(path, ensure_exists=ensure_exists)
    return path


def get_base_dir(*, allow_home_fallback: bool = True, ensure_exists: bool = False) -> Optional[Path]:
    """Return the cached base directory, refreshing it when necessary."""
    return _resolve_and_cache(
        resolver=PathManager.get_base_dir,
        allow_home_fallback=allow_home_fallback,
        ensure_exists=ensure_exists,
    )


def require_base_dir(*, allow_home_fallback: bool = True, ensure_exists: bool = False) -> Path:
    """Return the base directory, raising if it is unavailable."""
    base_dir = _resolve_and_cache(
        resolver=PathManager.require_base_dir,
        allow_home_fallback=allow_home_fallback,
        ensure_exists=ensure_exists,
    )
    if base_dir is None:
        raise RuntimeError("Unable to resolve the TSLies base directory.")
    return base_dir


# Initialise the cached paths on module import.
refresh_paths()


__all__ = [
    "RUN_TIMESTAMP",
    "DATE_FOLDER",
    "TIME_FOLDER",
    "BASE_DIR",
    "LOGS_DIR",
    "DATA_DIR",
    "RESULTS_DIR",
    "BACKGROUND_PREDICTION_DIR",
    "ANOMALIES_DIR",
    "ANOMALIES_PLOTS_DIR",
    "CATALOGS_DIR",
    "refresh_paths",
    "set_base_dir",
    "get_base_dir",
    "require_base_dir",
]
