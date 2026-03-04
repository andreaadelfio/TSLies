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


SESSION_TIMESTAMP: datetime = datetime.now()
DATE_FOLDER: str = SESSION_TIMESTAMP.strftime("%Y-%m-%d")
TIME_FOLDER: str = SESSION_TIMESTAMP.strftime("%H%M")

BASE_DIR: Optional[Path] = None
LOGS_DIR: Optional[Path] = None
DATA_DIR: Optional[Path] = None
RESULTS_DIR: Optional[Path] = None
BACKGROUND_PREDICTION_DIR: Optional[Path] = None
ANOMALIES_DIR: Optional[Path] = None
ANOMALIES_TIME_DIR: Optional[Path] = None
ANOMALIES_PLOTS_DIR: Optional[Path] = None
CATALOGS_DIR: Optional[Path] = None

_FALLBACK_WARNING_EMITTED = False


def _warn_if_fallback(base_dir: Optional[Path]) -> None:
    """
    Emit a warning the first time the home-directory fallback is triggered.

    Parameters
    ----------
    - base_dir (Optional[Path]): Base directory resolved by the path manager.

    Returns
    -------
    - None: This function only produces side effects.

    Raises
    ------
    - None
    """
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
    """
    Store derived filesystem paths in module-level globals.

    Parameters
    ----------
    - base_dir (Optional[Path]): Root directory from which relative paths are built.
    - ensure_exists (bool): Whether to create missing directories on disk.

    Returns
    -------
    - None: Paths are cached via module globals.

    Raises
    ------
    - OSError: Propagated if directory creation fails when ``ensure_exists`` is True.
    """
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
    module_globals["ANOMALIES_TIME_DIR"] = _prepare(
        results_dir.joinpath("anomalies", TIME_FOLDER) if results_dir else None
    )
    module_globals["ANOMALIES_PLOTS_DIR"] = _prepare(
        results_dir.joinpath("anomalies", TIME_FOLDER, "plots") if results_dir else None
    )
    module_globals["CATALOGS_DIR"] = _prepare(_join("catalogs"))


def _resolve_and_cache(*, resolver, allow_home_fallback: bool, ensure_exists: bool) -> Optional[Path]:
    """
    Resolve the base directory and refresh cached paths in a single call.

    Parameters
    ----------
    - resolver (Callable[..., Optional[Path]]): Function returning the base directory.
    - allow_home_fallback (bool): Whether resolving may fall back to ``~/.tslies``.
    - ensure_exists (bool): Create directories when they are missing.

    Returns
    -------
    - Optional[Path]: Absolute base directory, or ``None`` when resolution fails.

    Raises
    ------
    - OSError: Raised if directory creation fails while ensuring existence.
    """
    base_dir = resolver(allow_home_fallback=allow_home_fallback, ensure_exists=ensure_exists)
    if base_dir is not None:
        base_dir = base_dir.resolve()
    _warn_if_fallback(base_dir)
    _cache_paths(base_dir, ensure_exists=ensure_exists)
    return base_dir


def refresh_paths(*, ensure_exists: bool = False) -> Optional[Path]:
    """
    Recompute and cache directories using the current configuration.

    Parameters
    ----------
    - ensure_exists (bool): When True, create the resolved directories on disk.

    Returns
    -------
    - Optional[Path]: The resolved base directory, or ``None`` if unavailable.

    Raises
    ------
    - OSError: Propagated if directory creation fails when ``ensure_exists`` is True.
    """
    return _resolve_and_cache(
        resolver=PathManager.get_base_dir,
        allow_home_fallback=True,
        ensure_exists=ensure_exists,
    )


def set_base_dir(root_dir: str | Path, *, ensure_exists: bool = True) -> Path:
    """
    Persist the provided root directory and refresh cached filesystem paths.

    Parameters
    ----------
    - root_dir (str | Path): Absolute or relative path to use as TSLies root.
    - ensure_exists (bool): Create the directory tree if it does not exist.

    Returns
    -------
    - Path: Resolved absolute path of the persisted root directory.

    Raises
    ------
    - OSError: Raised if creating the directory structure fails.
    """
    path = PathManager.set_base_dir(root_dir).resolve()
    _warn_if_fallback(path)
    _cache_paths(path, ensure_exists=ensure_exists)
    return path


def get_base_dir(*, allow_home_fallback: bool = True, ensure_exists: bool = False) -> Optional[Path]:
    """
    Retrieve the cached base directory, resolving it on demand when absent.

    Parameters
    ----------
    - allow_home_fallback (bool): Permit falling back to ``~/.tslies`` when unset.
    - ensure_exists (bool): Create the directory tree if it is missing.

    Returns
    -------
    - Optional[Path]: Resolved base directory, or ``None`` if resolution fails.

    Raises
    ------
    - OSError: Propagated if directory creation fails while ensuring existence.
    """
    return _resolve_and_cache(
        resolver=PathManager.get_base_dir,
        allow_home_fallback=allow_home_fallback,
        ensure_exists=ensure_exists,
    )


def require_base_dir(*, allow_home_fallback: bool = True, ensure_exists: bool = False) -> Path:
    """
    Resolve the base directory and raise an error when it cannot be obtained.

    Parameters
    ----------
    - allow_home_fallback (bool): Permit fallback to ``~/.tslies`` when unset.
    - ensure_exists (bool): Create the directory tree if it does not exist.

    Returns
    -------
    - Path: Absolute base directory ensured to exist.

    Raises
    ------
    - RuntimeError: If the base directory cannot be resolved.
    - OSError: Propagated if directory creation fails while ensuring existence.
    """
    base_dir = _resolve_and_cache(
        resolver=PathManager.require_base_dir,
        allow_home_fallback=allow_home_fallback,
        ensure_exists=ensure_exists,
    )
    if base_dir is None:
        raise RuntimeError("Unable to resolve the TSLies base directory.")
    return base_dir

def require_existing_dir(path: Path) -> Path:
    """
    Ensure an existing directory is available, creating it when necessary.

    Parameters
    ----------
    - path (Path): Directory path that must exist.

    Returns
    -------
    - Path: Resolved path guaranteed to exist on disk.

    Raises
    ------
    - OSError: Raised if the directory cannot be created.
    """
    path = Path(path).resolve()
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path

# Initialise the cached paths on module import.
refresh_paths()


__all__ = [
    "SESSION_TIMESTAMP",
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
