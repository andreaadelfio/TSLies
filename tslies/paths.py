"""Utilities for resolving the base directory used by TSLies."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


class PathManager:
    """Central helpers to resolve the base directory for filesystem access."""

    _explicit_dir: Optional[Path] = None
    _main_dir: Optional[Path] = None

    @staticmethod
    def _normalise_path(value: str | os.PathLike[str] | None) -> Optional[Path]:
        """
        Convert a user-provided path value into a resolved ``Path`` instance.

        Parameters
        ----------
        - value (str | os.PathLike[str] | None): Path-like value to normalise.

        Returns
        -------
        - Optional[Path]: Absolute path when ``value`` is truthy, otherwise ``None``.
        """
        if not value:
            return None
        return Path(value).expanduser().resolve()

    @classmethod
    def _main_module_dir(self) -> Optional[Path]:
        """
        Resolve the directory containing the executing ``__main__`` module.

        Parameters
        ----------
        - None

        Returns
        -------
        - Optional[Path]: Absolute parent directory of the ``__main__`` module.
        """
        if self._main_dir is not None:
            return self._main_dir
        main_module = sys.modules.get("__main__")
        main_file = getattr(main_module, "__file__", None)
        if not main_file:
            return None
        self._main_dir = Path(main_file).expanduser().resolve().parent
        return self._main_dir

    @classmethod
    def set_base_dir(self, root_dir: str | os.PathLike[str]) -> Path:
        """
        Persist the base directory to use for all filesystem operations.

        Parameters
        ----------
        - root_dir (str | os.PathLike[str]): Directory that should become the base.

        Returns
        -------
        - Path: Resolved absolute path of the persisted base directory.

        Raises
        ------
        - ValueError: If ``root_dir`` cannot be normalised into a valid path.
        """
        path = self._normalise_path(root_dir)
        if path is None:
            raise ValueError("Invalid base directory provided to set_base_dir().")
        path.mkdir(parents=True, exist_ok=True)
        self._explicit_dir = path
        os.environ["TSLIES_DIR"] = str(path)
        return path

    @classmethod
    def clear_base_dir(self) -> None:
        """
        Reset the explicit base directory so auto-detection is used again.

        Parameters
        ----------
        - None

        Returns
        -------
        - None: The explicit directory cache is cleared in place.
        """
        self._explicit_dir = None

    @classmethod
    def get_base_dir(self, *, allow_home_fallback: bool = True, ensure_exists: bool = False) -> Optional[Path]:
        """
        Return the best-effort base directory or ``None`` when it cannot be resolved.

        Parameters
        ----------
        - allow_home_fallback (bool): Permit falling back to ``~/.tslies``.
        - ensure_exists (bool): Create missing directories before returning.

        Returns
        -------
        - Optional[Path]: Resolved base directory, or ``None`` when unavailable.

        Raises
        ------
        - OSError: Propagated if directory creation fails when ensuring existence.
        """
        candidates = (
            self._explicit_dir,
            self._normalise_path(os.environ.get("TSLIES_DIR")),
            self._main_module_dir(),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            if ensure_exists:
                candidate.mkdir(parents=True, exist_ok=True)
            return candidate

        if allow_home_fallback:
            fallback = Path.home().joinpath(".tslies")
            if ensure_exists:
                fallback.mkdir(parents=True, exist_ok=True)
            return fallback
        return None

    @classmethod
    def require_base_dir(self, *, allow_home_fallback: bool = False, ensure_exists: bool = False) -> Path:
        """
        Return the base directory or raise ``RuntimeError`` when it is unavailable.

        Parameters
        ----------
        - allow_home_fallback (bool): Permit ``~/.tslies`` fallback when True.
        - ensure_exists (bool): Create missing directories before returning.

        Returns
        -------
        - Path: Absolute base directory ensured to exist.

        Raises
        ------
        - RuntimeError: If the base directory cannot be resolved.
        - OSError: Propagated when directory creation fails.
        """
        base_dir = self.get_base_dir(allow_home_fallback=allow_home_fallback, ensure_exists=ensure_exists)
        if base_dir is None:
            raise RuntimeError(
                "TSLies base directory is not configured. Call tslies.config.set_base_dir(...) "
                "or set the TSLIES_DIR environment variable."
            )
        return base_dir

    @classmethod
    def resolve_subpath(self, *parts: str, allow_home_fallback: bool = True,
                        ensure_exists: bool = False, critical: bool = False) -> Optional[Path]:
        """
        Resolve a sub-path under the configured base directory.

        Parameters
        ----------
        - parts (tuple[str, ...]): Relative path components to append.
        - allow_home_fallback (bool): Allow fallback to ``~/.tslies`` unless critical.
        - ensure_exists (bool): Create the sub-path when it is missing.
        - critical (bool): When True, raise if the base directory is unresolved.

        Returns
        -------
        - Optional[Path]: Resolved sub-path or ``None`` when unresolved and non-critical.

        Raises
        ------
        - RuntimeError: If the base directory is unavailable while ``critical`` is True.
        - OSError: Propagated if directory creation fails while ensuring existence.
        """
        base_dir = self.get_base_dir(
            allow_home_fallback=allow_home_fallback and not critical,
            ensure_exists=ensure_exists and not critical,
        )
        if base_dir is None:
            if critical:
                raise RuntimeError(
                    "TSLies base directory is not configured. Call tslies.config.set_base_dir(...) "
                    "or set the TSLIES_DIR environment variable."
                )
            return None
        path = base_dir.joinpath(*parts)
        if ensure_exists:
            path.mkdir(parents=True, exist_ok=True)
        return path


__all__ = [
    "PathManager",
]
