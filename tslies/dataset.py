"""
Dataset ingestion helpers for TSLies pipelines.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .config import DATA_DIR, get_base_dir
from .utils import Time, Logger, logger_decorator, File

class DatasetReader():
    """Reader utility that assembles run-level data frames from pickled tiles."""
    logger = Logger('DatasetReader').get_logger()

    @logger_decorator(logger)
    def __init__(
        self,
        cols : list[str],
        data_dir: Optional[str | Path] = None,
        start: int = 0,
        end: int = -1,
    ):
        """
        Configure a reader capable of stitching together multiple run segments.

        Parameters
        ----------
        - cols (list[str]): Column names to extract from the stored tiles.
        - data_dir (Optional[str | Path]): Root directory containing run folders. Defaults to ``tslies.config.DATA_DIR``.
        - start (int): Inclusive index of the first run folder to load.
        - end (int): Inclusive index of the last run folder to load, or ``-1`` for all remaining runs.

        Raises
        ------
        - RuntimeError: If the base directory cannot be resolved for relative paths.
        """
        self.start = start
        self.end = end if end != -1 else None

        if data_dir is None:
            resolved_dir = DATA_DIR
            if resolved_dir is None:
                raise RuntimeError(
                    "TSLies base directory is not configured. Call tslies.config.set_base_dir(...) "
                    "or set the TSLIES_DIR environment variable before accessing the dataset."
                )
        else:
            resolved_dir = Path(data_dir)
            if not resolved_dir.is_absolute():
                base_dir = get_base_dir()
                if base_dir is None:
                    raise RuntimeError(
                        "TSLies base directory is not configured, so relative data paths cannot be resolved."
                    )
                resolved_dir = base_dir / resolved_dir
        resolved_dir = Path(resolved_dir)
        self.data_dir = str(resolved_dir)
        self.cols = cols
        self.runs_times = {}
        self.runs_dict = {}

    @logger_decorator(logger)
    def get_runs_times(self):
        """
        Retrieve cached temporal coverage for the loaded runs.

        Parameters
        ----------
        - None

        Returns
        -------
        - dict: Mapping run identifiers to ``(start_datetime, end_datetime)`` tuples.

        """
        return self.runs_times

    @logger_decorator(logger)
    def get_signal_df_from_dataset(self) -> pd.DataFrame:
        """
        Load tile signals, concatenate them, and enrich with datetime information.

        Parameters
        ----------
        - None

        Returns
        -------
        - pandas.DataFrame: Combined dataset containing the requested columns and ``datetime``.

        Raises
        ------
        - FileNotFoundError: Propagated if one of the run pickle files is unavailable.
        - ValueError: Propagated if read data cannot be concatenated.
        """
        dataset_df = File.read_dfs_from_runs_pk_folder(folder_path=self.data_dir, add_smoothing=True, mode='mean', window=35, start=self.start, stop=self.end, cols_list=self.cols + ['MET'])
        dataset_df['datetime'] = np.array(Time.from_elapsed_time_to_datetime(dataset_df['MET'] - 1))
        self.runs_times['dataset'] = (dataset_df['datetime'][0], dataset_df['datetime'].iloc[-1])
        return dataset_df

if __name__ == '__main__':
    cr = DatasetReader(cols=['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg'], start=0, end=-1)
    tile_signal_df = cr.get_signal_df_from_dataset()
    print(tile_signal_df.head())