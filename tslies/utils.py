"""
Utils module for the ACNBkg project.
"""
import sys
import os
import pprint
import re
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from scipy import fftpack
import gc
from tqdm import tqdm

from .config import LOGS_DIR, DATA_DIR, get_base_dir, require_base_dir, require_existing_dir

BASE_DIR_PATH = require_base_dir()
if LOGS_DIR is None or DATA_DIR is None:
    raise RuntimeError(
        "TSLies directories are not initialised. Configure the base directory via "
        "tslies.config.set_base_dir(...) or set the TSLIES_DIR environment variable before "
        "importing tslies.utils."
    )

USER = os.environ.get('USER', os.environ.get('USERNAME', 'default_user'))
DIR = str(BASE_DIR_PATH)
LOGGING_FOLDER_PATH = str(LOGS_DIR)
LOGGING_FILE_NAME = f'{USER}.log'
DATA_FOLDER_NAME = str(DATA_DIR)



class Logger():
    """Utility wrapper around the standard library logging facilities."""
    def __init__(self, logger_name: str,
                 log_file_prefix: str = '',
                 log_file_name: str = LOGGING_FILE_NAME,
                 log_folder_path: str | os.PathLike[str] = LOGGING_FOLDER_PATH,
                 log_level: int = logging.DEBUG):
        """
        Initialise a configured ``logging.Logger`` instance.

        Parameters
        ----------
        - logger_name (str): Qualifier used to register the logger.
        - log_file_prefix (str): Optional prefix prepended to the log filename.
        - log_file_name (str): Target log filename, defaults to ``LOGGING_FILE_NAME``.
        - log_folder_path (str | os.PathLike[str]): Directory where log files are stored.
        - log_level (int): Logging level threshold, defaulting to ``logging.DEBUG``.

        Raises
        ------
        - OSError: Propagated if the log directory cannot be created.
        """
        self.log_level = log_level
        self.logger_name = logger_name
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(self.log_level)
        self.format = '%(asctime)s %(name)s [%(levelname)s]: %(pathname)s - %(funcName)s : %(message)s'
        self.formatter = logging.Formatter(self.format)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        base_dir = get_base_dir()
        folder_path = Path(log_folder_path)
        if not folder_path.is_absolute() and base_dir is None:
            warnings.warn(
                'TSLies base directory is not configured. Logging output will be sent to stderr.',
                RuntimeWarning,
                stacklevel=2,
            )
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(self.log_level)
            stream_handler.setFormatter(self.formatter)
            self.logger.addHandler(stream_handler)
            self.log_file_name = None
            return

        if not folder_path.is_absolute() and base_dir is not None:
            folder_path = base_dir / folder_path
        folder_path.mkdir(parents=True, exist_ok=True)
        if log_file_prefix:
            log_file_prefix = f'{log_file_prefix}_'
        log_name = Path(log_file_name).name
        self.log_file_name = folder_path / f'{log_file_prefix}{log_name}'
        if self.log_file_name is not None:
            self.file_handler = logging.FileHandler(self.log_file_name)
            self.file_handler.setLevel(self.log_level)
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)

    def get_logger(self):
        """
        Provide access to the configured ``logging.Logger`` instance.

        Parameters
        ----------
        - None

        Returns
        -------
        - logging.Logger: Logger ready for use.
        """
        return self.logger

def logger_decorator(logger):
    """
    Wrap a function so its execution is logged with start/stop markers.

    Parameters
    ----------
    - logger (logging.Logger): Logger used to emit progress messages.

    Returns
    -------
    - Callable: Decorator that adds logging to the wrapped function.

    Raises
    ------
    - None
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            class CustomLogRecord(logging.LogRecord):
                """Custom log record adapting pathname and function name metadata."""
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.pathname = sys.modules.get(func.__module__).__file__
                    self.funcName = func.__name__
            logging.setLogRecordFactory(CustomLogRecord)
            logger.info('START')
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logging.setLogRecordFactory(CustomLogRecord)
                logger.error(e)
                logger.debug('Args: %s', pprint.pformat(args))
                logger.debug('Kwargs: %s', pprint.pformat(kwargs))
                raise
            logging.setLogRecordFactory(CustomLogRecord)
            logger.info('END')
            return result
        return wrapper
    return decorator

class Time:
    """Class to handle time datatype
    """
    logger = Logger('Time').get_logger()

    ref_time = datetime(2001, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    fermi_launch_time = datetime(2008, 8, 7, 3, 35, 44, tzinfo=timezone.utc)

    @staticmethod
    def _unit_to_seconds_factor(unit: str) -> float:
        """
        Return the multiplier that converts elapsed values in ``unit`` to seconds.

        Parameters
        ----------
        - unit (str): Time unit identifier (e.g., ``'s'``, ``'ms'``, ``'h'``).

        Returns
        -------
        - float: Multiplicative factor to transform the specified unit into seconds.

        Raises
        ------
        - ValueError: If the supplied ``unit`` is not supported.
        """
        factors = {
            's': 1.0, 'sec': 1.0, 'second': 1.0, 'seconds': 1.0,
            'ms': 1e-3, 'millisecond': 1e-3, 'milliseconds': 1e-3,
            'us': 1e-6, 'microsecond': 1e-6, 'microseconds': 1e-6,
            'ns': 1e-9, 'nanosecond': 1e-9, 'nanoseconds': 1e-9,
            'm': 60.0, 'min': 60.0, 'minute': 60.0, 'minutes': 60.0,
            'h': 3600.0, 'hr': 3600.0, 'hour': 3600.0, 'hours': 3600.0,
            'd': 86400.0, 'day': 86400.0, 'days': 86400.0,
        }
        factor = factors.get(unit.lower())
        if factor is None:
            raise ValueError(f"Unsupported time unit '{unit}'.")
        return factor

    @staticmethod
    def from_elapsed_time_to_datetime(elapsed_list: list, unit: str = 's') -> list:
        """
        Convert elapsed times since ``Time.ref_time`` to timezone-aware datetimes.

        Parameters
        ----------
        - elapsed_list (list): Elapsed values referenced to ``Time.ref_time``.
        - unit (str): Unit of measure for the elapsed values (default: seconds).

        Returns
        -------
        - datetime_list (list of datetime): Corresponding datetime objects.

        Raises
        ------
        - ValueError: If ``unit`` is not recognised by ``_unit_to_seconds_factor``.
        """
        factor = Time._unit_to_seconds_factor(unit)
        return [Time.ref_time + timedelta(seconds=float(elapsed) * factor) for elapsed in elapsed_list]

    @staticmethod
    def from_datetime_to_elapsed_time(datetime_list: list, unit: str = 's') -> list:
        """
        Convert datetimes referenced to ``Time.ref_time`` into elapsed values.

        Parameters
        ----------
        - datetime_list (list): Iterable of datetime objects to convert.
        - unit (str): Desired unit of measure for the elapsed values (default: seconds).

        Returns
        -------
        - elapsed_list (list of float): Elapsed values in the requested unit.

        Raises
        ------
        - ValueError: If ``unit`` is not recognised by ``_unit_to_seconds_factor``.
        """
        factor = Time._unit_to_seconds_factor(unit)
        return [((dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)) - Time.ref_time).total_seconds() / factor for dt in datetime_list]

    @staticmethod
    def date2yday(x):
        """
        Convert Matplotlib datenums to elapsed seconds relative to ``ref_time``.

        Parameters
        ----------
        - x (Iterable[float]): Matplotlib datenum values to convert.

        Returns
        -------
        - list[float]: Elapsed seconds corresponding to each datenum.
        """
        y = []
        for dt in x:
            y.append((mdates.num2date(dt) - Time.ref_time).total_seconds())
        return y

    @staticmethod
    def yday2date(x):
        """
        Convert elapsed seconds into Matplotlib datenums referenced to ``ref_time``.

        Parameters
        ----------
        - x (Iterable[float]): Elapsed seconds to convert.

        Returns
        -------
        - list[float]: Matplotlib datenum representation for each elapsed value.
        """
        y = []
        for dt in x:
            print(dt)
            y.append((mdates.num2date(dt) + Time.ref_time).total_seconds())
        return y

    @staticmethod
    def from_elapsed_time_to_datetime_str(elapsed_list: list, unit: str = 's') -> list[str]:
        """
        Convert elapsed values to datetime strings referenced to ``Time.ref_time``.

        Parameters
        ----------
        - elapsed_list (list): Elapsed values referenced to ``Time.ref_time``.
        - unit (str): Unit of measure for the elapsed values (default: seconds).

        Returns
        -------
        - datetime_list (list of str): Corresponding datetime objects as strings.
        """
        factor = Time._unit_to_seconds_factor(unit)
        return [str(Time.ref_time + timedelta(seconds=float(elapsed) * factor)) for elapsed in elapsed_list]

    @staticmethod
    def remove_milliseconds_from_datetime(datetime_list: list) -> list[datetime]:
        """
        Remove the milliseconds from the datetime object.

        Parameters
        ----------
        - datetime_list (list): The datetime list to convert.

        Returns
        -------
        - datetime_list (list of datetime): The datetime object without milliseconds.
        """
        return [dt.replace(microsecond=0, tzinfo=timezone.utc) for dt in datetime_list]

    @staticmethod
    def get_week_from_datetime(datetime_dict: dict) -> list:
        """
        Get the week number from the datetime object.

        Parameters
        ----------
        - datetime_dict (list): The datetime list to convert.

        Returns
        -------
        - week_list (list of int): The week number corresponding to the datetime.
        """
        weeks_set = set()
        for dt1, dt2 in datetime_dict.values():
            weeks_set.add(((dt1 - Time.fermi_launch_time).days) // 7 + 10)
            weeks_set.add(((dt2 - Time.fermi_launch_time).days) // 7 + 10)
        print(weeks_set)
        return list(range(min(weeks_set), max(weeks_set) + 1))

    @staticmethod
    def get_datetime_from_week(week: int) -> tuple:
        """
        Get the datetime from the week number.

        Parameters
        ----------
        - week (int): The week number.

        Returns
        -------
        - datetime_tuple (tuple): The datetime tuple corresponding to the week.
        """
        start = Time.fermi_launch_time + timedelta(weeks=week - 10)
        end = start + timedelta(weeks=1)
        return start, end

class Data():
    """
    A class that provides utility functions for data manipulation.
    """
    logger = Logger('Data').get_logger()

    @logger_decorator(logger)
    @staticmethod
    def get_masked_dataframe(data, start, stop, column='datetime', reset_index=False) -> pd.DataFrame:
        """
        Returns the masked data within the specified time range.

        Parameters
        ----------
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
            data (DataFrame): The input data.
            column (str, optional): The column name representing the time. Defaults to 'datetime'.

        Returns
        -------
            DataFrame: The masked data within the specified time range.
        """
        mask = None
        if isinstance(start, int) and isinstance(stop, int) and column == 'index':
            if stop == -1: stop = len(data)
            mask = (data.index >= start) & (data.index <= stop)
        elif isinstance(start, str) and isinstance(stop, str):
            mask = (data[column] >= start) & (data[column] <= stop)
        masked_data = data[mask].reset_index(drop=True) if reset_index else data[mask]
        return pd.DataFrame(masked_data)

    @staticmethod
    def get_excluded_dataframes(start, stop, data, column='datetime'):
        """
        Returns the excluded dataframes within the specified time range.

        Parameters
        ----------
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
            data (DataFrame): The input data.
            column (str, optional): The column name representing the time. Defaults to 'datetime'.

        Returns
        -------
            list: The excluded dataframes within the specified time range.
        """
        mask = (data[column] < start) | (data[column] > stop)
        excluded_data = data[mask]
        return excluded_data

    @staticmethod
    def get_masked_data(start, stop, data, column='datetime'):
        """
        Returns the masked data within the specified time range.

        Parameters
        ----------
            start (float): The start time of the desired data range.
            stop (float): The stop time of the desired data range.
            data (DataFrame): The input data.
            column (str, optional): The column name representing the time. Defaults to 'datetime'.

        Returns
        -------
            dict: The masked data within the specified time range, with column names as keys
                  and lists of values as values.
        """
        mask = (data[column] >= start) & (data[column] <= stop)
        masked_data = data[mask]
        return {name: masked_data.field(name).tolist() for name in masked_data.names}

    @logger_decorator(logger)
    @staticmethod
    def filter_dataframe_with_run_times(initial_dataframe, run_times):
        """
        Returns the spacecraft dataframe filtered on runs times.

        Parameters
        ----------
            initial_dataframe (DataFrame): The initial spacecraft data.
            run_times (DataFrame): The dataframe containing run times.

        Returns
        -------
            DataFrame: The filtered spacecraft dataframe.
        """
        df = pd.DataFrame()
        for start, end in run_times.values():
            df = pd.concat([df, Data.get_masked_dataframe(data=initial_dataframe,
                                                          start=start, stop=end)],
                            ignore_index=True)
        return df

    @logger_decorator(logger)
    @staticmethod
    def convert_to_df(data_to_df: np.ndarray) -> pd.DataFrame:
        """
        Converts the data containing the spacecraft data into a pd.DataFrame.

        Parameters
        ----------
            data_to_df (ndarray): The data to convert.

        Returns
        -------
            DataFrame: The dataframe containing the spacecraft data.
        """
        name_data_dict = {name: data_to_df.field(name).tolist() for name in data_to_df.dtype.names}
        return pd.DataFrame(name_data_dict)

    @logger_decorator(logger)
    @staticmethod
    def merge_dfs(first_dataframe: pd.DataFrame, second_dataframe: pd.DataFrame,
                  on_column='datetime') -> pd.DataFrame:
        """
        Merges two dataframes based on a common column.

        Parameters
        ----------
            first_dataframe (DataFrame): The first dataframe.
            second_dataframe (DataFrame): The second dataframe.
            on_column (str, optional): The column to merge on. Defaults to 'datetime'.

        Returns
        -------
            DataFrame: The merged dataframe.
        """
        second_dataframe[on_column] = pd.to_datetime(second_dataframe[on_column], utc=True)
        first_dataframe.loc[:, on_column] = pd.to_datetime(first_dataframe[on_column], utc=True)
        return pd.merge(first_dataframe, second_dataframe, on=on_column, how='inner')

class File:
    """Class to handle files
    """
    logger = Logger('File').get_logger()

    @logger_decorator(logger)
    @staticmethod
    def write_df_on_file(df: pd.DataFrame, filename: str='', fmt: str='pk'):
        """Write the dataframe to a file.

        Parameters
        ----------
            df (pd.DataFrame): The dataframe to write.
            filename (str, optional): The name of the file to write the dataframe to.
                                      Defaults to ''.
            fmt (str, optional): The format to write the dataframe in.
                                 Can be:
                                    'csv' to write a .csv file;
                                    'pk' to write a .pk file;
                                    'both' to write in both formats.
                                 Defaults to 'pk'.

        Returns
        -------
            None
        """
        path, filename = os.path.split(filename)
        if fmt == 'csv':
            require_existing_dir(os.path.join(path, 'csv'))
            df.to_csv(os.path.join(path, 'csv', filename + '.csv'), index=False)
        elif fmt == 'pk':
            require_existing_dir(os.path.join(path, 'pk'))
            df.to_pickle(os.path.join(path, 'pk', filename + '.pk'))
        elif fmt == 'both':
            require_existing_dir(os.path.join(path, 'csv'))
            require_existing_dir(os.path.join(path, 'pk'))
            df.to_csv(os.path.join(path, 'csv', filename + '.csv'), index=False)
            df.to_pickle(os.path.join(path, 'pk', filename + '.pk'))

    @logger_decorator(logger)
    @staticmethod
    def read_df_from_file(filename=''):
        """
        Read the dataframe from a file.

        Parameters
        ----------
            filename (str, optional): The name of the file to read the dataframe from.
                                      Defaults to ''.

        Returns
        -------
            DataFrame: The dataframe read from the file.
        """
        path = os.path.join(DIR, f'{filename}.pk')
        if os.path.exists(path):
            return pd.read_pickle(path)
        return None

    # @logger_decorator(logger)
    # @staticmethod
    # def read_dfs_from_runs_pk_folder(folder_path='', add_smoothing=False, mode='mean', window=30, start=0, stop=-1, cols_list=None):
    #     """
    #     Read the dataframe from pickle files in a folder.

    #     Parameters
    #     ----------
    #         folder_path (str, optional): The name of the folder to read the dataframe from.
    #                                   Defaults to ''.

    #     Returns
    #     -------
    #         DataFrame: The dataframe read from the file.
    #     """
    #     folder_path = os.path.join(folder_path, 'pk')
    #     merged_dfs: pd.DataFrame = None
    #     if os.path.exists(folder_path):
    #         dir_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path)
    #                     if file.endswith('.pk')]
    #         dir_list = sorted(dir_list, key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group(0)))[start:stop]
    #         dfs = [pd.read_pickle(file)[cols_list] if cols_list else pd.read_pickle(file) for file in dir_list]
    #         dfs = [df[(df.loc[:, df.columns.difference(['MET'])] != 0).any(axis=1)] for df in dfs]
    #         if add_smoothing:
    #             if mode == 'mean':
    #                 dfs = [Maths.add_smoothing_with_mean(df, window=window) for df in dfs]
    #             elif mode == 'fft':
    #                 dfs = [Maths.add_smoothing_with_fft(df) for df in dfs]
    #         print(dfs[0].columns)
    #         merged_dfs = pd.concat(dfs, ignore_index=True).drop_duplicates('MET', ignore_index=True) # patch, trovare sorgente del bug
    #     return merged_dfs

    @logger_decorator(logger)
    def read_dfs_from_weekly_pk_folder(self, folder_path=DATA_FOLDER_NAME, custom_sorter=lambda x: int(x.split('w')[-1].split('.')[0]), cols_list=None, start=None, stop=None, y_cols=[], resample_skip=1):
        """
        Read the dataframe from pickle files in a folder.

        Parameters
        ----------
            folder_path (str, optional): The name of the folder to read the dataframe from.
                                      Defaults to ''.

        Returns
        -------
            DataFrame: The dataframe read from the file.
        """
        folder_path = os.path.join(folder_path, 'pk')
        self.logger.info('Reading from: %s.', folder_path)
        merged_dfs: pd.DataFrame = None
        if os.path.exists(folder_path):
            dir_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) 
                        if file.endswith('.pk') and start <= int(file.split('w')[-1].split('.')[0]) <= stop]
            self.logger.info('Found %d files matching criteria.', len(dir_list))
            if dir_list == []:
                self.logger.warning('No files found in %s with weeks between %s and %s.', folder_path, start, stop)
                return None
            dir_list = sorted(dir_list, key=custom_sorter)
            dfs = []
            for file in tqdm(dir_list, desc='Reading dfs from files'):
                tmp_df = pd.read_pickle(file)[cols_list].iloc[::resample_skip] if cols_list else pd.read_pickle(file).iloc[::resample_skip]
                if y_cols:
                    tmp_df = tmp_df[(tmp_df[y_cols] != 0).all(axis=1)]
                dfs.append(tmp_df)
                gc.collect()
            merged_dfs = pd.concat(dfs, ignore_index=True)
        else:
            self.logger.error('Folder %s does not exist.', folder_path)
        return merged_dfs

    @logger_decorator(logger)
    @staticmethod
    def read_dfs_from_csv_folder(folder_path='', custom_sorter=lambda x: int(x.split('_w')[-1].split('.')[0])):
        """
        Read the dataframe from csv files in a folder.

        Parameters
        ----------
            folder_path (str, optional): The name of the folder to read the dataframe from.
                                      Defaults to ''.

        Returns
        -------
            DataFrame: The dataframe read from the file.
        """
        if folder_path != '':
            folder_path = os.path.join(DATA_FOLDER_NAME, folder_path)
        folder_path = os.path.join(folder_path, 'csv')
        if os.path.exists(folder_path):
            dir_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
            dir_list = sorted(dir_list, key=custom_sorter)
            dfs = [pd.read_csv(file) for file in dir_list]
            merged_dfs = pd.concat(dfs, ignore_index=True).drop_duplicates('MET', ignore_index=True) # patch, trovare sorgente del bug
        return merged_dfs

    @logger_decorator(logger)
    @staticmethod
    def write_on_file(data: dict, filename: str):
        """Writes data on file

        Parameters
        ----------
            data (dict): disctionary containing data
            filename (str): name of the file
        """
        with open(filename, 'w', encoding='utf-8') as file:
            for key, value in data.items():
                file.write(f'{key}: {value}\n')

    @logger_decorator(logger)
    @staticmethod
    def check_integrity_runs_pk_folder(folder_path=''):
        """
        Checks the integrity of the dataframe from pickle files in a folder.

        Parameters
        ----------
            folder_path (str, optional): The name of the folder to read the dataframe from.
                                      Defaults to ''.
        """
        print('Checking integrity...')
        folder_path = os.path.join(folder_path, 'pk')
        if os.path.exists(folder_path):
            dir_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) 
                        if file.endswith('.pk')]
            dir_list = sorted(dir_list, key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group(0)))
            for file in dir_list:
                df = pd.read_pickle(file)
                initial_len = len(df)
                df = df[(df.loc[:, df.columns.difference(['MET', 'datetime'])] != 0).any(axis=1)]
                if len(df) != initial_len:
                    print(initial_len, len(df), os.path.basename(file))
                    # os.remove(file)

class Maths:
    """Class to handle maths operations
    """
    logger = Logger('Maths').get_logger()
    
    @logger_decorator(logger)
    @staticmethod
    def add_smoothing_with_mean(tile_signal, window=30):
        """This function adds the smoothed histograms to the signal dataframe."""
        window = window if len(tile_signal) > window else len(tile_signal)
        for h_name in set(tile_signal.keys()) - {'MET', 'datetime'}:
            histc = tile_signal[h_name].to_list()
            filtered_sig1 = np.convolve(histc, np.ones(window)/window, mode='same')
            for i in range(window//2):
                filtered_sig1[i] = np.mean(histc[:i+window//2])
            for i in range(len(histc) - window//2, len(histc)):
                filtered_sig1[i] = np.mean(histc[i-window//2:])
            tile_signal[f'{h_name}_smooth'] = filtered_sig1
        return tile_signal

    @logger_decorator(logger)
    @staticmethod
    def add_smoothing_with_fft(tile_signal):
        """This function adds the smoothed histograms to the signal dataframe."""
        histx = tile_signal['MET']
        time_step = histx[2] - histx[1]
        nyquist_freq = 0.5 / time_step
        for h_name in set(tile_signal.keys()) - {'MET', 'datetime'}:
            histc = tile_signal[h_name].to_list()
            freq_cut1 = np.mean(histc) * nyquist_freq / 3
            sig_fft = fftpack.fft(histc)
            sample_freq = fftpack.fftfreq(len(histc), d=time_step)
            low_freq_fft1  = sig_fft.copy()
            low_freq_fft1[np.abs(sample_freq) > freq_cut1] = 0
            filtered_sig1  = np.array(fftpack.ifft(low_freq_fft1)).real
            tile_signal[f'{h_name}_smooth'] = filtered_sig1
        return tile_signal

if __name__ == '__main__':
    print('This is a utils module, not a main program.')