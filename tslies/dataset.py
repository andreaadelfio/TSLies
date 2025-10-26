'''
This module contains the class to manage the acd dataset.
'''
import os
import numpy as np
import pandas as pd
from .utils import Time, Logger, logger_decorator, File

DIR = os.environ.get('TSLIES_DIR')
DATA_LATACD_PROCESSED_FOLDER_NAME = os.path.join(DIR, 'data')

class DatasetReader():
    '''Class to read the dataset of runs and their properties'''
    logger = Logger('DatasetReader').get_logger()

    @logger_decorator(logger)
    def __init__(self, h_names, data_dir = DATA_LATACD_PROCESSED_FOLDER_NAME, start = 0, end = -1):
        '''
        Initialize the DatasetReader object.

        Parameters:
        - data_dir (str): The directory path where the dataset data is stored.
        - start (int): The index of the first run directory to consider.
        - end (int): The index of the last run directory to consider.
        '''
        self.start = start
        self.end = end if end != -1 else None
        self.data_dir = data_dir
        self.h_names = h_names
        self.runs_times = {}
        self.runs_dict = {}

    @logger_decorator(logger)
    def get_runs_times(self):
        '''
        Get the dictionary of run times.

        Returns:
        - runs_times (dict): The dictionary of run times.
        '''
        return self.runs_times

    @logger_decorator(logger)
    def get_signal_df_from_dataset(self) -> pd.DataFrame:
        '''
        Get the pandas.Dataframe containing the signals for each run.

        Parameters:
        - binning (int): The binning factor for the histograms (optional, deprecated).

        Returns:
        - runs_dict (pandas.Dataframe): The dataframe containing the signals for each run.
        '''
        dataset_df = File.read_dfs_from_runs_pk_folder(folder_path=self.data_dir, add_smoothing=True, mode='mean', window=35, start=self.start, stop=self.end, cols_list=self.h_names + ['MET'])
        dataset_df['datetime'] = np.array(Time.from_met_to_datetime(dataset_df['MET'] - 1))
        self.runs_times['dataset'] = (dataset_df['datetime'][0], dataset_df['datetime'].iloc[-1])
        return dataset_df

if __name__ == '__main__':
    cr = DatasetReader(h_names=['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg'], start=0, end=-1)
    tile_signal_df = cr.get_signal_df_from_dataset()
    print(tile_signal_df.head())