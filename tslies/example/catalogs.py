'''
ACD-specific module for astronomical event catalog management.

This module provides utilities for handling astronomical event catalogs
(GBM, LAT, etc.) specific to Fermi mission data analysis. This is a 
domain-specific application built on the generic time series framework.
'''

import os
import glob
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time, TimeDelta
from astropy.table import vstack

from tslies.utils import Logger, logger_decorator
# from modules.config import DIR

class CatalogsUtils:
    def __init__(self):
        pass

    def read_fits_file(self, file_path):
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            table_data = Table(data)
            return table_data

    def extract_from_fermigbm_fits(self, fits_table) -> Table:
        """
        Extracts the relevant columns from the fits table and renames them according to the provided mapping.
        """
        fits_table['NAME'] = [name.strip() for name in fits_table['NAME']]
        fits_table['TRIGGER_TYPE'] = [trigger_type.strip() for trigger_type in fits_table['TRIGGER_TYPE']]
        fits_table['TIME'] = (Time(fits_table['TRIGGER_TIME'] + 2400000.5, format='jd', scale='utc') - TimeDelta(20, format='sec')).iso
        fits_table['END_TIME'] = (Time(fits_table['TRIGGER_TIME'] + 2400000.5, format='jd', scale='utc') + TimeDelta(20, format='sec')).iso
        fits_table['TRIGGER_TIME'] = (Time(fits_table['TRIGGER_TIME'] + 2400000.5, format='jd', scale='utc')).iso
        fits_table['CAT_NAME'] = ['FERMIGBM'] * len(fits_table)
        fits_table.remove_columns(['RA', 'DEC', 'RELIABILITY', 'TRIGGER_NAME'])
        return fits_table

    def extract_from_fermigbmburst_fits(self, fits_table) -> Table:
        """
        Extracts the relevant columns from the fits table and renames them according to the provided mapping.
        """
        fits_table['NAME'] = [name.strip() for name in fits_table['NAME']]
        fits_table['TRIGGER_TYPE'] = ['GRB'] * len(fits_table)
        fits_table['TIME'] = (Time(fits_table['TRIGGER_TIME'] + 2400000.5, format='jd', scale='utc') - TimeDelta(20, format='sec')).iso
        fits_table['END_TIME'] = (Time(fits_table['TRIGGER_TIME'] + 2400000.5, format='jd', scale='utc') + TimeDelta(fits_table['T90'], format='sec')).iso
        fits_table['TRIGGER_TIME'] = (Time(fits_table['TRIGGER_TIME'] + 2400000.5, format='jd', scale='utc')).iso
        fits_table['CAT_NAME'] = ['FERMIGBMBURST'] * len(fits_table)
        fits_table.remove_columns(['T90'])
        return fits_table

    def extract_from_svom_fits(self, fits_table) -> Table:
        """
        Extracts the relevant columns from the fits table and renames them according to the provided mapping.
        """
        fits_table['NAME'] = [value.split(':')[0] for value in fits_table['subject']]
        fits_table['utc'] = [value[:19] for value in fits_table['utc']]
        fits_table['TRIGGER_TYPE'] = [
            'SGR' if 'SGR' in value else ('GRB' if 'GRB' in value else 'UNKNOWN') for value in fits_table['subject']
        ]
        fits_table['TIME'] = (Time(fits_table['utc'], scale='utc') - TimeDelta(20, format='sec')).iso
        fits_table['TRIGGER_TIME'] = (Time(fits_table['utc'], scale='utc')).iso
        fits_table['END_TIME'] = (Time(fits_table['utc'], scale='utc') + TimeDelta(20, format='sec')).iso
        fits_table['CAT_NAME'] = ['SVOM'] * len(fits_table)
        fits_table.remove_columns(['ra', 'dec', 'utc', 'subject'])
        return fits_table

    def extract_from_ep_fits(self, fits_table) -> Table:
        """
        Extracts the relevant columns from the fits table and renames them according to the provided mapping.
        """
        fits_table['col4'] = [value.strip() for value in fits_table['col4']]
        fits_table = fits_table[fits_table['col4'] != 'N/A']
        fits_table['NAME'] = [value.split(':')[0] for value in fits_table['col1']]
        fits_table['col4'] = [value[:19] for value in fits_table['col4']]
        fits_table['TRIGGER_TYPE'] = [
            'SGR' if 'SGR' in value else ('GRB' if 'GRB' in value else 'UNKNOWN') for value in fits_table['col1']
        ]
        fits_table['TIME'] = (Time(fits_table['col4'], scale='utc') - TimeDelta(20, format='sec')).iso
        fits_table['TRIGGER_TIME'] = Time(fits_table['col4'], scale='utc').iso
        fits_table['END_TIME'] = (Time(fits_table['col4'], scale='utc') + TimeDelta(20, format='sec')).iso
        fits_table['CAT_NAME'] = ['EP'] * len(fits_table)
        fits_table.remove_columns(['col1', 'col2', 'col3', 'col4'])
        return fits_table

    def extract_from_swift_fits(self, txt_table) -> Table:
        """
        Extracts the relevant columns from the txt table and renames them according to the provided mapping.
        """
        txt_table['NAME'] = [value.split(':')[0] for value in txt_table['GRBname']]
        txt_table['TRIGGER_TYPE'] = ['GRB'] * len(txt_table)
        txt_table['TIME'] = (Time(txt_table['Trig_time_UTC'], scale='utc') - TimeDelta(20, format='sec')).iso
        txt_table['TRIGGER_TIME'] = Time(txt_table['Trig_time_UTC'], scale='utc').iso
        txt_table['END_TIME'] = (Time(txt_table['Trig_time_UTC'], scale='utc') + TimeDelta(txt_table['T90'], format='sec')).iso
        txt_table['CAT_NAME'] = ['SWIFT'] * len(txt_table)
        txt_table.remove_columns(['GRBname', 'Trig_ID', 'Trig_time_met', 'Trig_time_UTC', 'RA_ground', 'DEC_ground', 'Image_position_err', 'Image_SNR', 'T90', 'T90_err', 'T50', 'T50_err', 'Evt_start_sincetrig', 'Evt_stop_sincetrig', 'pcode', 'Trigger_method', 'XRT_detection', 'comment'])
        return txt_table

    def create_fits_from_table(self, filename, table):
        """
        Create a FITS file from a Table.
        """
        table.write(filename, format='fits', overwrite=True)


class CatalogsReader:
    """Class to handle reading and processing of catalogs and detection files.
    """
    logger = Logger('CatalogsReader').get_logger()

    @logger_decorator(logger)
    def __init__(self):
        """
        Initialize the Catalogs object.
        """
        self.catalog_df = self.read_fits_file()

    @logger_decorator(logger)
    def read_fits_file(self):
        fits_file_path = os.path.join(os.environ['TSLIES_DIR'], 'catalogs', 'total_catalog.fits')
        file_paths = glob.glob(fits_file_path)
        df = pd.DataFrame()
        for file_path in file_paths:
            with fits.open(file_path) as hdul:
                data = hdul[1].data
                df_tmp = pd.DataFrame(data)
                df_tmp['TRIGGER_TIME'] = pd.to_datetime(df_tmp['TRIGGER_TIME'], utc=False)
                df_tmp['END_TIME'] = pd.to_datetime(df_tmp['END_TIME'], utc=False)
                df_tmp['TIME'] = pd.to_datetime(df_tmp['TIME'], utc=False)
                df = pd.concat([df, df_tmp])
        return df
    

if __name__ == "__main__":
    cu = CatalogsUtils()
    SVOM_FITS_FILE_PATH = os.path.join(os.environ['TSLIES_DIR'], 'catalogs', 'SVOM.fits')
    fits_table = cu.read_fits_file(SVOM_FITS_FILE_PATH)
    final_table = cu.extract_from_svom_fits(fits_table)

    EP_FITS_FILE_PATH = os.path.join(os.environ['TSLIES_DIR'], 'catalogs', 'EP.fits')
    fits_table = cu.read_fits_file(EP_FITS_FILE_PATH)
    final_table = vstack([final_table, cu.extract_from_ep_fits(fits_table)])

    FermiGBMTrig_FITS_FILE_PATH = os.path.join(os.environ['TSLIES_DIR'], 'catalogs', 'FermiGBMTrig.fits')
    fits_table = cu.read_fits_file(FermiGBMTrig_FITS_FILE_PATH)
    final_table = vstack([final_table, cu.extract_from_fermigbm_fits(fits_table)])

    SWIFT_FITS_FILE_PATH = os.path.join(os.environ['TSLIES_DIR'], 'catalogs', 'SWIFT.txt')
    txt_table = Table.read(SWIFT_FITS_FILE_PATH, format='ascii')
    final_table = vstack([final_table, cu.extract_from_swift_fits(txt_table)])

    FermiGBMBurst_FITS_FILE_PATH = os.path.join(os.environ['TSLIES_DIR'], 'catalogs', 'FermiGBMBurstCat.fits')
    fits_table = cu.read_fits_file(FermiGBMBurst_FITS_FILE_PATH)
    final_table = vstack([final_table, cu.extract_from_fermigbmburst_fits(fits_table)])

    final_table.sort('TRIGGER_TIME')
    final_cat_FILE_PATH = os.path.join(os.environ['TSLIES_DIR'], 'catalogs', 'total_catalog.fits')
    cu.create_fits_from_table(final_cat_FILE_PATH, final_table)

    cr = CatalogsReader()
    final_table = cr.read_fits_file()
    # print solo tre righe d'esempio per ciascun catalogo
    print("SVOM Catalog:")
    print(final_table[final_table['CAT_NAME'] == 'SVOM'][:3])
    print("\nEP Catalog:")
    print(final_table[final_table['CAT_NAME'] == 'EP'][:3])
    print("\nFERMI Catalog:")
    print(final_table[final_table['CAT_NAME'] == 'FERMIGBM'][:3])
    print("\nSWIFT Catalog:")
    print(final_table[final_table['CAT_NAME'] == 'SWIFT'][:3])
    print("\nFERMI GBM Burst Catalog:")
    print(final_table[final_table['CAT_NAME'] == 'FERMIGBMBURST'][:3])