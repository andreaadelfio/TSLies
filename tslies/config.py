'''
Configuration file containing the folders and filenames for time series anomaly detection modules.
To check your configuration, run this file as a script.
'''
import os

def set_dir(root_dir):
    os.environ['TSLIES_DIR'] = root_dir
