# """
# TSLies: Time Series Anomaly Detection Framework - Core Modules

# This package contains the core modules for time series anomaly detection:
# - background: Machine learning models for background/trend prediction
# - trigger: Generic anomaly detection algorithms (FOCuS, change point detection)
# - utils: Utility functions and classes for data handling
# - plotter: Visualization tools for time series and anomalies
# - config: Configuration management system
# - dataset: Generic data loading and preprocessing utilities

# TSLies is domain-agnostic and can be applied to various time series data.
# Domain-specific applications (like ACD data) are in the 'applications' package.
# """

# # Core framework modules
# from . import config
# from . import utils
# from . import background
# from . import trigger
# from . import plotter

# # Common utilities that are frequently needed
# from .utils import Logger, logger_decorator, Data, File
# # from .config import get_effective_dir

# __version__ = '1.0.0'
# __author__ = 'Andrea Adelfio'
# __description__ = 'TSLies: Time Series Anomaly Detection Framework'
