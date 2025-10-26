'''
K-Nearest Neighbors predictors for time series anomaly detection.
Contains implementations of median and mean-based KNN regressors.
'''
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor

from ..utils import Logger, logger_decorator


class MedianKNeighborsRegressor(KNeighborsRegressor):
    '''The class for the Median K-Nearest Neighbors model.'''
    logger = Logger('MedianKNeighborsRegressor').get_logger()

    @logger_decorator(logger)
    def predict(self, X):
        '''Predict the target for the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs), dtype=int
            Target values.
        '''
        if self.weights == "uniform":
            # In that case, we do not need the distances to perform
            # the weighting so we do not compute them.
            neigh_ind = self.kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self.kneighbors(X)

        weights = None

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            y_pred = np.median(_y[neigh_ind], axis=1)

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred


class MultiMedianKNeighborsRegressor():
    '''The class for the Multi Median K-Nearest Neighbors model.'''
    logger = Logger('MultiMedianKNeighborsRegressor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, y_cols, x_cols, y_cols_raw=None, y_pred_cols=None, y_smooth_cols=None, latex_y_cols=None, with_generator=False):
        self.y_cols = y_cols
        self.x_cols = x_cols
        self.df_data = df_data
        # Optional parameters for consistency with MLObject interface
        self.y_cols_raw = y_cols_raw or y_cols
        self.y_pred_cols = y_pred_cols or y_cols
        self.y_smooth_cols = y_smooth_cols or y_cols
        self.latex_y_cols = latex_y_cols
        self.with_generator = with_generator

        self.y = None
        self.X = None
        self.multi_reg = None

    @logger_decorator(logger)
    def create_model(self, n_neighbors=5):
        self.y = self.df_data[self.y_cols].astype('float32')
        self.X = self.df_data[self.x_cols].astype('float32')
        self.multi_reg = MultiOutputRegressor(MedianKNeighborsRegressor(n_neighbors=n_neighbors))

    @logger_decorator(logger)
    def train(self):
        self.multi_reg.fit(self.X, self.y)

    @logger_decorator(logger)
    def predict(self, start=0, end=-1):
        df_data = self.df_data[start:end]
        df_ori = df_data[self.y_cols].reset_index(drop=True)
        y_pred = self.multi_reg.predict(df_data[self.x_cols])
        return df_ori, y_pred


class MultiMeanKNeighborsRegressor():
    '''The class for the Multi Mean K-Nearest Neighbors model.'''
    logger = Logger('MultiMeanKNeighborsRegressor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, y_cols, x_cols, y_cols_raw=None, y_pred_cols=None, y_smooth_cols=None, latex_y_cols=None, with_generator=False):
        self.y_cols = y_cols
        self.x_cols = x_cols
        self.df_data = df_data
        # Optional parameters for consistency with MLObject interface
        self.y_cols_raw = y_cols_raw or y_cols
        self.y_pred_cols = y_pred_cols or y_cols
        self.y_smooth_cols = y_smooth_cols or y_cols
        self.latex_y_cols = latex_y_cols
        self.with_generator = with_generator

        self.y = None
        self.X = None
        self.multi_reg = None

    @logger_decorator(logger)
    def create_model(self, n_neighbors=5):
        self.y = self.df_data[self.y_cols].astype('float32')
        self.X = self.df_data[self.x_cols].astype('float32')
        self.multi_reg = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=5)

    @logger_decorator(logger)
    def train(self):
        self.multi_reg.fit(self.X, self.y)

    @logger_decorator(logger)
    def predict(self, start=0, end=-1):
        df_data = self.df_data[start:end]
        df_ori = df_data[self.y_cols].reset_index(drop=True)
        y_pred = self.multi_reg.predict(df_data[self.x_cols])
        return df_ori, pd.DataFrame(y_pred, columns=self.y_cols)
