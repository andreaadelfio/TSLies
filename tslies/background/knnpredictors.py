"""
K-Nearest Neighbors predictors for time series anomaly detection.
Contains implementations of median and mean-based KNN regressors.
"""
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor

from ..utils import Logger, logger_decorator


class MedianKNeighborsRegressor(KNeighborsRegressor):
    """KNN regressor variant computing the median instead of the mean for neighbours."""
    logger = Logger('MedianKNeighborsRegressor').get_logger()

    @logger_decorator(logger)
    def predict(self, X):
        """
        Predict regression outputs using the median of nearest neighbours.

        Parameters
        ----------
        - X (array-like | sparse matrix): Query samples of shape ``(n_queries, n_features)`` or
          ``(n_queries, n_indexed)`` if distance metrics are precomputed.

        Returns
        -------
        - ndarray: Predicted values of shape ``(n_queries, n_outputs)``.

        Raises
        ------
        - ValueError: Propagated when input validation in scikit-learn fails.
        """
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
    """Wrapper providing a multi-output median KNN regressor for background modelling."""
    logger = Logger('MultiMedianKNeighborsRegressor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, y_cols, x_cols, y_pred_cols=None, latex_y_cols=None, with_generator=False):
        """
        Initialise the multi-output median KNN regressor with data and settings.

        Parameters
        ----------
        - df_data (pd.DataFrame): Data frame containing features and targets.
        - y_cols (List[str]): Target columns to predict.
        - x_cols (List[str]): Feature columns feeding the regressors.
        - y_pred_cols (Optional[List[str]]): Destination column names for predictions.
        - latex_y_cols (Optional[List[str]]): Optional LaTeX labels for plots.
        - with_generator (bool): Placeholder flag for API parity with neural predictors.

        Returns
        -------
        - None
        """
        self.y_cols = y_cols
        self.x_cols = x_cols
        self.df_data = df_data
        # Optional parameters for consistency with MLObject interface
        self.y_pred_cols = y_pred_cols or y_cols
        self.latex_y_cols = latex_y_cols
        self.with_generator = with_generator

        self.y = None
        self.X = None
        self.multi_reg = None

    @logger_decorator(logger)
    def create_model(self, n_neighbors=5):
        """
        Instantiate the underlying scikit-learn estimator for the requested neighbour count.

        Parameters
        ----------
        - n_neighbors (int): Number of neighbours to consider for each prediction.

        Raises
        ------
        - ValueError: If ``n_neighbors`` is incompatible with the dataset size.
        """
        self.y = self.df_data[self.y_cols].astype('float32')
        self.X = self.df_data[self.x_cols].astype('float32')
        self.multi_reg = MultiOutputRegressor(MedianKNeighborsRegressor(n_neighbors=n_neighbors))

    @logger_decorator(logger)
    def train(self):
        """Fit the estimator on the cached feature/target arrays."""
        self.multi_reg.fit(self.X, self.y)

    @logger_decorator(logger)
    def predict(self, start=0, end=-1):
        """
        Produce predictions for the requested slice of the original data frame.

        Parameters
        ----------
        - start (int): Inclusive start index for slicing ``df_data``.
        - end (int): Exclusive end index for slicing ``df_data``.

        Returns
        -------
        - Tuple[pd.DataFrame, pd.DataFrame]: Ground-truth slice and prediction frame.
        """
        df_data = self.df_data[start:end]
        df_ori = df_data[self.y_cols].reset_index(drop=True)
        y_pred = self.multi_reg.predict(df_data[self.x_cols])
        return df_ori, y_pred


class MultiMeanKNeighborsRegressor():
    """Wrapper providing a multi-output mean KNN regressor for background modelling."""
    logger = Logger('MultiMeanKNeighborsRegressor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, y_cols, x_cols, y_pred_cols=None, latex_y_cols=None, with_generator=False):
        """
        Initialise the multi-output mean KNN regressor with training data and metadata.

        Parameters
        ----------
        - df_data (pd.DataFrame): Data frame containing inputs and targets.
        - y_cols (List[str]): Target columns for the regression.
        - x_cols (List[str]): Feature columns for the regression.
        - y_pred_cols (Optional[List[str]]): Optional alias for prediction columns.
        - latex_y_cols (Optional[List[str]]): Optional LaTeX labels for plotting.
        - with_generator (bool): Placeholder flag for API compatibility with MLObject subclasses.

        Returns
        -------
        - None
        """
        self.y_cols = y_cols
        self.x_cols = x_cols
        self.df_data = df_data
        # Optional parameters for consistency with MLObject interface
        self.y_pred_cols = y_pred_cols or y_cols
        self.latex_y_cols = latex_y_cols
        self.with_generator = with_generator

        self.y = None
        self.X = None
        self.multi_reg = None

    @logger_decorator(logger)
    def create_model(self, n_neighbors=5):
        """
        Build the scikit-learn estimator for the specified neighbour count.

        Parameters
        ----------
        - n_neighbors (int): Number of neighbours for averaging predictions.

        Raises
        ------
        - ValueError: If the neighbour count is invalid for the dataset.
        """
        self.y = self.df_data[self.y_cols].astype('float32')
        self.X = self.df_data[self.x_cols].astype('float32')
        self.multi_reg = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=5)

    @logger_decorator(logger)
    def train(self):
        """Fit the estimator on the cached feature/target arrays."""
        self.multi_reg.fit(self.X, self.y)

    @logger_decorator(logger)
    def predict(self, start=0, end=-1):
        """
        Produce predictions for the requested slice of the original data frame.

        Parameters
        ----------
        - start (int): Inclusive start index for slicing ``df_data``.
        - end (int): Exclusive end index for slicing ``df_data``.

        Returns
        -------
        - Tuple[pd.DataFrame, pd.DataFrame]: Ground-truth slice and prediction frame.
        """
        df_data = self.df_data[start:end]
        df_ori = df_data[self.y_cols].reset_index(drop=True)
        y_pred = self.multi_reg.predict(df_data[self.x_cols])
        return df_ori, pd.DataFrame(y_pred, columns=self.y_cols)
