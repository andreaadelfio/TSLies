
import os
import gc
import pandas as pd
import numpy as np

# Keras
from keras.callbacks import LearningRateScheduler # pylint: disable=E0401
# TensorFlow for Bayesian Neural Network
import tf_keras
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions
# ACDAnomalies modules
from ..utils import Logger, logger_decorator, File, Data
from .mlobject import MLObject


class BNNPredictor(MLObject):
    """Bayesian neural network predictor leveraging dense layers with uncertainty head."""
    logger = Logger('BNNPredictor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, y_cols, x_cols, y_pred_cols=None, latex_y_cols=None, units=None, with_generator=False):
        """
        Initialise the Bayesian neural network predictor with data and configuration.

        Parameters
        ----------
        - df_data (pd.DataFrame): Source data frame with feature and target columns.
        - y_cols (List[str]): Target column names to model.
        - x_cols (List[str]): Feature column names feeding the network.
        - y_pred_cols (Optional[List[str]]): Optional prediction column aliases.
        - latex_y_cols (Optional[List[str]]): Optional LaTeX-friendly labels for plots.
        - units (Optional[List[int]]): Units metadata associated with targets.
        - with_generator (bool): Flag enabling generator-based training (not implemented).

        Raises
        ------
        - ValueError: Propagated from the base class when mandatory data are missing.
        """
        super().__init__(df_data, y_cols, x_cols, y_pred_cols, latex_y_cols=latex_y_cols, units=units, with_generator=with_generator)

    @logger_decorator(logger)
    def create_model(self):
        """
        Build and compile the deterministic network with probabilistic output head.

        Parameters
        ----------
        - None

        Raises
        ------
        - ValueError: If layer configuration is missing from ``units_for_layers``.
        """

        self.nn_r = tf_keras.Sequential([
            tf_keras.Input(shape=(len(self.x_cols), )),
        ])

        for units in list(self.units_for_layers):
            self.nn_r.add(tf_keras.layers.Dense(units, activation='relu'))#, kernel_regularizer=tf_keras.regularizers.l2(1e-4)))
            # self.nn_r.add(tf_keras.layers.Dropout(0.1))
        self.nn_r.add(tf_keras.layers.Dense(2*len(self.y_cols), activation='linear'))

        self.nn_r.compile(optimizer=tf_keras.optimizers.Adam(),
            loss=self.loss,
            metrics=self.metrics)
    
    @logger_decorator(logger)
    def train(self):
        """
        Train the Bayesian neural network using the prepared train/test split.

        Parameters
        ----------
        - None

        Returns
        -------
        - tf.keras.callbacks.History: History containing loss and metric traces.

        Raises
        ------
        - NotImplementedError: When generator-based training is requested.
        """
        if self.with_generator:
            raise NotImplementedError('With generator not implemented yet for ABNNPredictor')
        else:
            history = self.nn_r.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.bs, validation_split=0.3,
                      callbacks=self.callbacks)

        return history
    
    @logger_decorator(logger)
    def predict(self, start = 0, end = -1, mask_column='index', write_bkg=True, write_frg=False, num_batches=1, save_predictions_plot=False, support_variables=[]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Produce background predictions along with estimated standard deviations.

        Parameters
        ----------
        - start (Union[int, str]): Start index or timestamp for selecting the slice.
        - end (Union[int, str]): End index or timestamp for selecting the slice.
        - mask_column (str): Data frame column used for slicing the input data.
        - write_bkg (bool): Persist background predictions to the model folder.
        - write_frg (bool): Persist foreground values to disk if ``True``.
        - num_batches (int): Number of batches to split inference over.
        - save_predictions_plot (bool): Enable saving plots of predictions vs targets.
        - support_variables (List[str]): Additional variables to include in plot data.

        Returns
        -------
        - Tuple[pd.DataFrame, pd.DataFrame]: Original targets and predictions with uncertainty columns.: Empty slices result in empty data frames without raising.
        """
        if start != 0 or end != -1:
            df_data = Data.get_masked_dataframe(data=self.df_data, start=start, stop=end, column=mask_column, reset_index=False)
            if df_data.empty:
                return pd.DataFrame(), pd.DataFrame()
            scaled_data = self.scaler_x.transform(df_data[self.x_cols].values)
        else:
            df_data = self.df_data
            scaled_data = self.X
        y_pred = np.zeros(shape=(0, 2*len(self.y_cols)))
        batch_size = len(scaled_data)//num_batches
        for i in range(0, len(scaled_data), batch_size):
            y_pred = np.append(y_pred, self.nn_r.predict(scaled_data[i:i + batch_size]), axis=0)
        mean_pred = self.scaler_y.inverse_transform(y_pred[:, :len(self.y_cols)])
        log_var_pred = y_pred[:, len(self.y_cols):] + 2 * np.log(self.scaler_y.scale_)

        std_pred = np.sqrt(np.exp(log_var_pred))
        y_pred = pd.DataFrame(mean_pred, columns=self.y_cols)
        y_std = pd.DataFrame(std_pred, columns=[f'{col}_std' for col in self.y_cols])
        y_pred = pd.concat([y_pred, y_std], axis=1)
        y_pred['datetime'] = df_data['datetime'].values
        y_pred.reset_index(drop=True, inplace=True)
        y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(self.y_pred_cols, self.y_cols)}).drop(columns=self.y_cols)
        df_ori = df_data[self.y_cols].copy()
        df_ori.loc[:, 'datetime'] = df_data['datetime'].values
        df_ori.reset_index(drop=True, inplace=True)
        if write_bkg:
            path = os.path.join(os.path.dirname(self.model_path))
            if not self.model_id:
                path = os.path.dirname(self.model_path)
            File.write_df_on_file(y_pred, os.path.join(path, 'bkg'))
            gc.collect()

            if write_frg:
                path = os.path.join(os.path.dirname(self.model_path))
                if not self.model_id:
                    path = os.path.dirname(self.model_path)
                File.write_df_on_file(df_ori, os.path.join(path, 'frg'))
        if save_predictions_plot:
            tiles_df = Data.merge_dfs(df_data[['Xpos_middle'] + ['datetime'] + support_variables], y_pred)
            self.save_predictions_plots(tiles_df, start, end, self.params)
        return df_ori, y_pred
    