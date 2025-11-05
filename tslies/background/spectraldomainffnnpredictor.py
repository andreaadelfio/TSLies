
import os
import pandas as pd

# TensorFlow for Bayesian Neural Network
import tf_keras
# ACDAnomalies modules
from ..utils import Logger, logger_decorator, File, Data
from .mlobject import MLObject

class SpectralDomainFFNNPredictor(MLObject):
    """Feed-Forward Neural Network working in the frequency domain."""
    logger = Logger('SpectralDomainFFNNPredictor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, y_cols, x_cols, y_pred_cols=None, latex_y_cols=None, units=None, with_generator=False):
        super().__init__(df_data, y_cols, x_cols, y_pred_cols, latex_y_cols=latex_y_cols, units=units, with_generator=with_generator)

    @logger_decorator(logger)
    def create_model(self):
        """Builds the spectral-domain neural network model."""
        self.nn_r = tf_keras.Sequential()
        self.nn_r.add(tf_keras.Input(shape=(len(self.x_cols),)))
        for units in list(self.units_for_layers):
            self.nn_r.add(tf_keras.layers.Dense(units, activation='relu'))
        self.nn_r.add(tf_keras.layers.Dense(len(self.y_cols), activation='linear'))

        self.nn_r.compile(
            optimizer=tf_keras.optimizers.Adam(learning_rate=self.lr),
            loss=self.loss,
            metrics=self.metrics)

    @logger_decorator(logger)
    def train(self):
        """Trains the model."""
        history = self.nn_r.fit(
            self.X_train, self.y_train,
            validation_split=0.3,
            batch_size=self.bs,
            epochs=self.epochs,
            callbacks=self.callbacks
        )

        return history

    @logger_decorator(logger)
    def predict(self, start=0, end=-1, mask_column='index', write_bkg=True, write_frg=False, num_batches=1, save_predictions_plot=False, support_variables=[]):
        """Predicts the output and returns the result in time domain."""
        if start != 0 or end != -1:
            df_data = Data.get_masked_dataframe(data=self.df_data, start=start, stop=end, column=mask_column, reset_index=False)
            if df_data.empty:
                return pd.DataFrame(), pd.DataFrame()
            scaled_data = self.scaler_x.transform(df_data[self.x_cols])
        else:
            df_data = self.df_data
            scaled_data = self.X

        y_pred = self.nn_r.predict(scaled_data)
        y_pred = self.scaler_y.inverse_transform(y_pred)

        y_pred_df = pd.DataFrame(y_pred, columns=self.y_pred_cols)
        y_pred_df['datetime'] = df_data['datetime'].values
        y_pred_df.reset_index(drop=True, inplace=True)

        if write_bkg:
            path = os.path.join(os.path.dirname(self.model_path))
            File.write_df_on_file(y_pred_df, os.path.join(path, 'bkg'))

        if save_predictions_plot:
            tiles_df = Data.merge_dfs(df_data[self.y_cols + ['datetime'] + support_variables], y_pred_df)
            self.save_predictions_plots(tiles_df, start, end, self.params)

        return df_data[self.y_cols], y_pred_df
