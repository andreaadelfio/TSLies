
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
    '''The class for the Bayesian Neural Network model.'''
    logger = Logger('BNNPredictor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, y_cols, x_cols, y_cols_raw, y_pred_cols, y_smooth_cols, latex_y_cols=None, units=None, with_generator=False):
        super().__init__(df_data, y_cols, x_cols, y_cols_raw, y_pred_cols, y_smooth_cols, latex_y_cols=latex_y_cols, units=units, with_generator=with_generator)

    @logger_decorator(logger)
    def create_model(self):
        '''Builds the Bayesian Neural Network model.'''

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
        '''Trains the model.'''
        es = tf_keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.002,
                           patience=10, start_from_epoch=190)
        mc = tf_keras.callbacks.ModelCheckpoint(self.model_path,
                             monitor='val_loss', mode='min', verbose=0, save_best_only=True)
        
        if not self.lr:
            callbacks = [self.custom_callback(self)]
        else:
            call_lr = LearningRateScheduler(self.scheduler)
        callbacks = [mc, self.custom_callback(self, 5)]

        if self.with_generator:
            history = self.nn_r.fit(self.df_data, epochs=self.epochs, batch_size=32, validation_split=0.3)
        else:
            history = self.nn_r.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.bs, validation_split=0.3,
                      callbacks=callbacks)

        return history
    
    @logger_decorator(logger)
    def predict(self, start = 0, end = -1, mask_column='index', write_bkg=True, write_frg=False, num_batches=1, save_predictions_plot=False, support_variables=[]) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''Predicts the output data.
        
        Parameters:
        ----------
            start (int): The starting index. Default is 0.
            end (int): The ending index. Defualt is -1.
            '''
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
    