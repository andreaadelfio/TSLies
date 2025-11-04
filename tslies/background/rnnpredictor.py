
# import itertools
import os
import gc
import pandas as pd
import numpy as np
# import pickle

# sklearn
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE, median_absolute_error as MeAE
# Keras
from keras.optimizers import Adam, Nadam, RMSprop, SGD # pylint: disable=E0401
from keras import Input, Model
from keras.layers import Dense, Dropout, BatchNormalization, LSTM # pylint: disable=E0401
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler # pylint: disable=E0401
from keras.models import load_model # pylint: disable=E0401
from keras.utils import plot_model # pylint: disable=E0401
# TensorFlow for Bayesian Neural Network
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions
# ACDAnomalies modules
from ..utils import Logger, logger_decorator, File, Data
from .mlobject import MLObject
    
class RNNPredictor(MLObject):
    logger = Logger('RNN').get_logger()

    def __init__(self, df_data, y_cols, x_cols, y_pred_cols=None, latex_y_cols=None, with_generator=False):
        super().__init__(df_data, y_cols, x_cols, y_pred_cols, latex_y_cols, with_generator)

    @logger_decorator(logger)
    def reshape_data(self, x, y):
        """Reshapes the data for the RNN model."""
        y = y[self.params['timesteps']:]
        x = np.array([x[i:i + self.params['timesteps']] for i in np.arange(len(x) - self.params['timesteps'])])
        x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2]))
        return x, y

    @logger_decorator(logger)
    def create_model(self):
        """Creates the model."""
        
        inputs = Input(shape=(None, len(self.x_cols)))
        hidden = LSTM(90)(inputs)
        for count, units in enumerate(list(self.units_for_layers)):
            hidden = Dense(units, activation='relu')(hidden)
            if self.norm:
                hidden = BatchNormalization()(hidden)
            if self.drop:
                hidden = Dropout(self.do)(hidden)
        outputs = Dense(len(self.y_cols), activation='linear')(hidden)

        self.nn_r = Model(inputs=[inputs], outputs=outputs)
        plot_model(self.nn_r, to_file=os.path.join(os.path.dirname(self.model_path), 'schema.png'),
                   show_shapes=True, show_layer_names=True, rankdir='TB')

        if self.opt_name == 'Adam':
            opt = Adam(beta_1=0.9, beta_2=0.99, epsilon=1e-07)
        elif self.opt_name == 'Nadam':
            opt = Nadam(beta_1=0.9, beta_2=0.99, epsilon=1e-07)
        elif self.opt_name == 'RMSprop':
            opt = RMSprop(rho=0.6, momentum=0.0, epsilon=1e-07)
        elif self.opt_name == 'SGD':
            opt = SGD()

        self.nn_r.compile(loss=self.loss_type, optimizer=opt, metrics=['accuracy'])

    @logger_decorator(logger)
    def train(self):
        """Trains the model."""
        self.X_train, self.y_train = self.reshape_data(self.X_train, self.y_train)
        self.X_test, self.y_test = self.reshape_data(self.X_test, self.y_test)
        
        es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01,
                           patience=10, start_from_epoch=80)
        mc = ModelCheckpoint(self.model_path, 
                             monitor='val_loss', mode='min', verbose=0, save_best_only=True)

        if not self.lr:
            callbacks = [es, mc]
        else:
            call_lr = LearningRateScheduler(self.scheduler)
            callbacks = [es, mc, call_lr]
        history = self.nn_r.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.bs,
                        validation_split=0.3, callbacks=self.callbacks)
        
        nn_r = load_model(self.model_path)

        pred_train = nn_r.predict(self.X_train)
        pred_test = nn_r.predict(self.X_test)
        idx = 0
        text = ''
        for col in self.y_cols:
            mae_tr = MAE(self.y_train[:, idx], pred_train[:, idx])
            self.mae_tr_list.append(mae_tr)
            mae_te = MAE(self.y_test[:, idx], pred_test[:, idx])
            diff_i = (self.y_test[:, idx] - pred_test[:, idx])
            mean_diff_i = (diff_i).mean()
            meae_tr = MeAE(self.y_train[:, idx], pred_train[:, idx])
            meae_te = MeAE(self.y_test[:, idx], pred_test[:, idx])
            median_diff_i = np.median(diff_i)
            text += f"MAE_train_{col} : {mae_tr:0.5f}\t" + \
                    f"MAE_test_{col} : {mae_te:0.5f}\t" + \
                    f"mean_diff_test_pred_{col} : {mean_diff_i:0.5f}\t" + \
                    f"MeAE_train_{col} {meae_tr:0.5f}\t" + \
                    f"MeAE_test_{col} {meae_te:0.5f}\t" + \
                    f"median_diff_test_pred_{col} {median_diff_i:0.5f}\n"
            idx = idx + 1

        nn_r.save(self.model_path)
        self.nn_r = nn_r
        with open(os.path.join(os.path.dirname(self.model_path), 'performance.txt'), "w") as text_file:
            text_file.write(text)
        self.text = text
        return history

    @logger_decorator(logger)
    def predict(self, start = 0, end = -1, write_bkg=True, write_frg=False, num_batches=1, save_predictions_plot=False, support_variables=[]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Predicts the output data.
        
        Parameters:
        ----------
            start (int): The starting index. (Default is 0).
            end (int): The ending index. (Defualt is -1).
            write (bool): If the predicted and original dataset will be written in a file. (Defualt is True)
            batched (bool): If the dataset will be modeled in batch. (Defualt is False)"""
        df_data = Data.get_masked_dataframe(data=self.df_data, start=start, stop=end, reset_index=False)
        if df_data.empty:
            return pd.DataFrame(), pd.DataFrame()
        data = self.scaler_x.transform(df_data)
        data = np.array([data[i:i + self.params['timesteps']] for i in np.arange(len(data) - self.params['timesteps'])])
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2]))
        if num_batches > 1:
            pred_x_tot = np.array([])
            batch_size = self.params['timesteps']
            for i in range(0, len(data), batch_size):
                pred_x_tot = np.append(pred_x_tot, self.nn_r.predict(data[i:i + batch_size]))
            pred_x_tot = np.reshape(pred_x_tot, (len(data), len(self.y_cols)))
        else:
            pred_x_tot = self.nn_r.predict(data)
        gc.collect()

        df_ori = self.df_data[start:end][self.y_cols].reset_index(drop=True)
        y_pred = pd.DataFrame(pred_x_tot, columns=self.y_cols)
        df_ori['datetime'] = self.df_data[start:end]['datetime'].values
        y_pred['datetime'] = self.df_data[start+self.timesteps:end]['datetime'].values

        df_ori.reset_index(drop=True, inplace=True)
        y_pred.reset_index(drop=True, inplace=True)
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
            y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(self.y_pred_cols, self.y_cols)}).drop(columns=self.y_cols)
            tiles_df = Data.merge_dfs(df_data[self.y_cols + ['datetime'] + support_variables], y_pred)
            self.save_predictions_plots(tiles_df, start, end, self.params)
        return df_ori, y_pred
