
# import itertools
import os
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
# import pickle

# sklearn
# from sklearn.model_selection import train_test_split
# Keras
# TensorFlow for Bayesian Neural Network
import tf_keras
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions
import tensorflow as tf
# ACDAnomalies modules
from ..utils import Logger, logger_decorator, File, Data
from .mlobject import MLObject
    
class PBNNPredictor(MLObject):
    """The class for the Probabilistic Bayesian Neural Network model."""
    logger = Logger('PBNNPredictor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, y_cols, x_cols, y_pred_cols=None, latex_y_cols=None, with_generator=False):
        super().__init__(df_data, y_cols, x_cols, y_pred_cols, latex_y_cols, with_generator)
    
    def prior_trainable(self, kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf_keras.Sequential([
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t, scale=1),
                reinterpreted_batch_ndims=1)),
        ])

    def random_gaussian_initializer(self, shape, dtype):
        n = int(shape / 2)
        loc_norm = tf.random_normal_initializer(mean=0., stddev=0.1)
        loc = tf.Variable(
            initial_value=loc_norm(shape=(n,), dtype=dtype)
        )
        scale_norm = tf.random_normal_initializer(mean=-3., stddev=0.1)
        scale = tf.Variable(
            initial_value=scale_norm(shape=(n,), dtype=dtype)
        )
        return tf.concat([loc, scale], 0)
    
    # The posterior is modeled as n_weights independent Normal distribution with learnable parameters
    def posterior_mean_field(self, kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return tf_keras.Sequential([
            tfp.layers.VariableLayer(2 * n, dtype=dtype, initializer=self.random_gaussian_initializer, trainable=True),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n],
                            scale=1e-5 + 0.001*tf.nn.softplus(c + t[..., n:])),
                reinterpreted_batch_ndims=1)),
        ])
    
    def normal_sp(self, params):
        return tfd.Normal(loc=params[:,:len(self.y_cols)],\
                      scale=1e-5 + 0.00001*tf_keras.backend.exp(params[:,len(self.y_cols):]))# both parameters are learnable

    @logger_decorator(logger)
    def create_model(self):
        """Builds the Bayesian Neural Network model."""

        self.nn_r = tf_keras.Sequential([
            tf_keras.Input(shape=(len(self.x_cols), )),
        ])

        for units in list(self.units_for_layers):
            self.nn_r.add(tfpl.DenseVariational(units, self.posterior_mean_field, self.prior_trainable, kl_weight=1/self.X_train.shape[0]))
        self.nn_r.add(tfpl.DenseVariational(2*len(self.y_cols), self.posterior_mean_field, self.prior_trainable, kl_weight=1/self.X_train.shape[0]))
        self.nn_r.add(tfpl.DistributionLambda(self.normal_sp))

        if self.lr:
            opt = tf_keras.optimizers.Adam(learning_rate=self.lr)
        else:
            opt = tf_keras.optimizers.Adam()

        self.nn_r.compile(optimizer=opt,
                  loss=self.closses.NLL,
                  metrics=[self.closses.mae])
    
    @logger_decorator(logger)
    def train(self):
        """Trains the model."""
        if self.with_generator:
            raise NotImplementedError('With generator not implemented yet for ABNNPredictor')
        else:
            history = self.nn_r.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.bs, validation_split=0.3,
                      callbacks=self.callbacks)
        
        return history
    
    @logger_decorator(logger)
    def predict(self, start = 0, end = -1, runs=250, mask_column='index', write_bkg=True, write_frg=False, num_batches=1, save_predictions_plot=True, support_variables=[]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Predicts the output data.
        
        Parameters:
        ----------
            start (int): The starting index. Default is 0.
            end (int): The ending index. Defualt is -1."""
        df_data = Data.get_masked_dataframe(data=self.df_data, start=start, stop=end, column=mask_column, reset_index=False)
        if df_data.empty:
            return pd.DataFrame(), pd.DataFrame()
        scaled_data = self.scaler_x.transform(df_data[self.x_cols])
        if num_batches > 1:
            preds = np.array([])
            batch_size = len(scaled_data), num_batches
            for i in range(0, len(scaled_data), batch_size):
                preds = np.append(preds, self.nn_r.predict(scaled_data[i:i + batch_size]))
        else:
            preds = tf.zeros([runs, len(scaled_data), len(self.y_cols)]).numpy()
            for i in tqdm(range(0, runs)):
                preds[i,:] = self.nn_r.predict(scaled_data, verbose=0)
                
        preds_std = tf.experimental.numpy.std( preds, axis=0, keepdims=None ).numpy()
        preds_mean = tf.experimental.numpy.mean( preds, axis=0, dtype=None, out=None, keepdims=None ).numpy()
        y_pred = self.scaler_y.inverse_transform(preds_mean)
        y_std = preds_std * self.scaler_y.scale_
        y_pred = pd.DataFrame(preds_mean, columns=self.y_cols)
        y_std = pd.DataFrame(preds_std, columns=[f'{col}_std' for col in self.y_cols])
        y_pred = pd.concat([y_pred, y_std], axis=1)
        y_pred['datetime'] = df_data['datetime'].values
        y_pred.reset_index(drop=True, inplace=True)
        df_ori = df_data[self.y_cols].reset_index(drop=True)
        df_ori['datetime'] = df_data['datetime'].values
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
            y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(self.y_pred_cols, self.y_cols)}).drop(columns=self.y_cols)
            tiles_df = Data.merge_dfs(df_data[self.y_cols + ['datetime'] + support_variables], y_pred)
            self.save_predictions_plots(tiles_df, start, end, self.params)
        return df_ori, y_pred
