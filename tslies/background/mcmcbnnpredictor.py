
# import itertools
import os
import gc
import pandas as pd
import numpy as np
# import pickle

# sklearn
# from sklearn.model_selection import train_test_split
# Keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler # pylint: disable=E0401
# TensorFlow for Bayesian Neural Network
import tf_keras
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions
import tensorflow as tf
# ACDAnomalies modules
from ..utils import Logger, logger_decorator, File, Data
from .mlobject import MLObject
    
class MCMCBNNPredictor(MLObject):
    """The class for the Bayesian Neural Network model optimized with MCMC."""
    logger = Logger('MCMCBNNPredictor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, y_cols, x_cols, y_pred_cols=None, latex_y_cols=None, with_generator=False):
        raise NotImplementedError('MCMCBNNPredictor is not implemented yet.')
        super().__init__(df_data, y_cols, x_cols, y_pred_cols, latex_y_cols, with_generator)

    def prior_trainable(self, kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf_keras.Sequential([
            tfpl.VariableLayer(n, dtype=dtype),
            tfpl.DistributionLambda(lambda t: tfd.Independent(
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

    def posterior_mean_field(self, kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return tf_keras.Sequential([
            tfpl.VariableLayer(2 * n, dtype=dtype, initializer=self.random_gaussian_initializer, trainable=True),
            tfpl.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n],
                           scale=1e-5 + 0.001 * tf.nn.softplus(c + t[..., n:])),
                reinterpreted_batch_ndims=1)),
        ])

    def normal_sp(self, params):
        return tfd.Normal(loc=params[:, :len(self.y_cols)],
                          scale=1e-5 + 0.00001 * tf_keras.backend.exp(params[:, len(self.y_cols):]))

    @logger_decorator(logger)
    def create_model(self):
        """Builds the Bayesian Neural Network model."""

        self.nn_r = tf_keras.Sequential([
            tf_keras.Input(shape=(len(self.x_cols),)),
        ])

        for units in list(self.units_for_layers):
            self.nn_r.add(tfpl.DenseVariational(units, self.posterior_mean_field, self.prior_trainable, kl_weight=1/self.X_train.shape[0]))
        self.nn_r.add(tfpl.DenseVariational(2 * len(self.y_cols), self.posterior_mean_field, self.prior_trainable, kl_weight=1/self.X_train.shape[0]))
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
        """Trains the model using MCMC."""
        # Define the MCMC transition kernel
        num_results = 1000
        num_burnin_steps = 500

        def target_log_prob_fn(*params):
            return -self.nn_r(self.X_train).log_prob(self.y_train)

        initial_state = [tf.zeros([self.X_train.shape[1], units]) for units in self.units_for_layers]

        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=0.01,
            num_leapfrog_steps=3)

        @tf.function
        def run_chain():
            return tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=initial_state,
                kernel=kernel,
                trace_fn=lambda current_state, kernel_results: kernel_results)

        samples, kernel_results = run_chain()

        # Use the samples to set the weights of the model
        for i, layer in enumerate(self.nn_r.layers):
            if isinstance(layer, tfpl.DenseVariational):
                layer.kernel_posterior = tfd.Normal(loc=samples[i][0], scale=samples[i][1])


        if self.with_generator:
            raise NotImplementedError('With generator not implemented yet for ABNNPredictor')
        else:
            history = self.nn_r.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.bs, validation_split=0.3,
                                    callbacks=self.callbacks)

        return history

    @logger_decorator(logger)
    def predict(self, start=0, end=-1, mask_column='index', write_bkg=True, write_frg=False, num_batches=1, save_predictions_plot=False, support_variables=[]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Predicts the output data.
        
        Parameters
        ----------
            start (int): The starting index. Default is 0.
            end (int): The ending index. Default is -1.
            """
        if start != 0 or end != -1:
            df_data = Data.get_masked_dataframe(data=self.df_data, start=start, stop=end, column=mask_column, reset_index=False)
            if df_data.empty:
                return pd.DataFrame(), pd.DataFrame()
            scaled_data = self.scaler_x.transform(df_data[self.x_cols])
        else:
            df_data = self.df_data
            scaled_data = self.X
        y_pred = np.zeros(shape=(0, 2 * len(self.y_cols)))
        batch_size = len(scaled_data) // num_batches
        for i in range(0, len(scaled_data), batch_size):
            y_pred = np.append(y_pred, self.nn_r.predict(scaled_data[i:i + batch_size]), axis=0)
        mean_pred = y_pred[:, :len(self.y_cols)]
        log_var_pred = y_pred[:, len(self.y_cols):]
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
            tiles_df = Data.merge_dfs(df_data[self.y_cols + ['datetime'] + support_variables], y_pred)
            self.save_predictions_plots(tiles_df, start, end, self.params)
        return df_ori, y_pred
    