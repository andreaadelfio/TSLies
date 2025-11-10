
import os
import gc
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error as MAE, median_absolute_error as MeAE
# Keras
from keras.optimizers import Adam, Nadam, RMSprop, SGD # pylint: disable=E0401
from keras import Input, Model
from keras.layers import Dense, Dropout, BatchNormalization # pylint: disable=E0401
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler # pylint: disable=E0401
from keras.models import load_model # pylint: disable=E0401
from keras.utils import plot_model # pylint: disable=E0401
# ACDAnomalies modules
from ..utils import Logger, logger_decorator, File, Data
from .mlobject import MLObject


class FFNNPredictor(MLObject):
    """Feed-forward neural network predictor built on the base MLObject utilities."""
    logger = Logger('FFNNPredictor').get_logger()

    @logger_decorator(logger)
    def __init__(self, df_data, y_cols, x_cols, y_pred_cols=None, latex_y_cols=None, with_generator=False):
        """
        Initialise the feed-forward predictor with training data and metadata.

        Parameters
        ----------
        - df_data (pd.DataFrame): Data frame containing features, targets, and auxiliary columns.
        - y_cols (List[str]): Target column names to forecast.
        - x_cols (List[str]): Feature column names used as network inputs.
        - y_pred_cols (Optional[List[str]]): Optional names reserved for prediction outputs.
        - latex_y_cols (Optional[List[str]]): Optional LaTeX-formatted labels for plotting.
        - with_generator (bool): Toggle for generator-based training (not implemented).

        Raises
        ------
        - ValueError: Bubbled from ``MLObject`` if mandatory data are missing.
        """
        super().__init__(df_data, y_cols, x_cols, y_pred_cols, latex_y_cols, with_generator)

    @logger_decorator(logger)
    def create_model(self):
        """
        Assemble and compile the dense network, including optional norm/dropout layers.

        Parameters
        ----------
        - None

        Raises
        ------
        - ValueError: If optimiser selection fails due to an unknown ``opt_name``.
        """
        inputs = Input(shape=(self.X_train.shape[1],))
        for count, units in enumerate(list(self.units_for_layers)):
            hidden = Dense(units, activation='relu')(inputs if count == 0 else hidden)
            if self.norm:
                hidden = BatchNormalization()(hidden)
            if self.drop:
                hidden = Dropout(self.do)(hidden)
        outputs = Dense(len(self.y_cols), activation='linear')(hidden)

        self.nn_r = Model(inputs=[inputs], outputs=outputs)
        plot_model(self.nn_r, to_file=os.path.join(os.path.dirname(self.model_path), 'schema.png'),
                   show_shapes=True, show_layer_names=True, rankdir='TB')

        opt = None
        if self.opt_name == 'Adam':
            opt = Adam(beta_1=0.9, beta_2=0.99, epsilon=1e-07)
        elif self.opt_name == 'Nadam':
            opt = Nadam(beta_1=0.9, beta_2=0.99, epsilon=1e-07)
        elif self.opt_name == 'RMSprop':
            opt = RMSprop(rho=0.6, momentum=0.0, epsilon=1e-07)
        elif self.opt_name == 'SGD':
            opt = SGD()

        self.nn_r.compile(loss=self.loss_type, optimizer=opt, metrics=['mae'])

    @logger_decorator(logger)
    def train(self):
        """
        Train the feed-forward network and persist artefacts and performance metrics.

        Parameters
        ----------
        - None

        Returns
        -------
        - tf.keras.callbacks.History: Training history with loss/metric traces.

        Raises
        ------
        - NotImplementedError: When generator-based training is requested.
        """
        if self.with_generator:
            raise NotImplementedError('With generator not implemented yet for ABNNPredictor')
        else:
            history = self.nn_r.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.bs,
                            validation_split=0.3, callbacks=self.callbacks)
        
        nn_r = load_model(self.model_path)

        pred_train = nn_r.predict(self.X_train)
        pred_test = nn_r.predict(self.X_test)
        idx = 0
        text = ''
        for col in self.y_cols:
            mae_tr = MAE(self.y_train.iloc[:, idx], pred_train[:, idx])
            self.mae_tr_list.append(mae_tr)
            mae_te = MAE(self.y_test.iloc[:, idx], pred_test[:, idx])
            diff_i = (self.y_test.iloc[:, idx] - pred_test[:, idx])
            mean_diff_i = (diff_i).mean()
            meae_tr = MeAE(self.y_train.iloc[:, idx], pred_train[:, idx])
            meae_te = MeAE(self.y_test.iloc[:, idx], pred_test[:, idx])
            median_diff_i = (diff_i).median()
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
    def predict(self, start:str|int = 0, end:str|int = -1, mask_column='index', write_bkg=True, write_frg=False, num_batches=1, save_predictions_plot=False, support_variables=[]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate deterministic predictions and optionally persist artefacts and plots.

        Parameters
        ----------
        - start (Union[str, int]): Starting index or timestamp for slicing data.
        - end (Union[str, int]): Ending index or timestamp for slicing data.
        - mask_column (str): Column used to filter the data frame between ``start`` and ``end``.
        - write_bkg (bool): Write background predictions to the model directory.
        - write_frg (bool): Write foreground targets alongside predictions.
        - num_batches (int): Number of batches used to split inference.
        - save_predictions_plot (bool): Save prediction plots via ``Plotter`` when ``True``.
        - support_variables (List[str]): Additional columns merged into the plotting data frame.

        Returns
        -------
        - Tuple[pd.DataFrame, pd.DataFrame]: Original targets and predicted series.: An empty slice returns empty data frames instead of raising.
        """
        df_data = Data.get_masked_dataframe(data=self.df_data, start=start, stop=end, column=mask_column, reset_index=False)
        if df_data.empty:
            return pd.DataFrame(), pd.DataFrame()
        scaled_data = self.scaler_x.transform(df_data[self.x_cols])
        if num_batches > 1:
            pred_x_tot = np.array([])
            batch_size = len(scaled_data)//num_batches
            for i in range(0, len(scaled_data), batch_size):
                pred_x_tot = np.append(pred_x_tot, self.nn_r.predict(scaled_data[i:i + batch_size]))
        else:
            pred_x_tot = self.nn_r.predict(scaled_data)
        pred_x_tot = self.scaler_y.inverse_transform(pred_x_tot)
        y_pred = pd.DataFrame(pred_x_tot, columns=self.y_cols)
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
            tiles_df = Data.merge_dfs(df_data[['Xpos_middle'] + ['datetime'] + support_variables], y_pred)
            self.save_predictions_plots(tiles_df, start, end, self.params)
        return df_ori, y_pred