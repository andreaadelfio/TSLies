import os
import pickle
import itertools
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Union, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow.keras as tf_keras
from tensorflow.keras.models import load_model

from ..config import (
    RESULTS_DIR,
    BACKGROUND_PREDICTION_DIR,
    require_base_dir,
    require_existing_dir
)
from ..plotter import Plotter
from .losses import CustomLosses

BASE_DIR_PATH = require_base_dir()

if RESULTS_DIR is None or BACKGROUND_PREDICTION_DIR is None:
    raise RuntimeError(
        "TSLies output directories are not initialised. Configure the base directory via "
        "tslies.config.set_base_dir(...) or set the TSLIES_DIR environment variable before using ML components."
    )

BACKGROUND_PREDICTION_FOLDER_NAME = str(BACKGROUND_PREDICTION_DIR)
DIR = str(BASE_DIR_PATH)

class MLObject(ABC):
    """Abstract base class defining a common interface for background models."""

    def __init__(
        self,
        df_data: pd.DataFrame,
        y_cols: List[str],
        x_cols: List[str],
        y_pred_cols: Optional[List[str]] = None,
        latex_y_cols: Optional[List[str]] = None,
        units: Optional[List[int]] = None,
        with_generator: bool = False,
    ):
        """
        Initialise shared state for supervised background forecasting models.

        Parameters
        ----------
        - df_data (pd.DataFrame): Data frame containing both features and targets.
        - y_cols (List[str]): Column names of the target variables.
        - x_cols (List[str]): Column names of the input features.
        - y_pred_cols (Optional[List[str]]): Names reserved for model predictions.
        - latex_y_cols (Optional[List[str]]): LaTeX-friendly labels for plotting.
        - units (Optional[List[int]]): Optional units associated with each target.
        - with_generator (bool): Whether training relies on a data generator pipeline.

        Raises
        ------
        - ValueError: If ``df_data`` is ``None`` when ``with_generator`` is False.
        """
        self.model_name = self.__class__.__name__
        self.training_date = pd.Timestamp.time(pd.Timestamp.now()).strftime('%H%M')
        print(f'{self.model_name} - {self.training_date}')
        self.with_generator = with_generator
        self.y_cols = y_cols
        self.x_cols = x_cols
        print(f'x_cols: {x_cols}')
        print(f'y_cols: {y_cols}')
        self.y_pred_cols = y_pred_cols or [col + '_pred' for col in y_cols]
        self.latex_y_cols = latex_y_cols or y_cols
        self.units = units or {}

        if with_generator: # if the data is too large, use tensorflow DataGenerator
            # to be implemented
            pass
        else:
            self.df_data: pd.DataFrame = df_data
            if self.df_data is None:
                raise ValueError("df_data cannot be None. Please check data loading in the calling script.")
        self.y = self.df_data[self.y_cols].astype('float32')
        self.X = self.df_data[self.x_cols].astype('float32')
        self.scalers_params_dict = self.set_scalers(self.X, self.y)
        self.X = self.scaler_x.transform(self.X.values)
        self.y = self.scaler_y.transform(self.y.values)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=42, shuffle=True)

        self.closses = CustomLosses({'mae': 1,
                                     'mse': 1})

        self.model_path = os.path.join(BACKGROUND_PREDICTION_FOLDER_NAME, self.training_date, self.model_name)
        require_existing_dir(self.model_path)
        self.model_params_path = os.path.join(self.model_path, 'models_params.csv')
        self.nn_r = None
        self.text = None
        self.model_id = None
        self.params = {}
        self.norm = None
        self.drop = None
        self.units_for_layers = None
        self.bs = None
        self.do = None
        self.opt_name = None
        self.lr = None
        self.loss_type = None
        self.loss = None
        self.metrics = None
        self.epochs = None
        self.mae_tr_list = None
        self.callbacks = None

    def get_hyperparams_combinations(self, hyperparams_combinations:dict, use_previous:bool=False) -> list:
        """
        Deduplicate hyperparameter grids by removing repeated layer configurations.

        Parameters
        ----------
        - hyperparams_combinations (dict): Search-space specification for each parameter.
        - use_previous (bool): Reuse existing runs to skip already trained combinations.

        Returns
        -------
        - list[dict]: Unique hyperparameter dictionaries ready for training.
        """
        hyperparams_combinations_tmp = []
        uniques = set()
        model_id = 0
        first = 0
        if not use_previous:
            if os.path.exists(self.model_params_path):
                os.remove(self.model_params_path)
        else:
            first = int(sorted(os.listdir(self.model_path), key=(lambda x: int(x) if len(x) < 4 else 0))[-1])

        units_for_layers = list(itertools.product(*hyperparams_combinations['units_for_layers']))
        other_params = [hyperparams_combinations[key] for key in hyperparams_combinations if key != 'units_for_layers']
        all_combinations = list(itertools.product(units_for_layers, *other_params))

        for combination in all_combinations:
            sorted_tuple = tuple([value for value in combination[0] if value and value > 0] + list(combination[1:]))
            if len(sorted_tuple) == 8 or sorted_tuple in uniques:
                continue
            else:
                uniques.add(sorted_tuple)
                hyperparams_dict = {key: value for key, value in zip(hyperparams_combinations.keys(), combination)}
                hyperparams_dict['model_id'] = model_id
                hyperparams_combinations_tmp.append(hyperparams_dict)
                model_id += 1

        return hyperparams_combinations_tmp[int(first):]

    def set_hyperparams(self, params, use_previous=False):
        """
        Persist the selected hyperparameters and prepare the model directory.

        Parameters
        ----------
        - params (dict): Hyperparameter dictionary containing architecture and training keys.
        - use_previous (bool): Skip resetting artefacts when resuming existing runs.

        Raises
        ------
        - OSError: If the model directory cannot be created.

        Examples
        --------
        >>> params = {'model_id': 0, 'units_for_layers': (128, 128), 'norm': True,
        ...           'drop': True, 'epochs': 100, 'bs': 32, 'do': 0.5,
        ...           'opt_name': 'Adam', 'lr': 0.001, 'loss_type': 'mse', 'metrics': 'mae'}
        >>> ml_obj.set_hyperparams(params)
        """
        self.params = params
        self.params['model_name'] = self.model_name
        self.params['training_date'] = self.training_date
        self.params['x_cols'] = self.x_cols
        self.params['y_cols'] = self.y_cols
        self.model_id = params['model_id']
        self.model_path = os.path.join(BACKGROUND_PREDICTION_FOLDER_NAME, self.training_date, self.model_name, str(self.model_id))
        require_existing_dir(self.model_path)
        self.params['model_path'] = self.model_path = os.path.join(self.model_path, 'model.keras')
        # # self.params['dataframe_path'] = DATA_LATACD_PROCESSED_FOLDER_NAME

        max_len = max(map(len, self.params.keys()))
        if not use_previous:
            if os.path.exists(self.model_params_path):
                os.remove(self.model_params_path)
            with open(self.model_params_path, 'w') as f:
                f.write('\t'.join(list(self.params.keys()) + self.y_cols) + '\n')
        with open(os.path.join(os.path.dirname(self.model_path), 'params.txt'), "w") as params_file:
            for key, value in self.params.items():
                params_file.write(f'{key:>{max_len}} : {value}\n')
        with open(os.path.join(os.path.dirname(self.model_path), 'scalers.pkl'), 'wb') as f:
            pickle.dump(self.scalers_params_dict, f)
        self.units_for_layers = params['units_for_layers']
        self.bs = params['bs']
        self.do = params['do']
        self.opt_name = params['opt_name']
        self.lr = params['lr']
        self.loss_type = params['loss_type']
        self.epochs = params['epochs']
        self.norm = params['norm']
        self.drop = params['drop']
        self.mae_tr_list = []

        self.set_loss()
        self.set_metrics()

    def set_model(self, model_path: str, compile: bool = True):
        """
        Load an already trained model from disk and attach it to this instance.

        Parameters
        ----------
        - model_path (str): Filesystem path pointing to the serialized model.
        - compile (bool): Whether to compile the model when loading via Keras.

        Returns
        -------
        - tf.keras.Model: Loaded neural network ready for inference.

        Raises
        ------
        - FileNotFoundError: If the specified model file cannot be located.
        """
        if model_path is not None and model_path != '':
            if os.path.exists(model_path):
                self.model_path = model_path
            elif os.path.exists(os.path.join(DIR, model_path)):
                self.model_path = os.path.join(DIR, model_path)
        self.nn_r = load_model(self.model_path, compile=compile)
        self.params['model_path'] = os.path.dirname(self.model_path)
        return self.nn_r

    def set_scalers(self, train_x: pd.DataFrame = None, train_y: pd.DataFrame = None):
        """
        Fit feature and target standard scalers, optionally using provided data.

        Parameters
        ----------
        - train_x (Optional[pd.DataFrame]): Feature matrix used to fit the input scaler.
        - train_y (Optional[pd.DataFrame]): Target matrix used to fit the output scaler.

        Returns
        -------
        - dict: Serialized scaler parameters (means and scales for x/y).

        Raises
        ------
    - KeyError: If ``x_cols`` or ``y_cols`` are missing from the training data frame.
        """
        if train_x is None:
            train_x = self.df_data[self.x_cols]
        if train_y is None:
            train_y = self.df_data[self.y_cols]
        self.scaler_x = StandardScaler()
        self.scaler_x.fit(train_x.values)
        self.scaler_x.feature_names_in_ = None
        self.scaler_y = StandardScaler()
        self.scaler_y.fit(train_y.values)
        self.scaler_y.feature_names_in_ = None
        self.scalers_params_dict = {
            'x_mean': self.scaler_x.mean_,
            'x_scale': self.scaler_x.scale_,
            'y_mean': self.scaler_y.mean_,
            'y_scale': self.scaler_y.scale_
        }
        return self.scalers_params_dict

    def load_scalers(self):
        """
        Restore fitted standard scalers from the persisted ``scalers.pkl`` file.

        Parameters
        ----------
        - None

        Raises
        ------
        - FileNotFoundError: If the scalers pickled file cannot be located.
        """
        scalers_path = os.path.join(os.path.dirname(self.model_path), 'scalers.pkl')
        if not os.path.exists(scalers_path):
            raise FileNotFoundError(f'Scalers file not found: {scalers_path}')
        self.scalers_params_dict = pickle.load(open(scalers_path, 'rb'))
        self.scaler_x = StandardScaler()
        self.scaler_x.mean_ = np.array(self.scalers_params_dict['x_mean'])
        self.scaler_x.scale_ = np.array(self.scalers_params_dict['x_scale'])
        self.scaler_x.feature_names_in_ = None
        self.scaler_y = StandardScaler()
        self.scaler_y.mean_ = np.array(self.scalers_params_dict['y_mean'])
        self.scaler_y.scale_ = np.array(self.scalers_params_dict['y_scale'])
        self.scaler_y.feature_names_in_ = None

    def set_loss(self):
        """
        Configure the training loss by combining requested loss components.

        Parameters
        ----------
        - None

        Raises
        ------
        - KeyError: If a requested loss name is not registered in ``CustomLosses``.
        """
        self.loss_type_list = self.params['loss_type'].split('+')
        loss_functions = self.closses.get_loss_list()

        if len(self.loss_type_list) == 1:
            self.loss = loss_functions[self.loss_type_list[0]]
        else:
            def combined_loss(y_true, y_pred):
                total_loss = 0
                for loss_type in self.loss_type_list:
                    total_loss += loss_functions[loss_type](y_true, y_pred)
                return total_loss
            self.loss = combined_loss

    def set_metrics(self):
        """
        Populate the list of Keras metrics according to configured names.

        Parameters
        ----------
        - None: Unregistered metric identifiers are silently ignored.
        """
        self.metrics_list = self.params['metrics'].split('+')
        metrics_functions = self.closses.get_metrics_list()

        metrics_list = []
        for metric in self.metrics_list:
            if metric in metrics_functions:
                metrics_list.append(metrics_functions[metric])
        self.metrics = metrics_list

    def set_callbacks(self):
        """
        Configure the callbacks collection executed during training.

        Parameters
        ----------
        - None
        """
        # es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.002,
        #                    patience=10, start_from_epoch=190)
        mc = tf_keras.callbacks.ModelCheckpoint(self.model_path,
                             monitor='val_loss', mode='min', verbose=0, save_best_only=True)
        # call_lr = LearningRateScheduler(self.scheduler)
        # custom_callback = self.custom_callback(self, 5)
        self.callbacks = [mc]

    def scheduler(self, epoch, lr_actual):
        """
        Piecewise learning-rate schedule tuned for warm-up and decay phases.

        Parameters
        ----------
        - epoch (int): Current training epoch (zero-indexed).
        - lr_actual (float): Scalar learning rate proposed by the optimizer.

        Returns
        -------
        - float: Learning rate value to use for the upcoming epoch.
        """
        if epoch < 0.06 * self.epochs:
            return self.lr*1.25
        if 0.06 * self.epochs <= epoch < 0.20 * self.epochs:
            return self.lr*0.2
        if epoch >= 0.20 * self.epochs:
            return self.lr/30
        return lr_actual
        
    def update_summary(self):
        """
        Append the latest training metrics to ``models_params.csv`` summary.

        Parameters
        ----------
        - None

        Raises
        ------
        - OSError: If the summary file cannot be opened for appending.
        """
        with open(self.model_params_path, 'a') as f:
            list_tmp = list(self.params.values()) + self.mae_tr_list
            f.write('\t'.join([str(value) for value in list_tmp] + ['\n']))

    class custom_callback(tf_keras.callbacks.Callback):
        """
        Custom Keras callback that tracks history, plots diagnostics, and optionally stops training.

        Parameters
        ----------
        - predictor (MLObject): Background predictor owning the training loop.
        - interval (int): Epoch cadence for generating prediction plots.
        """

        def __init__(self, predictor, interval=5):
            """
            Initialise the callback with a reference model and plotting cadence.

            Parameters
            ----------
            - predictor (MLObject): Background predictor owning the training loop.
            - interval (int): Number of epochs between prediction visualisations.

            Returns
            -------
            - None

            Raises
            ------
            - None
            """
            super().__init__()
            self.predictor = predictor
            self.interval = interval
            self.history = {'history': {'loss': [], 'val_loss': []}}
            for metric in self.predictor.metrics_list:
                self.history['history'][metric] = []
                self.history['history'][f'val_{metric}'] = []

        def _implements_train_batch_hooks(self):
            """
            Provide TensorFlow with the callback hook availability declaration.

            Returns
            -------
            - bool: ``True`` if the callback implements train batch hooks.
            """
            return False

        def _implements_test_batch_hooks(self):
            """
            Provide TensorFlow with the callback hook availability declaration.

            Returns
            -------
            - bool: ``True`` if the callback implements test batch hooks.
            """
            return False

        def _implements_predict_batch_hooks(self):
            """
            Provide TensorFlow with the callback hook availability declaration.

            Returns
            -------
            - bool: ``True`` if the callback implements predict batch hooks.
            """
            return False

        def on_epoch_end(self, epoch, logs={}):
            """
            Persist history, produce diagnostic plots, and execute periodic inference.

            Parameters
            ----------
            - epoch (int): Index of the completed epoch.
            - logs (dict): Metrics dictionary emitted by Keras for the epoch.

            Returns
            -------
            - None

            Raises
            ------
            - KeyError: If mandatory keys are missing when accessing ``logs``.
            """
            for key in logs.keys():
                if not key in self.history['history']:
                    self.history['history'][key] = []
                self.history['history'][key].append(logs[key])
            Plotter().plot_history(self.history)
            Plotter.save(BACKGROUND_PREDICTION_FOLDER_NAME, params={'model_path': self.predictor.params['model_path']})
            if (epoch + 1) % self.interval == 0:
                for start, end in [('2024-03-10 12:08:00', '2024-03-10 12:30:00'),
                           ('2024-03-28 20:50:00', '2024-03-28 21:10:00'),
                           ('2024-05-08 21:00:00', '2024-05-08 23:30:00'),
                           ('2024-05-11 01:00:00', '2024-05-11 02:00:00'),
                           ('2024-05-15 14:15:00', '2024-05-15 15:40:00'),
                           ('2024-06-20 23:00:00', '2024-06-20 23:30:00'), 
                           (str(self.predictor.df_data['datetime'].iloc[10]), str(self.predictor.df_data['datetime'].iloc[4500]))]:
                    self.predictor.predict(start=start, end=end, mask_column='datetime', write_bkg=False, save_predictions_plot=True)

            if logs['loss'] == 0.5:
                self.model.stop_training = True

    def save_predictions_plots(self, tiles_df, start, end, params):
        """
        Save tiled prediction plots for a given temporal window to disk.

        Parameters
        ----------
        - tiles_df (pd.DataFrame): Data frame containing target and prediction columns.
        - start (Union[str, int]): Start timestamp or index used in the plot title.
        - end (Union[str, int]): End timestamp or index used in the plot title.
        - params (dict): Model parameters dictionary containing the ``model_path`` key.

        Raises
        ------
        - KeyError: If ``params`` lacks ``model_path``.
        """
        title = os.path.join(os.path.dirname(params['model_path']), f'tiles_{start}_{end}.png')
        Plotter(df=tiles_df, label=title).df_plot_tiles(self.y_cols, x_col='datetime', latex_y_cols=self.latex_y_cols, init_marker=',',
                                                        show=False, smoothing_key='pred', save=True, show_std=True, units=self.units, figsize=(5, 3))

    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def create_model(self) -> None:
        """
        Build and assign the underlying learning architecture.

        Raises
        ------
        - NotImplementedError: Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def train(self) -> Any:
        """
        Execute the end-to-end training routine.

        Returns
        -------
        - Any: Optional training artefacts supplied by subclasses.

        Raises
        ------
        - NotImplementedError: Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(
        self, 
        start: Union[str, int] = 0, 
        end: Union[str, int] = -1, 
        mask_column: str = 'index',
        write_bkg: bool = True, 
        write_frg: bool = False, 
        num_batches: int = 1, 
        save_predictions_plot: bool = False, 
        support_variables: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Make predictions using a consistent interface across background models.

        Parameters
        ----------
        - start (Union[str, int]): Start index or datetime string delimiting the window.
        - end (Union[str, int]): End index or datetime string delimiting the window.
        - mask_column (str): Column name used to filter the data frame for inference.
        - write_bkg (bool): Persist background predictions to disk when ``True``.
        - write_frg (bool): Persist foreground data to disk when ``True``.
        - num_batches (int): Number of batches to use during inference.
        - save_predictions_plot (bool): Save diagnostic plots when ``True``.
        - support_variables (Optional[List[str]]): Additional columns to show in plots.

        Returns
        -------
        - Tuple[pd.DataFrame, pd.DataFrame]: Original data and model predictions.

        Raises
        ------
        - NotImplementedError: Must be implemented by subclasses.
        """
        pass
