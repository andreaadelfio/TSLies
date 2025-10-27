import os
from tslies.config import set_dir

set_dir('tslies/example')

from datetime import datetime

now = datetime.now()
DATE_FOLDER = now.strftime('%Y-%m-%d')
dir_path = os.environ.get('TSLIES_DIR')
if dir_path is None:
    raise ValueError("TSLIES_DIR not set")

DATA_FOLDER_NAME = os.path.join(dir_path, 'data')
RESULTS_FOLDER_NAME = os.path.join(dir_path, 'results', DATE_FOLDER)
BACKGROUND_PREDICTION_FOLDER_NAME = os.path.join(RESULTS_FOLDER_NAME, 'background_prediction')

from tslies.background.bnnpredictor import BNNPredictor
from tslies.trigger import Trigger
from tslies.utils import Data
from tslies.plotter import Plotter
from tslies.utils import File

from example_config import y_cols, y_cols_raw, y_pred_cols, y_smooth_cols, x_cols, x_cols_excluded, units, latex_y_cols, thresholds

from catalogs import CatalogsReader


def run_bnn(inputs_outputs, y_cols, y_cols_raw, cols_pred, y_smooth_cols, x_cols):
    '''Runs the neural network model'''
    nn = BNNPredictor(inputs_outputs, y_cols, x_cols, y_cols_raw, cols_pred, y_smooth_cols, latex_y_cols, units, False)
    hyperparams_combinations = { # the hyperparams_combinations is meant to fast tests with different settings
        'units_for_layers' : ([90], [90], [90], [70], [50]),
        'epochs' : [6],
        'bs' : [1000],
        'do' : [0.02],
        'norm' : [0],
        'drop' : [0],
        'opt_name' : ['Adam'],
        'lr' : [0.001],
        'metrics' : ['negative_log_likelihood_var+mae_bnn+spectral_loss_bnn'], # found in background/losses.py
        'loss_type' : ['negative_log_likelihood_var+mae_bnn+spectral_loss_bnn'] # found in background/losses.py
    }

    for params in nn.get_hyperparams_combinations(hyperparams_combinations, use_previous=False): # the hyperparams_combinations is meant to fast tests with different settings
        nn.set_hyperparams(params)
        nn.create_model()
        nn.train()
        nn.update_summary()
        Plotter.save(BACKGROUND_PREDICTION_FOLDER_NAME, params)
        break  # solo il primo per esempio
    return nn

def run_trigger_bnn(inputs_outputs_df, y_cols, y_cols_raw, y_cols_pred, x_cols, model_path):
    '''Runs the model'''
    nn = BNNPredictor(inputs_outputs_df, y_cols, x_cols, y_cols_raw, y_cols_pred, y_smooth_cols, latex_y_cols, units)
    nn.set_model(model_path=model_path, compile=False)
    nn.load_scalers()
    # y_pred = File.read_df_from_file('results/2025-03-03/background_prediction/1644/BNNPredictor/0/pk/bkg')
    y_pred = None
    if y_pred is None or len(y_pred) == 0:
        start, end = 0, -1
        batch_size = len(inputs_outputs_df)
        for i in range(0, len(inputs_outputs_df), batch_size):
            _, y_pred = nn.predict(start=i, end=i + batch_size, write_bkg=True, num_batches=1, save_predictions_plot=False)
    tiles_df = Data.merge_dfs(inputs_outputs_df, y_pred)

    for face, face_pred in zip(y_cols, y_pred_cols):
        tiles_df[f'{face}_norm'] = (tiles_df[face] - tiles_df[face_pred]) / tiles_df[f'{face}_std']

    Trigger(tiles_df, y_cols_raw, y_cols_pred, y_cols_raw, units, latex_y_cols).run(thresholds, type='focus', save_anomalies_plots=True, support_vars=['GOES_XRSA_HARD_EARTH_OCCULTED'])


def run_trigger_mean(inputs_outputs_df, y_cols, y_cols_raw, y_cols_pred, x_cols, catalog):
    '''Runs the model'''
    import pandas as pd
    tiles_df = inputs_outputs_df.copy()
    stats = []
    for face in y_cols:
        tiles_df[f'{face}_std'] = tiles_df[face].rolling(window=120, center=True, min_periods=1).std()
        tiles_df[f'{face}_pred'] = tiles_df[face].rolling(window=120, center=True, min_periods=1).mean()

        norm_face = (tiles_df[face] - tiles_df[f'{face}_pred']) / tiles_df[f'{face}_std']
        stats.append({
            'face': face,
            'std': round(norm_face.std(), 3),
            'mean': round(norm_face.mean(), 3)
        })
    stats_df = pd.DataFrame(stats)
    print(stats_df.set_index('face').T.to_string(header=True))
    support_vars = ['GOES_XRSA_HARD_EARTH_OCCULTED']
    Trigger(tiles_df, y_cols, y_cols_pred, y_cols, units, latex_y_cols).run(thresholds, type='FOCUS', save_anomalies_plots=True, support_vars=support_vars, catalog=catalog)



if __name__ == '__main__':
    catalog = CatalogsReader().catalog_df

    x_cols = [col for col in x_cols if col not in x_cols_excluded]
    inputs_outputs_df = File().read_dfs_from_weekly_pk_folder(start=0, stop=1000)
    # nn = run_bnn(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, y_smooth_cols, x_cols)
    # model_path = '/home/andrea-adelfio/OneDrive/Workspace INFN/TSLies/tslies/example/results/2025-10-27/background_prediction/1456/BNNPredictor/0/model.keras'
    # run_trigger_bnn(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, x_cols, model_path)
    run_trigger_mean(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, x_cols, catalog)
