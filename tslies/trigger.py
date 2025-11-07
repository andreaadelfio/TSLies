"""
This module contains the implementation of the FOCuS algorithm for change point detection.
"""
import os
from math import log
import multiprocessing
from datetime import timedelta
from tqdm import tqdm
import numpy as np
import pandas as pd
import bisect


from .config import (
    RESULTS_DIR,
    ANOMALIES_DIR,
    ANOMALIES_TIME_DIR,
    ANOMALIES_PLOTS_DIR,
    require_existing_dir
)
from .plotter import Plotter
from .utils import Data, Logger, logger_decorator

if RESULTS_DIR is None or ANOMALIES_DIR is None or ANOMALIES_PLOTS_DIR is None:
    raise RuntimeError(
        "TSLies output directories are not initialised. Configure the base directory via "
        "tslies.config.set_base_dir(...) or set the TSLIES_DIR environment variable before using tslies.trigger."
    )

TRIGGER_FOLDER_NAME = str(ANOMALIES_DIR)
TRIGGER_TIME_FOLDER_NAME = str(ANOMALIES_TIME_DIR)
PLOT_TRIGGER_FOLDER_NAME = str(ANOMALIES_PLOTS_DIR)



class Curve:
    """
    From the original python implementation of
    FOCuS Poisson by Kester Ward (2021). All rights reserved.
    """

    def __init__(self, k_T, lambda_1, t=0):
        self.a = k_T
        self.b = -lambda_1
        self.t = t

    def __repr__(self):
        return "({:d}, {:.2f}, {:d})".format(self.a, self.b, self.t)

    def evaluate(self, mu):
        return max(self.a * log(mu) + self.b * (mu - 1), 0)

    def update(self, k_T, lambda_1):
        return Curve(self.a + k_T, -self.b + lambda_1, self.t - 1)

    def ymax(self):
        return self.evaluate(self.xmax())

    def xmax(self):
        return -self.a / self.b

    def is_negative(self):
        # returns true if slope at mu=1 is negative (i.e. no evidence for positive change)
        return (self.a + self.b) <= 0

    def dominates(self, other_curve):
        return (self.a + self.b >= other_curve.a + other_curve.b) and (self.a * other_curve.b <= other_curve.a * self.b)

class Quadratic:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __repr__(self):
        return f'Quadratic: {self.a}x^2+{self.b}x'

    def __sub__(self, other_quadratic):
        #subtraction: needed for quadratic differences
        return Quadratic(self.a-other_quadratic.a, self.b-other_quadratic.b)

    def __add__(self, other_quadratic):
        #addition: needed for quadratic differences
        return Quadratic(self.a+other_quadratic.a, self.b+other_quadratic.b)

    def evaluate(self, mu):
        return np.maximum(self.a*mu**2 + self.b*mu, 0)

    def update(self, X_T, decay_factor=0.8):
        return Quadratic(self.a - 1, self.b * decay_factor + 2 * X_T)

    def ymax(self):
        return -self.b**2/(4*self.a) 

    def xmax(self):
        if (self.a==0)and(self.b==0):
            return 0
        else:
            return -self.b/(2*self.a)

    def dominates(self, other_quadratic):
        return (self.b>other_quadratic.b)and(self.xmax()>other_quadratic.xmax())

class Trigger:
    logger = Logger('Trigger').get_logger()

    def __init__(self, tiles_df, y_cols, y_cols_pred, thresholds=None, trigger_type='z_score', units={}, latex_y_cols={}):
        self.tiles_df = tiles_df
        self.y_cols = y_cols
        self.y_cols_pred = y_cols_pred
        self.units = units
        self.latex_y_cols = latex_y_cols
        self.trigger_type = trigger_type
        self.thresholds = thresholds if thresholds is not None else {y_col: 3.0 for y_col in y_cols}

        self.triggs_dict = {}
        self.merged_anomalies = {}
        self.mask = None

    def focus_step_quad(self, quadratic_list, X_T, decay_factor=0.99):
        new_quadratic_list = []
        global_max = 0
        time_offset = 0
        
        if not quadratic_list: #list is empty
            
            if X_T <= 0:
                return new_quadratic_list, global_max, time_offset
            else:
                updated_q = Quadratic(-1, 2*X_T)
                new_quadratic_list.append(updated_q)
                global_max = updated_q.ymax()
                time_offset = updated_q.a
                
        else: #list not empty: go through and prune
            
            updated_q = quadratic_list[0].update(X_T, decay_factor) #check leftmost quadratic separately
            if updated_q.b < 0: #our leftmost quadratic is negative i.e. we have no quadratics
                return new_quadratic_list, global_max, time_offset
            else:
                new_quadratic_list.append(updated_q)
                if updated_q.ymax() > global_max:   #we have a new candidate for global maximum
                    global_max = updated_q.ymax()
                    time_offset = updated_q.a

                for q in quadratic_list[1:]+[Quadratic(0, 0)]:#add on new quadratic to end of list
                    updated_q = q.update(X_T)

                    if new_quadratic_list[-1].dominates(updated_q):
                        break #quadratic q and all quadratics to the right of it are pruned out by q's left neighbour
                    else:
                        new_quadratic_list.append(updated_q)

                        if updated_q.ymax() > global_max:   #we have a new candidate for global maximum
                            global_max = updated_q.ymax()
                            time_offset = updated_q.a
            
        return new_quadratic_list, global_max, time_offset


    def focus_step_curve(self, curve_list, k_T, lambda_1):
        """
        From the original python implementation of
        FOCuS Poisson by Kester Ward (2021). All rights reserved.
        """
        if not curve_list:  # list is empty
            if k_T <= lambda_1:
                return [], 0., 0
            else:
                updated_c = Curve(k_T, lambda_1, t=-1)
                return [updated_c], updated_c.ymax(), updated_c.t

        else:  # list not empty: go through and prune

            updated_c = curve_list[0].update(k_T, lambda_1)  # check leftmost quadratic separately
            if updated_c.is_negative():  # our leftmost quadratic is negative i.e. we have no quadratics
                return [], 0., 0,
            else:
                new_curve_list = [updated_c]
                global_max = updated_c.ymax()
                time_offset = updated_c.t

                for c in curve_list[1:] + [Curve(0, 0)]:  # add on new quadratic to end of list
                    updated_c = c.update(k_T, lambda_1)
                    if new_curve_list[-1].dominates(updated_c):
                        break
                    else:
                        new_curve_list.append(updated_c)
                        ymax = updated_c.ymax()
                        if ymax > global_max:  # we have a new candidate for global maximum
                            global_max = ymax
                            time_offset = updated_c.t

        return new_curve_list, global_max, time_offset

    def trigger_face_z_score(self, signal, face, reset_indices, threshold):
        """
        Calculates the z-score of the signal and the offset of the change point.
        """
        result = {f'{face}_triggered': signal > threshold, f'{face}_offset': signal*0}
        return result

    def trigger_gauss_focus(self, signal, face, reset_indices, threshold):
        """
        From the original python implementation of
        FOCuS Poisson by Kester Ward (2021). All rights reserved.
        """
        result = {f'{face}_offset': [], f'{face}_triggered': [], f'{face}_significance': []}
        curve_list = []
        
        start = 0
        for end in tqdm(reset_indices, desc=face):
            for value in signal[start:end]:
                curve_list, global_max, offset = self.focus_step_quad(curve_list, value, 0.95)
                result[f'{face}_offset'].append(offset)
                result[f'{face}_significance'].append(global_max)
            curve_list = []
            start = end

        result[f'{face}_significance'] = np.sqrt(2 * np.array(result[f'{face}_significance']))
        result[f'{face}_triggered'] = result[f'{face}_significance'] > threshold
        return result

    def compute_direction(self, values_dict):

        orig_max_values = pd.Series(values_dict)[['Xpos_middle', 'Xneg_middle', 'Ypos_middle', 'Yneg_middle', 'top_middle']]
        max_values = orig_max_values.clip(lower=0)
        if max_values['Xpos_middle'] >= max_values['Xneg_middle']:
            vx = max_values['Xpos_middle']
        else:
            vx = -max_values['Xneg_middle']

        if max_values['Ypos_middle'] >= max_values['Yneg_middle']:
            vy = max_values['Ypos_middle']
        else:
            vy = -max_values['Yneg_middle']
        vz = max_values['top_middle']
        norm = np.sqrt(vx*vx + vy*vy + vz*vz)
        if norm == 0:
            print('max_values:', orig_max_values)
            print('vx, vy, vz:', vx, vy, vz)
        ux = vx / norm
        uy = vy / norm
        uz = vz / norm

        theta = np.arccos(uz)                      # angolo da +Z
        phi = np.arctan2(uy, ux) % (2*np.pi)     # azimutale nel piano XY

        return {
            'theta_deg': np.degrees(theta),
            'phi_deg': np.degrees(phi)
        }

    @logger_decorator(logger)
    def run(self, reset_condition=None, use_multiprocessing=True):
        """Run the trigger algorithm on the dataset.

        Args:
            `tiles_df` (pd.DataFrame): dataframe containing the data
            `y_cols` (list): list of columns to be used for the trigger
            `y_pred_cols` (list): list of columns containing the predictions

        Returns
            dict: dict containing the anomalies
        """
        if reset_condition is None:
            reset_condition = np.zeros(len(self.tiles_df), dtype=bool)
        reset_indices = np.where(reset_condition)[0]
        reset_indices = np.append(reset_indices, len(reset_condition))

        triggerer = self.trigger_face_z_score if self.trigger_type.lower() == 'z_score' else self.trigger_gauss_focus
        if use_multiprocessing:
            pool = multiprocessing.Pool(5)
            results = []
            
            for face, face_pred in zip(self.y_cols, self.y_cols_pred):
                signal = (self.tiles_df[face] - self.tiles_df[face_pred]) / self.tiles_df[f'{face}_std']
                result = pool.apply_async(triggerer, (signal.values, face, reset_indices, self.thresholds[face]))
                results.append(result)

            for result in results:
                self.triggs_dict.update(result.get())
            pool.close()
            pool.join()
        else:
            for face, face_pred in zip(self.y_cols, self.y_cols_pred):
                signal = (self.tiles_df[face] - self.tiles_df[face_pred]) / self.tiles_df[f'{face}_std']
                result = triggerer(signal.values, face, reset_indices, self.thresholds[face])
                self.triggs_dict.update(result)

        triggered_cols = [f'{face}_triggered' for face in self.y_cols]
        triggered_arrays = [np.array(self.triggs_dict[col]) for col in triggered_cols]
        self.mask = np.any(triggered_arrays, axis=0)
        self.tiles_df['anomaly'] = self.mask.astype(int)
        return self.tiles_df

    def identify_and_merge_triggers(self, merge_interval=3):
        triggs_df = pd.DataFrame(self.triggs_dict)
        triggs_df['datetime'] = self.tiles_df['datetime']
        self.return_df = triggs_df.copy()
        triggs_df = triggs_df[self.mask]

        anomalies_faces = {face: [] for face in self.y_cols}
        old_stopping_time = {face: -1 for face in self.y_cols}

        if triggs_df.empty:
            self.logger.info('No triggers detected.')
            return {}, self.return_df

        for face in self.y_cols:
            triggs_df[f'new_start_datetime_{face}'] = triggs_df.apply(lambda r: str(r['datetime'] + timedelta(seconds=r[f'{face}_offset'])), axis=1)
        triggs_df['new_stop_datetime'] = triggs_df['datetime'] + timedelta(seconds=1)

        for row in tqdm(triggs_df.itertuples(index=True), total=len(triggs_df), desc='Identifying triggers'):
            index = row.Index
            for face in self.y_cols:
                if getattr(row, f'{face}_triggered'):
                    new_start_index = getattr(row, f'{face}_offset') + index
                    new_start_datetime = getattr(row, f'new_start_datetime_{face}')
                    new_stop_index = index + 1
                    new_stop_datetime = row.new_stop_datetime

                    if index == old_stopping_time[face] + 1 or new_start_index <= old_stopping_time[face] + merge_interval and anomalies_faces[face]:
                        last_anomaly = anomalies_faces[face].pop()
                        new_start_index = last_anomaly[1]
                        new_start_datetime = last_anomaly[3]
                    new_anomaly = (face, new_start_index, new_stop_index, new_start_datetime, new_stop_datetime)
                    anomalies_faces[face].append(new_anomaly)
                    old_stopping_time[face] = new_stop_index
            
        merged_starts = []
        anomalies_list = [anomaly for face_anomalies in anomalies_faces.values() for anomaly in face_anomalies]
        anomalies_list.sort(key=lambda x: x[1])
        print(f'Merging {len(anomalies_list)} triggers...', end=' ')
        for face, start, stopping_time, start_datetime, stop_datetime in anomalies_list:
            if returned := self.is_mergeable(start, merged_starts, tolerance=merge_interval):
                start, old_start = returned
                if start < old_start:
                    self.merged_anomalies[start] = self.merged_anomalies[old_start]
                    del self.merged_anomalies[old_start]
                elif start > old_start:
                    start = old_start
                self.merged_anomalies[start][face] = {'start_index': start, 'stop_index': stopping_time, 'start_datetime': start_datetime, 'stop_datetime': stop_datetime}
                if start not in merged_starts:
                    bisect.insort(merged_starts, start)
                if old_start in merged_starts and old_start != start:
                    merged_starts.remove(old_start)
            else:
                self.merged_anomalies[start] = {face: {'start_index': start, 'stop_index': stopping_time, 'start_datetime': start_datetime, 'stop_datetime': stop_datetime}}
                bisect.insort(merged_starts, start)

        print(f'{len(self.merged_anomalies)} anomalies in total.')
        return self.merged_anomalies, self.return_df
    
    def get_detections_df(self, cols=[]) -> pd.DataFrame:
        detections = {}
        default_cols = ['datetime', 'timestamp']
        first_anomaly = next(iter(self.merged_anomalies.values()), {})
        first_face = next(iter(first_anomaly.values()), {})
        keys = [def_col for key in first_face.keys() for def_col in default_cols if def_col in key]
        keys = set(keys + cols)
        for key in keys:
            detections[f'start_{key}'] = []
            detections[f'stop_{key}'] = []
        detections['triggered_faces'] = []

        for _, anomaly in sorted(self.merged_anomalies.items(), key=lambda x: int(x[0]), reverse=True):
            start_idx = int(min(face['start_index'] for face in anomaly.values()))
            end_idx = int(max(face['stop_index'] for face in anomaly.values()))
            triggered_faces = list(anomaly.keys())

            start_row = self.tiles_df.iloc[start_idx]
            end_row = self.tiles_df.iloc[end_idx]

            detections['triggered_faces'].append('/'.join(triggered_faces))
            for key in keys:
                detections[f'start_{key}'].append(start_row[key])
                detections[f'stop_{key}'].append(end_row[key])

        return pd.DataFrame(detections)

    def save_detections_csv(self, detections_df: pd.DataFrame, file='', suffix=''):
        require_existing_dir(TRIGGER_TIME_FOLDER_NAME)
        file = f'_{file}' if file else ''

        detections_file_path = os.path.join(TRIGGER_TIME_FOLDER_NAME, f'detections{file}.csv')
        detections_file_path = detections_file_path.replace('.csv', f'{suffix}.csv')
        detections_df.to_csv(detections_file_path, index=False)

    def filter_from_catalog(self, catalog : pd.DataFrame, merged_anomalies=None, detections_df : pd.DataFrame=None) -> tuple[pd.DataFrame, dict]:
        if catalog is None or catalog.empty:
            raise ValueError("Catalog parameter is None or empty.")
        if detections_df is None:
            detections_df = self.get_detections_df()

        catalog_times = catalog['TIME'].to_numpy(dtype='datetime64[ns]')
        catalog_end_times = catalog['END_TIME'].to_numpy(dtype='datetime64[ns]')
        starts = detections_df['start_datetime'].to_numpy(dtype='datetime64[ns]')
        stops = detections_df['stop_datetime'].to_numpy(dtype='datetime64[ns]')
        
        match_matrix = (
            ((starts[:, None] >= catalog_times) & (starts[:, None] <= catalog_end_times))
            | ((stops[:, None] >= catalog_times) & (stops[:, None] <= catalog_end_times))
        )

        matches = [
            catalog.iloc[idx].to_dict('records') if idx.size else np.nan
            for idx in (np.flatnonzero(match_matrix_row) for match_matrix_row in match_matrix)
        ]

        detections_df = detections_df.copy()
        detections_df['catalog_triggers'] = matches
        detections_df.dropna(subset=['catalog_triggers'], inplace=True)
        detections_df.reset_index(drop=True, inplace=True)

        results_dict = {}
        for an_time, anomalies in merged_anomalies.items():
            start_idx = min(face['start_index'] for face in anomalies.values())
            detection_time = self.tiles_df.loc[start_idx, 'datetime']
            match = detections_df.loc[detections_df['start_datetime'] == detection_time]
            if not match.empty and match['catalog_triggers'].notna().iloc[0]:
                results_dict[an_time] = dict(anomalies)
                for key in results_dict[an_time].keys():
                    results_dict[an_time][key]['catalog_triggers'] = match['catalog_triggers'].iloc[0]

        return detections_df, results_dict

    def plot_anomalies(self, merged_anomalies=None, return_df=None, support_vars=[], show=False):
        if merged_anomalies is None:
            merged_anomalies = self.merged_anomalies
        if return_df is None:
            return_df = self.return_df
        tiles_df = Data.merge_dfs(self.tiles_df[self.y_cols + self.y_cols_pred + support_vars + ['datetime'] + [f'{y_col}_std' for y_col in self.y_cols]], return_df, on_column='datetime')
        Plotter(df = merged_anomalies).plot_anomalies(self.trigger_type, support_vars, self.thresholds, tiles_df, self.y_cols, self.y_cols_pred, show=show, units=self.units, latex_y_cols=self.latex_y_cols)

    def is_mergeable(self, start: int, merged_starts: list, tolerance=60) -> tuple[int, int]:
        """Check if mergeable using binary search on sorted list."""
        left = bisect.bisect_left(merged_starts, start - tolerance)
        right = bisect.bisect_right(merged_starts, start + tolerance)
        for i in range(left, right):
            anomaly_start = merged_starts[i]
            if (start - tolerance) < anomaly_start < (start + tolerance):  # Usa included per controllo preciso
                return start, anomaly_start
        return False

if __name__ == '__main__':
    print('to be implemented')
