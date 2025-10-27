'''
This module contains the implementation of the FOCuS algorithm for change point detection.
'''
import os
from math import log
import multiprocessing
from datetime import timedelta
from tqdm import tqdm
import numpy as np
import pandas as pd

from .plotter import Plotter
from .utils import Data, Logger, logger_decorator

now = pd.Timestamp.now()
DATE_FOLDER = pd.Timestamp.date(now).strftime('%Y-%m-%d')
TIME_FOLDER = pd.Timestamp.time(now).strftime('%H%M')

dir_path = os.environ.get('TSLIES_DIR')
if dir_path is None:
    raise ValueError("TSLIES_DIR not set")

RESULTS_FOLDER_NAME = os.path.join(dir_path, 'results', DATE_FOLDER)
TRIGGER_FOLDER_NAME = os.path.join(RESULTS_FOLDER_NAME, 'anomalies')
PLOT_TRIGGER_FOLDER_NAME = os.path.join(TRIGGER_FOLDER_NAME, TIME_FOLDER, 'plots')



class Curve:
    '''
    From the original python implementation of
    FOCuS Poisson by Kester Ward (2021). All rights reserved.
    '''

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

    def __init__(self, tiles_df, y_cols, y_cols_pred, y_cols_raw, units, latex_y_cols):
        self.tiles_df = tiles_df
        self.y_cols = y_cols
        self.y_cols_raw = y_cols_raw
        self.y_cols_pred = y_cols_pred
        self.units = units
        self.latex_y_cols = latex_y_cols

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
        '''
        From the original python implementation of
        FOCuS Poisson by Kester Ward (2021). All rights reserved.
        '''
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

    def trigger_face_z_score(self, signal, face, reset):
        '''
        Calculates the z-score of the signal and the offset of the change point.
        '''
        result = {f'{face}_significance': signal, f'{face}_offset': signal*0}
        return result

    def trigger_gauss_focus(self, signal, face, reset):
        '''
        From the original python implementation of
        FOCuS Poisson by Kester Ward (2021). All rights reserved.
        '''
        result = {f'{face}_offset': [], f'{face}_significance': []}
        curve_list = []
        for T in tqdm(range(len(signal)), desc=face):
            if reset[T]:
                curve_list = []
            curve_list, global_max, offset = self.focus_step_quad(curve_list, signal[T], 0.95)
            result[f'{face}_offset'].append(offset)
            result[f'{face}_significance'].append(global_max)
        result[f'{face}_significance'] = np.sqrt(2 * np.array(result[f'{face}_significance']))
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
    def run(self, thresholds: dict, type='z_score', save_anomalies_plots=True, support_vars=[], file='', catalog=None, use_multiprocessing=True):
        '''Run the trigger algorithm on the dataset.

        Args:
            `tiles_df` (pd.DataFrame): dataframe containing the data
            `y_cols` (list): list of columns to be used for the trigger
            `y_pred_cols` (list): list of columns containing the predictions
            `thresholds` (dict): thresholds dictionary for each signal

        Returns:
            dict: dict containing the anomalies
        '''
        if not os.path.exists(TRIGGER_FOLDER_NAME):
            os.makedirs(TRIGGER_FOLDER_NAME)
        if not os.path.exists(PLOT_TRIGGER_FOLDER_NAME):
            os.makedirs(PLOT_TRIGGER_FOLDER_NAME)

        triggs_dict = {}
        diff = self.tiles_df['MET'].diff()
        reset = diff > 60
        triggerer = self.trigger_face_z_score if type.lower() == 'z_score' else self.trigger_gauss_focus
        if use_multiprocessing:
            pool = multiprocessing.Pool(5)
            results = []
            
            for face, face_pred in zip(self.y_cols, self.y_cols_pred):
                signal = (self.tiles_df[face] - self.tiles_df[face_pred]) / self.tiles_df[f'{face}_std']
                result = pool.apply_async(triggerer, (signal.values, face, reset.values))
                results.append(result)

            for result in results:
                triggs_dict.update(result.get())
            pool.close()
            pool.join()
        else:
            for face, face_pred in zip(self.y_cols, self.y_cols_pred):
                signal = (self.tiles_df[face] - self.tiles_df[face_pred]) / self.tiles_df[f'{face}_std']
                result = triggerer(signal.values, face, reset.values)
                triggs_dict.update(result)

        triggs_df = pd.DataFrame(triggs_dict)
        triggs_df['datetime'] = self.tiles_df['datetime']
        return_df = triggs_df.copy()
        
        for face in self.y_cols:
            triggs_df[f'{face}_triggered'] = triggs_df[f'{face}_significance'] > thresholds[face]

        mask = triggs_df[[f'{face}_triggered' for face in self.y_cols]].any(axis=1)
        triggs_df = triggs_df[mask]

        anomalies_faces = {face: [] for face in self.y_cols}
        old_stopping_time = {face: -1 for face in self.y_cols}
        count = 0

        for face in self.y_cols:
            triggs_df[f'new_start_datetime_{face}'] = triggs_df.apply(lambda r: str(r['datetime'] + timedelta(seconds=r[f'{face}_offset'])), axis=1)
        triggs_df['new_stop_datetime'] = triggs_df['datetime'] + timedelta(seconds=1)

        for row in tqdm(triggs_df.itertuples(index=True), desc='Identifying triggers'):
            index = row.Index
            for face in self.y_cols:
                if getattr(row, f'{face}_triggered'):
                    new_start_index = getattr(row, f'{face}_offset') + index
                    new_start_datetime = getattr(row, f'new_start_datetime_{face}')
                    new_stop_index = index + 1
                    new_stop_datetime = row.new_stop_datetime

                    if index == old_stopping_time[face] + 1 or new_start_index <= old_stopping_time[face] + 60 and anomalies_faces[face]:
                        last_anomaly = anomalies_faces[face].pop()
                        new_start_index = last_anomaly[1]
                        new_start_datetime = last_anomaly[3]
                    else:
                        count += 1
                    new_anomaly = (face, new_start_index, new_stop_index, new_start_datetime, new_stop_datetime)
                    anomalies_faces[face].append(new_anomaly)
                    old_stopping_time[face] = new_stop_index
        
        anomalies_list = [anomaly for face_anomalies in anomalies_faces.values() for anomaly in face_anomalies]

        print('Merging triggers...', end=' ')
        merged_anomalies = {}
        for face, start, stopping_time, start_datetime, stop_datetime in anomalies_list:
            if returned := self.is_mergeable(start, merged_anomalies, minutes=1):
                start, old_start = returned
                if start < old_start:
                    merged_anomalies[start] = merged_anomalies[old_start]
                    del merged_anomalies[old_start]
                elif start > old_start:
                    start = old_start
                merged_anomalies[start][face] = {'start_index': start, 'stop_index': stopping_time, 'start_datetime': start_datetime, 'stop_datetime': stop_datetime}
            else:
                merged_anomalies[start] = {face: {'start_index': start, 'stop_index': stopping_time, 'start_datetime': start_datetime, 'stop_datetime': stop_datetime}}
        print(f'{len(merged_anomalies)} anomalies in total.')

        detections_file_path = os.path.join(PLOT_TRIGGER_FOLDER_NAME, f'detections_{file}.csv') if file else os.path.join(PLOT_TRIGGER_FOLDER_NAME, 'detections.csv')
        with open(detections_file_path, 'w') as f:
            f.write("start_datetime,stop_datetime,start_met,stop_met,triggered_faces\n")
            for anomaly_start, anomaly in sorted(merged_anomalies.items(), key=lambda x: int(x[0]), reverse=True):
                anomaly_end = -1
                for face in anomaly.values():
                    if face['stop_index'] > anomaly_end:
                        anomaly_end = face['stop_index']
                    if face['start_index'] < anomaly_start:
                        anomaly_start = face['start_index']
                triggered_faces = [face for face in anomaly.keys()]
                inputs_outputs_df_tmp = self.tiles_df[anomaly_start:anomaly_end + 1]
                # print(self.tiles_df[anomaly_start:anomaly_end + 1][self.y_cols + self.y_cols_pred])
                # print(inputs_outputs_df_tmp[triggered_faces])
                max_indices = inputs_outputs_df_tmp[triggered_faces].idxmax().values[0]
                values_dict = {}
                for face, face_pred in zip(self.y_cols, self.y_cols_pred):
                    values_dict[face] = (inputs_outputs_df_tmp.loc[max_indices, face] - inputs_outputs_df_tmp.loc[max_indices, face_pred])

                # print(self.tiles_df['datetime'][int(anomaly_start)], self.tiles_df['datetime'][int(anomaly_end)], self.compute_direction(values_dict))

                f.write(f"{self.tiles_df['datetime'][int(anomaly_start)]},{self.tiles_df['datetime'][int(anomaly_end)]},{self.tiles_df['MET'][int(anomaly_start)]},{self.tiles_df['MET'][int(anomaly_end)]},{'/'.join(triggered_faces)}\n")

        self.tiles_df = Data.merge_dfs(self.tiles_df[self.y_cols + self.y_cols_pred + support_vars + ['datetime'] + [f'{y_col}_std' for y_col in self.y_cols]], return_df, on_column='datetime')
        if save_anomalies_plots:
            Plotter(df = merged_anomalies).plot_anomalies_in_catalog(type, support_vars, thresholds, self.tiles_df, self.y_cols_raw, self.y_cols_pred, only_in_catalog=True, show=False, units=self.units, latex_y_cols=self.latex_y_cols, detections_file_path=detections_file_path, catalog=catalog)

        return merged_anomalies, return_df
            
    def is_mergeable(self, start: int, merged_anomalies: dict, minutes = 1) -> tuple[int, int]:
        '''Check if the current anomaly is mergeable with any of the previous anomalies.

        Args:
            `start` (int): start time of the current anomaly
            `merged_anomalies` (dict): dict of anomalies

        Returns:
            tuple: (start, anomaly_start) if the anomaly is mergeable, False otherwise
        '''
        for anomaly_start in merged_anomalies:
            if start - 60 * minutes < anomaly_start < start + 60 * minutes:
                return start, anomaly_start
        return False


if __name__ == '__main__':
    print('to be implemented')
