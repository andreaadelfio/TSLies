"""
Plotter module for plotting data points and curves.
"""
import os
import gc
from pathlib import Path
from datetime import timedelta

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .config import (
    RESULTS_DIR,
    BACKGROUND_PREDICTION_DIR,
    ANOMALIES_DIR,
    ANOMALIES_PLOTS_DIR,
)
from .utils import Logger, logger_decorator, Time

if (
    RESULTS_DIR is None
    or BACKGROUND_PREDICTION_DIR is None
    or ANOMALIES_DIR is None
    or ANOMALIES_PLOTS_DIR is None
):
    raise RuntimeError(
        "TSLies output directories are not initialised. Configure the base directory via "
        "tslies.config.set_base_dir(...) or set the TSLIES_DIR environment variable before using tslies.plotter."
    )

RESULTS_FOLDER_NAME = str(RESULTS_DIR)
BACKGROUND_PREDICTION_FOLDER_NAME = str(BACKGROUND_PREDICTION_DIR)
TRIGGER_FOLDER_NAME = str(ANOMALIES_DIR)
PLOT_TRIGGER_FOLDER_NAME = str(ANOMALIES_PLOTS_DIR)


class Plotter:
    """
    This class provides methods for plotting data points and curves.
    """
    logger = Logger('Plotter').get_logger()
    
    @logger_decorator(logger)
    def __init__(self, x = None, y = None, df: pd.DataFrame = None, xy: dict = None, label = '', latex = False):
        """
        Initialize the Plotter object.

        Parameters:
        ----------
            x (list): The x-coordinates of the data points (default: None).
            y (list): The y-coordinates of the data points (default: None).
            df (pd.DataFrame): The y-coordinates of the data points (default: None).
            xy (dict): A dictionary of x, y, and smooth y values for multiple curves (default: None).
            label (str): The label for the plot (default: '').
        """
        self.x = x
        self.y = y
        self.df = df
        self.xy = xy
        self.label = label
        if latex:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "sans.serif",
                "font.sans-serif": ["Helvetica"],
                "font.size": 35
            })

    # @logger_decorator(logger)
    # def plot_tiles(self, marker = '-', lw = 0.2, with_smooth = False, show = True):
    #     """
    #     Plot multiple curves as tiles.

    #     Parameters:
    #     ----------
    #         lw (float): Line width of the curves (default: 0.1).
    #         with_smooth (bool): Whether to plot smoothed curves as well (default: False).
    #     """
    #     i = 0
    #     _, axs = plt.subplots(len(self.xy), 1, sharex=True)
    #     plt.tight_layout(pad = 0.4)
    #     axs[0].set_title(self.label)
    #     for label, xy in self.xy.items():
    #         axs[i].plot(xy[0], xy[1], marker = marker, lw = lw, label=label)
    #         if with_smooth:
    #             axs[i].plot(xy[0], xy[2], marker = marker, label=f'{label} smooth')
    #         axs[i].legend()
    #         axs[i].grid()
    #         axs[i].set_xlim(xy[0][0], xy[0][-1])
    #         i += 1
    #     if show:
    #         plt.show()

    @logger_decorator(logger)
    def df_plot_corr_tiles(self, x_col, excluded_cols = None, marker = '-', ms = 1, lw = 0.1, smoothing_key = 'smooth', show = True):
        """
        Plot multiple curves as tiles.

        Parameters:
        ----------
            lw (float): Line width of the curves (default: 0.1).
            with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        """
        if not excluded_cols:
            excluded_cols = []
        df_columns = [column for column in self.df.columns if f'_{smoothing_key}' not in column and column not in excluded_cols and column != 'datetime']
        n_plots = len(df_columns)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, squeeze=True, figsize=(17, 10), num=self.label)
        plt.tight_layout(pad = 0.4)
        fig.subplots_adjust(bottom = 0.08, hspace = 0.13, wspace = 0.08, left = 0.04)
        if n_plots > 1:
            axs = axs.flatten()
            for i, column in enumerate(df_columns):
                if column == x_col:
                    x = self.df['datetime']
                    axs[i].plot(x, self.df[column], marker = marker, ms = ms, lw = lw, label=column)
                else:
                    x = self.df[x_col]
                    axs[i].plot(self.df[column], x, marker = marker, ms = ms, lw = lw, label=column)
                if smoothing_key != '' and f'{column}_{smoothing_key}' in self.df.columns:
                    axs[i].plot(x, self.df[f'{column}_{smoothing_key}'], marker = '.', ms = 0.2, lw = '0.1', label=f'{column} {smoothing_key}')
                axs[i].legend(loc='upper right')
                axs[i].set_xlabel(column)
                axs[i].set_ylabel(x_col)
                axs[i].grid()
                axs[i].tick_params(axis="x", labelrotation=30)
            for j in range(i + 1, len(axs)):
                axs[j].axis('off')
        else:
            column = df_columns[0]
            axs.plot(self.df[column], x, marker = marker, lw = lw, label=column)
            axs.set_xlabel(column)
            axs.set_ylabel(x_col)
            axs.legend(loc='upper right')
            axs.grid()
            axs.tick_params(axis="x", labelrotation=45)
        if show:
            plt.show()

    @logger_decorator(logger)
    def df_plot_tiles(self, y_cols, x_col, latex_y_cols=None, top_x_col=None, excluded_cols=None, init_marker=',', lw=0.1, smoothing_key='smooth', show=True, save=False, units=None, show_std=True, figsize=(12, 6)):
        """
        Plot multiple curves as tiles.

        Parameters:
        ----------
            lw (float): Line width of the curves (default: 0.1).
            with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        """
        if units is None:
            units = {}
        if not excluded_cols:
            excluded_cols = []
        df_columns = [column for column in self.df.columns if f'_{smoothing_key}' not in column and '_std' not in column and column not in excluded_cols and column != x_col and column != top_x_col]
        n_plots = len(df_columns)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        fig = plt.figure(figsize=figsize, num=self.label)
        x = self.df[x_col]
        axs = [fig.add_subplot(n_rows, n_cols, 1)]
        for i in range(1, n_plots):
            axs.append(fig.add_subplot(n_rows, n_cols, i + 1, sharex=axs[0]))

        for i, column in tqdm(enumerate(df_columns), desc='Plotting tiles'):
            fmt = init_marker
            plot_lw = lw
            if column in y_cols:
                fmt = 'k-.'
                plot_lw = None
            label = f"${latex_y_cols[column]}$" if column in latex_y_cols else f'{column}'
            label += f' [${units[column]}$]' if column in units else ' [NA]'
            axs[i].plot(x.values, self.df[column], fmt, lw=plot_lw, label=label)

            if smoothing_key and f'{column}_{smoothing_key}' in self.df.columns:
                label = f'${latex_y_cols[column]}$ {smoothing_key}' if column in latex_y_cols else f'{column} {smoothing_key}'
                axs[i].plot(x.values, self.df[f'{column}_{smoothing_key}'], 'r-', ms=0.2, label=label)
                if f'{column}_std' in self.df.columns and show_std:
                    axs[i].fill_between(x.values, self.df[f'{column}_{smoothing_key}']-self.df[f'{column}_std'], self.df[f'{column}_{smoothing_key}']+self.df[f'{column}_std'], label='error')
            
            if i >= n_cols * (n_rows - 1) - (n_cols * n_rows - n_plots):
                axs[i].tick_params(axis="x", labelrotation=30)
                plt.setp(axs[i].get_xticklabels(), visible=True)
                offset_text = axs[i].get_xaxis().get_offset_text().get_text()
                xlabel = f'{x_col} ({offset_text})' if offset_text else f'{x_col} ({str(x.iloc[0]).split("+")[0]})'
                axs[i].set_xlabel(xlabel)
            else:
                plt.setp(axs[i].get_xticklabels(), visible=False)
            axs[i].get_xaxis().get_offset_text().set_visible(False)
            
            # axs[i].legend(loc='upper right', fontsize=13)
            axs[i].grid()
            axs[i].set_xlim(x.iloc[0], x.iloc[-1])
            axs[i].tick_params(axis="y", labelrotation=30)
            if top_x_col and top_x_col in self.df.columns and i < n_cols:
                if top_x_col == 'datetime':
                    secax = axs[i].secondary_xaxis('top', functions=(Time.from_met_to_datetime, lambda x: x))
                    secax.set_xlabel(f'{top_x_col} ({self.df[top_x_col].iloc[0]})')
                elif top_x_col == 'MET':
                    secax = axs[i].secondary_xaxis('top', functions=(Time.date2yday, lambda x: x))
                    secax.get_xaxis().get_offset_text().set_visible(False)
                    secax.set_xlabel(f'{top_x_col} ({self.df[top_x_col].iloc[0] / 1e8} $\dot 10^8$)')
                secax.tick_params(axis="x", labelrotation=30)
                # secax.set_xticklabels(labels=secax.xaxis.get_majorticklabels(), ha='left')

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')
        fig.subplots_adjust(hspace=0, right=0.97, left=0.05)
        # plt.tight_layout()
        if show:
            plt.show()

        if save:
            fig.savefig(
                os.path.join(self.label),
                dpi=200, bbox_inches='tight')
            plt.close(fig)
            gc.collect()

    @logger_decorator(logger)
    def plot_tile(self, tiles_df, face='top', col='datetime', smoothing_key = 'smooth', units = {}, support_vars = None, show_std=True, save=False):
        # with sns.plotting_context("talk"):
        fig, axs = plt.subplots(2 + len(support_vars), 1, sharex=True, figsize=(20, 12), num=face)
        # fig.subplots_adjust(hspace=0)
        axs[0].plot(pd.to_datetime(tiles_df[col]), tiles_df[face], 'k-.', label=f'{face}')
        axs[0].plot(pd.to_datetime(tiles_df[col]), tiles_df[f'{face}_{smoothing_key}'], 'r-', label='background')
        axs[0].set_ylabel(f'{units[face] if face in units else "NA"}')

        axs[1].plot(pd.to_datetime(tiles_df[col]), tiles_df[face] - tiles_df[f'{face}_{smoothing_key}'], 'k-.', label='residuals')

        if f'{face}_std' in tiles_df.columns and show_std:
            axs[0].fill_between(pd.to_datetime(tiles_df[col]), tiles_df[f'{face}_{smoothing_key}']-tiles_df[f'{face}_std'], tiles_df[f'{face}_{smoothing_key}']+tiles_df[f'{face}_std'], label='prediction error')
            axs[1].fill_between(pd.to_datetime(tiles_df[col]), -tiles_df[f'{face}_std'], tiles_df[f'{face}_std'])
        
        axs[1].axhline(0, color='red')
        # axs[1].plot(pd.to_datetime(tiles_df[col]).ffill(), [0 for _ in tiles_df[col].ffill()], 'k-')
        axs[1].set_xlim(tiles_df[col].iloc[0], tiles_df[col].iloc[-1])
        plt.xticks(rotation=0)
        axs[1].set_ylabel(f'{units[face] if face in units else "NA"}')
        axs[0].legend()
        axs[1].legend()

        for i in range(len(support_vars)):
            axs[2 + i].plot(pd.to_datetime(tiles_df[col]), tiles_df[support_vars[i]], color='green', label=f'{support_vars[i]}')
            axs[2 + i].set_xlabel(f"{col} ({tiles_df[col].iloc[0]})")
            axs[2 + i].set_ylabel(f"{units[support_vars[i]] if support_vars[i] in units else 'NA'}")
            axs[2 + i].legend()

        plt.show()
        if save:
            fig.savefig(
                os.path.join(PLOT_TRIGGER_FOLDER_NAME, 'Output.png'),
                dpi=200, bbox_inches='tight')

    @logger_decorator(logger)
    def plot_history(self, history):
        for feature in history['history'].keys():
            if 'val' in feature:
                continue
            plt.figure(f"history_{feature}", layout="tight")
            plt.plot(history['history'][feature][1:])
            if f'val_{feature}' in history['history']:
                plt.plot(history['history'][f'val_{feature}'][1:])
            plt.ylabel(feature)
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')

    @logger_decorator(logger)
    def plot_correlation_matrix(self, show = True, save = False):
        """Function to plot the correlation matrix."""
        correlations = self.df.corr()
        plt.figure(figsize=(18, 18), num='correlations_matrix')
        sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        if show:
            plt.show()
        if save:
            Plotter.save(BACKGROUND_PREDICTION_FOLDER_NAME)

    @logger_decorator(logger)
    def plot_confusion_matrix(self, y_true, y_pred, show = True):
        """Function to plot the confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8), num='confusion_matrix')
        sns.heatmap(cm, annot=True, cmap='coolwarm', fmt=".2f")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        if show:
            plt.show()
    
    def read_detections_files(self, directory):
        sorted_df = pd.read_csv(directory)
        sorted_df['start_datetime'] = pd.to_datetime(sorted_df['start_datetime'], utc=False)
        sorted_df['stop_datetime'] = pd.to_datetime(sorted_df['stop_datetime'], utc=False)
        return sorted_df
    
    @logger_decorator(logger)
    def plot_anomalies_in_catalog(self, trigger_algo_type, support_vars, thresholds, tiles_df, y_cols, y_pred_cols, save=True, show=False, extension='png', units={}, latex_y_cols={}, detections_file_path='', catalog=None):
        """Plots the anomalies passed as `df` in Plotter."""
        for an_time, anomalies in tqdm(self.df.items(), desc='Plotting anomalies'):
            faces = list(anomalies.keys())

            anomaly_end = -1
            anomaly_start = tiles_df.index[-1]
            in_catalog = False
            for anomaly in anomalies.values():
                if anomaly['stop_index'] > anomaly_end:
                    anomaly_end = anomaly['stop_index']
                if anomaly['start_index'] < anomaly_start:
                    anomaly_start = anomaly['start_index']
                if 'catalog_triggers' in anomaly:
                    cat_event = anomaly['catalog_triggers'][0]
                    in_catalog = True
            
            num_signal_residual_pairs = len(y_cols)
            num_support_vars = len(support_vars)
            total_rows = 2 * num_signal_residual_pairs + num_support_vars

            fig = plt.figure(figsize=(8.5, 7.5 + 0.37 * total_rows))
            gs0 = GridSpec(num_signal_residual_pairs + num_support_vars, 1, figure=fig, hspace=0.05, left=0.1, right=0.99, top=0.98, bottom=0.04)
            gss = []
            for i in range(num_signal_residual_pairs):
                gss.append(GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[i], hspace=0.))
            gss.append(GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[-1], hspace=0.))

            axs = []
            anomaly_duration = anomaly_end - int(anomaly_start)
            anomaly_delta = max((anomaly_duration) // 5, 120)
            start = max(int(anomaly_start) - anomaly_delta, 0)
            end = min(anomaly_end + anomaly_delta, len(tiles_df))
            start_xlim = tiles_df['datetime'][int(anomaly_start)] - timedelta(seconds=anomaly_delta)
            end_xlim = tiles_df['datetime'][anomaly_end] + timedelta(seconds=anomaly_delta-1)
            tiles_df_subset = tiles_df[start:end]
            if not latex_y_cols:
                latex_y_cols = {col: col for col in faces if col in tiles_df_subset}
            latex_faces = [f'${latex_y_cols[col]}$' for col in faces if col in latex_y_cols]
            i = 0
            for face, face_pred in zip(y_cols, y_pred_cols):
                formatter = mticker.ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((0, 0))
                ax_signal = fig.add_subplot(gss[i][:-1, :])
                axs.append(ax_signal)

                ax_residuals = fig.add_subplot(gss[i][-1, :], sharex=ax_signal)
                axs.append(ax_residuals)


                face_color = "black"
                max_val = max(max(tiles_df_subset[f'{face}_significance']), thresholds[face] * 1.1)
                if face in faces:
                    # face_color = None
                    changepoint = anomalies[face]['stop_index']
                    stopping_time = anomalies[face]['start_index']
                    ax_signal.axvline(tiles_df['datetime'][changepoint], color='red', lw=0.8)
                    ax_signal.axvline(tiles_df['datetime'][stopping_time], color='red', lw=0.8)
                    ax_signal.fill(
                        (tiles_df['datetime'][int(anomaly_start)], tiles_df['datetime'][anomaly_end],
                        tiles_df['datetime'][anomaly_end], tiles_df['datetime'][int(anomaly_start)]),
                        (-5, -5, max_val, max_val), color="red", alpha=0.1
                    )
                    ax_residuals.axvline(tiles_df['datetime'][changepoint], color='red', lw=0.8)
                    ax_residuals.axvline(tiles_df['datetime'][stopping_time], color='red', lw=0.8)
                    ax_residuals.fill(
                        (tiles_df['datetime'][int(anomaly_start)], tiles_df['datetime'][anomaly_end],
                        tiles_df['datetime'][anomaly_end], tiles_df['datetime'][int(anomaly_start)]),
                        (-5, -5, max_val, max_val), color="red", alpha=0.1
                    )
                    if in_catalog:
                        ax_signal.axvline(cat_event['TIME'], color='green', lw=0.8)
                        ax_signal.axvline(cat_event['TRIGGER_TIME'], color='blue', lw=0.8)
                        ax_signal.axvline(cat_event['END_TIME'], color='green', lw=0.8)
                        ax_signal.fill(
                            (cat_event['TIME'], cat_event['END_TIME'],
                            cat_event['END_TIME'], cat_event['TIME']),
                            (-5, -5, max_val, max_val), color="yellow", alpha=0.1
                        )
                        ax_residuals.axvline(cat_event['TIME'], color='green', lw=0.8)
                        ax_residuals.axvline(cat_event['TRIGGER_TIME'], color='blue', lw=0.8)
                        ax_residuals.axvline(cat_event['END_TIME'], color='green', lw=0.8)
                        ax_residuals.fill(
                            (cat_event['TIME'], cat_event['END_TIME'],
                            cat_event['END_TIME'], cat_event['TIME']),
                            (-5, -5, max_val, max_val), color="yellow", alpha=0.1
                        )

                ax_residuals.yaxis.set_label_coords(-0.06, 1.02)
                ax_signal.yaxis.set_label_coords(-0.06, 1.02)

                if f'{face}_std' in tiles_df.columns:
                    ax_signal.fill_between(tiles_df_subset['datetime'], tiles_df_subset[face_pred] - tiles_df_subset[f'{face}_std'], tiles_df_subset[face_pred] + tiles_df_subset[f'{face}_std'], alpha=0.6)
                    
                ax_residuals.axhline(thresholds[face], color="darkorange", ls='-.', label=f"${thresholds[face]}\\sigma$ threshold")
                ax_residuals.plot(tiles_df_subset['datetime'], tiles_df_subset[f'{face}_significance'],
                                color="blue", label=f"$S_{{{trigger_algo_type}}}$", lw=0.7)
                ax_signal.plot(tiles_df_subset['datetime'], tiles_df_subset[face], label=f'${latex_y_cols[face]}$' if face in latex_y_cols else "N/A",
                            color=face_color)
                ax_signal.plot(tiles_df_subset['datetime'], tiles_df_subset[face_pred], label='background',
                            color='red')
                ax_signal.yaxis.set_major_formatter(formatter)
                ax_signal.set_ylabel(f'$[{units[face]}]$' if face in units else "N/A")
                ax_signal.legend(loc="upper right")
                ax_residuals.legend(loc="upper right")
                ax_signal.set_ylim(min(tiles_df_subset[face]), 1.01 * max(tiles_df_subset[face]))
                ax_residuals.set_ylim(0, 1.01 * max_val)
                ax_signal.set_xlim(start_xlim, end_xlim)
                ax_residuals.set_xlim(start_xlim, end_xlim)

                plt.setp(ax_signal.get_xticklabels(), visible=False)
                if num_support_vars > 0 or i < len(y_cols) - 1:
                    plt.setp(ax_residuals.get_xticklabels(), visible=False)
                i += 1
                
            for var in support_vars:
                formatter = mticker.ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((0, 0))
                ax_support = fig.add_subplot(gss[i][0:1, :])
                axs.append(ax_support)
                ax_support.plot(tiles_df_subset['datetime'], tiles_df_subset[var], color="green", label=var)
                ax_support.set_ylabel(f'[{units[var] if var in units else "N/A"}]')
                ax_support.yaxis.set_major_formatter(formatter)
                ax_support.legend(loc="upper right")

            plt.draw()
            fig.canvas.draw()
            for ax in axs:
                ax.grid(ls='--', alpha=0.6)
                current_label = ax.yaxis.get_label().get_text()
                offset_text = ax.yaxis.get_offset_text().get_text()
                if current_label:
                    ax.set_ylabel(f'{current_label}\n({offset_text})')
                ax.yaxis.offsetText.set_visible(False)

            axs[-1].set_xlim(start_xlim, end_xlim)
            start_datetime = tiles_df['datetime'][int(anomaly_start)].strftime('%Y-%m-%d %H:%M:%S')
            axs[-1].set_xlabel(f"datetime ({start_datetime})")
            axs[0].set_title(f"Triggers with {trigger_algo_type} in {', '.join(latex_faces)}, $t_{{{'trigger'}}}$={start_datetime}, duration={anomaly_duration}s")

            if save:
                path_name = f"{start_datetime}_{'_'.join(faces)}_{cat_event['NAME']}_{cat_event['CAT_NAME']}" if in_catalog else f"{start_datetime}_{'_'.join(faces)}"
                fig.savefig(
                    os.path.join(PLOT_TRIGGER_FOLDER_NAME, f"{path_name}.{extension}"),
                    dpi=180, bbox_inches='tight')
            if show:
                plt.show()
            plt.close('all')

    
    @logger_decorator(logger)
    def plot_anomalies(self, trigger_algo_type, support_vars, thresholds, tiles_df, y_cols, y_pred_cols, save=True, show=False, extension='png', units={}, latex_y_cols={}):
        """Plots the anomalies passed as `df` in Plotter."""
        for an_time, anomalies in tqdm(self.df.items(), desc=f'Plotting anomalies with {trigger_algo_type}'):
            faces = list(anomalies.keys())
            
            num_signal_residual_pairs = len(y_cols)
            num_support_vars = len(support_vars)
            total_rows = 2 * num_signal_residual_pairs + num_support_vars

            fig = plt.figure(figsize=(8.5, 7.5 + 0.37 * total_rows))
            gs0 = GridSpec(num_signal_residual_pairs + num_support_vars, 1, figure=fig, hspace=0.2, left=0.1, right=0.99, top=0.98, bottom=0.04)
            gss = []
            for i in range(num_signal_residual_pairs):
                gss.append(GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[i], hspace=0.))
            gss.append(GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[-1], hspace=0.))

            axs = []

            anomaly_end = -1
            anomaly_start = tiles_df.index[-1]
            for anomaly in anomalies.values():
                if anomaly['start_index'] > anomaly_end:
                    anomaly_end = anomaly['start_index']
                if anomaly['stop_index'] < anomaly_start:
                    anomaly_start = anomaly['stop_index']
            anomaly_delta = 20#max((anomaly_end - int(anomaly_start)) // 5, 120)
            start = max(int(anomaly_start) - anomaly_delta, 0)
            end = min(anomaly_end + anomaly_delta, len(tiles_df))
            start_xlim = tiles_df['DAT_DEFF'][int(anomaly_start)] - timedelta(seconds=anomaly_delta)
            end_xlim = tiles_df['DAT_DEFF'][anomaly_end] + timedelta(seconds=anomaly_delta-1)

            tiles_df_subset = tiles_df[start:end]

            i = 0
            for face, face_pred in zip(y_cols, y_pred_cols):
                ax_signal = fig.add_subplot(gss[i][:-1, :])
                axs.append(ax_signal)

                ax_residuals = fig.add_subplot(gss[i][-1, :], sharex=ax_signal)
                axs.append(ax_residuals)


                face_color = "black"
                max_val = max(max(tiles_df_subset[f'{face}_significance']), thresholds[face] * 1.1)
                if face in faces:
                    face_color = None
                    changepoint = anomalies[face]['stop_index']
                    stopping_time = anomalies[face]['start_index']
                    ax_signal.axvline(tiles_df['DAT_DEFF'][changepoint], color='red', lw=0.8)
                    ax_signal.axvline(tiles_df['DAT_DEFF'][stopping_time], color='red', lw=0.8)
                    ax_signal.fill(
                        (tiles_df['DAT_DEFF'][int(anomaly_start)], tiles_df['DAT_DEFF'][anomaly_end],
                        tiles_df['DAT_DEFF'][anomaly_end], tiles_df['DAT_DEFF'][int(anomaly_start)]),
                        (-5, -5, max_val, max_val), color="red", alpha=0.1
                    )
                    ax_residuals.axvline(tiles_df['DAT_DEFF'][changepoint], color='red', lw=0.8)
                    ax_residuals.axvline(tiles_df['DAT_DEFF'][stopping_time], color='red', lw=0.8)
                    ax_residuals.fill(
                        (tiles_df['DAT_DEFF'][int(anomaly_start)], tiles_df['DAT_DEFF'][anomaly_end],
                        tiles_df['DAT_DEFF'][anomaly_end], tiles_df['DAT_DEFF'][int(anomaly_start)]),
                        (-5, -5, max_val, max_val), color="red", alpha=0.1
                    )
                ax_residuals.set_ylabel(f'${face}$')
                ax_residuals_2 = ax_residuals.twinx()
                ax_residuals_2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
                if face in units:
                    ax_residuals_2.set_ylabel(f'[{units[face]}]')

                ax_residuals.yaxis.set_label_coords(-0.06, 1.02)
                ax_residuals_2.yaxis.set_label_coords(1.02, 1.02)

                if f'{face}_std' in tiles_df.columns:
                    ax_signal.fill_between(tiles_df_subset['DAT_DEFF'], tiles_df_subset[face_pred] - tiles_df_subset[f'{face}_std'], tiles_df_subset[face_pred] + tiles_df_subset[f'{face}_std'], label='prediction error', alpha=0.7)
                    
                ax_residuals.axhline(thresholds[face], color="orange", label=f"${thresholds[face]}\\sigma$")
                ax_residuals.plot(tiles_df_subset['DAT_DEFF'], tiles_df_subset[f'{face}_significance'],
                                color="blue", label=r"$S$", lw=0.7)
                ax_signal.plot(tiles_df_subset['DAT_DEFF'], tiles_df_subset[face], label='signal',
                            color=face_color)
                ax_signal.plot(tiles_df_subset['DAT_DEFF'], tiles_df_subset[face_pred], label='background',
                            color='red')
                ax_signal.legend(loc="upper left")
                ax_residuals.legend(loc="upper left")
                # ax_signal.set_ylim(min(tiles_df_subset[face]), 1.01 * max(tiles_df_subset[face]))
                # ax_residuals.set_ylim(0, 1.01 * max_val)
                ax_signal.set_xlim(start_xlim, end_xlim)
                ax_residuals.set_xlim(start_xlim, end_xlim)

                plt.setp(ax_signal.get_xticklabels(), visible=False)
                i += 1

            for var in support_vars:
                ax_support = fig.add_subplot(gss[i][0:1, :])
                axs.append(ax_support)

                ax_support.plot(tiles_df_subset['DAT_DEFF'], tiles_df_subset[var], color="green", label=var)
                ax_support.set_ylabel(f'[{units[var] if var in units else "NA"}]')
                ax_support.legend(loc="upper left")

            axs[-1].set_xlim(start_xlim, end_xlim)
            start_datetime = tiles_df['DAT_DEFF'][int(anomaly_start)].strftime('%Y-%m-%d %H:%M:%S')
            stop_datetime = tiles_df['DAT_DEFF'][anomaly_end].strftime('%Y-%m-%d %H:%M:%S')
            axs[-1].set_xlabel(f"datetime {tiles_df['DAT_DEFF'][int(anomaly_start)] - timedelta(seconds=anomaly_delta)}")
            axs[0].set_title(f"Triggers from {trigger_algo_type} in ${', '.join(faces)}$ between {start_datetime} and {stop_datetime}")

            if save:
                fig.savefig(
                    os.path.join('intesa sanpaolo', f"{start_datetime}_{'_'.join(faces)}.{extension}"),
                    dpi=200, bbox_inches='tight')
            if show:
                plt.show()
            plt.close('all')

    @logger_decorator(logger)
    @staticmethod
    def show():
        """Shows the plots"""
        plt.show()

    @logger_decorator(logger)
    @staticmethod
    def save(params = None, folder_name = BACKGROUND_PREDICTION_FOLDER_NAME, indexes: tuple = None):
        """Saves the plots"""
        target_path = None
        if params and isinstance(params, dict) and 'model_path' in params:
            target_path = Path(params['model_path']).parent
        elif folder_name:
            target_path = Path(folder_name)
        else:
            target_path = Path(BACKGROUND_PREDICTION_FOLDER_NAME)

        if target_path.suffix:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            file_target = target_path
            folder_target = target_path.parent
        else:
            target_path.mkdir(parents=True, exist_ok=True)
            file_target = None
            folder_target = target_path

        for i in plt.get_fignums():
            title = plt.figure(i).get_label()
            name = f'{title}.png' if not indexes else f'{title}_{indexes[0]}_{indexes[1]}.png'
            if file_target is not None:
                plt.savefig(str(file_target))
            else:
                plt.savefig(str(folder_target / name))
        plt.close('all')

if __name__ == '__main__':
    print('to do')