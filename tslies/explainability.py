"""This module contains functions to explain the model using LIME and SHAP."""
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from keras.model import load_model

from .lime import lime_tabular
from .plotter import Plotter

def get_feature_importance(model_path, inputs_outputs_df, y_cols, x_cols, num_sample = 100, model = None, show=True, save=True):
    """Get the feature importance using LIME and SHAP and plots it with matplotlib barh.
    
    Parameters
    ----------
        model_path (str): The path to the model.
        inputs_outputs_df (pd.DataFrame): The input and output data.
        y_cols (list): The columns of the output data.
        x_cols (list): The columns of the input data.
        show (bool): Whether to show the plot."""
    X_test = inputs_outputs_df[x_cols]
    if model is None:
        model = load_model(model_path)
        scaler = StandardScaler()
        scaler.fit(X_test)
    else:
        scaler = model.scaler
        model = model.nn_r
    X_test =  pd.DataFrame(scaler.transform(X_test), columns=x_cols)

    y_test = inputs_outputs_df[y_cols]
    X_back = X_test.sample(num_sample)
    importance_dict = {face: {col: 0 for col in X_back.columns} for face in y_cols}
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_back),
        feature_names=X_back.columns,
        class_names=list(y_test.columns),
        mode='regression'
    )

    for i in range(num_sample):
        for mylabel, face in enumerate(y_cols):
            exp = explainer.explain_instance(
                data_row=X_back.iloc[i],
                mylabel=mylabel,
                predict_fn=model.predict,
                num_features=len(X_back.columns)
            )
            for feature, weight in exp.as_list(label=mylabel):
                for col in set(x_cols):
                    if col in feature:
                        importance_dict[face][col] += np.abs(weight) / num_sample
                        break

    summed_importance_dict = {col: 0 for col in importance_dict[y_cols[0]].keys()}
    for face in y_cols:
        for col, value in importance_dict[face].items():
            summed_importance_dict[col] += value

    summed_sorted_importance_dict = dict(sorted(summed_importance_dict.items(), key=lambda item: item[1]))

    sorted_importance_dict = {face: dict(sorted(value.items(), key=lambda item: item[1])) for face, value in importance_dict.items()}
    
    file_name = 'Feature Importance with Lime' if model_path.endswith('.keras') else model_path
    plt.figure(num=file_name, figsize=(10, 8))
    left = {col: 0 for col in importance_dict[y_cols[0]]}
    for i, key in enumerate(importance_dict.keys()):
        sorted_importance_dict[key] = {k: sorted_importance_dict[key][k] for k in summed_sorted_importance_dict}
        left_arr = [left[key] for key in sorted_importance_dict[key]]
        bars = plt.barh(list(sorted_importance_dict[key].keys()), list(sorted_importance_dict[key].values()), left=left_arr, label=key)
        for col in sorted_importance_dict[key]:
            left[col] += sorted_importance_dict[key][col]
    for bar, sum in zip(bars, summed_sorted_importance_dict.values()):
        plt.text(sum * 1.002, bar.get_y() + bar.get_height() / 2, f'{sum:.4f}', va='center', color='grey')
    plt.legend(loc='lower right')
    plt.tight_layout()

    # e = shap.KernelExplainer(model, X_back)
    # shap_values_orig = e(X_back)
    # shap_values_sum = np.mean(shap_values_orig.values, axis=2)
    # shap_values_sum = shap.Explanation(values=shap_values_sum,
    #                                 base_values=shap_values_orig.base_values[0],
    #                                 data=shap_values_orig.data,
    #                                 feature_names=shap_values_orig.feature_names)
    # plt.figure(num='Feature Importance with SHAP')
    # shap.plots.bar(shap_values_sum, max_display=len(X_back.columns), show=False)

    # idx = 0
    # plt.figure(num='Waterfall plot with SHAP')
    # shap.plots._waterfall.waterfall_legacy(e.expected_value[idx], shap_values_orig[0].T[idx],
    #                                        feature_names=x_cols, max_display=len(x_cols))

    if show:
        plt.show()
    if save:
        Plotter.save(os.path.dirname(model_path) if model_path.endswith('.keras') else model_path)
