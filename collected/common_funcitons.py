# %% [markdown]
# Imports

# %%
from statistics import LinearRegression
import json
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy import stats

import sklearn
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# %% [markdown]
# analysis functions

# %%
# the simulated signals were presented in randomized order to each human labeler. The ordering was recorded as a list of indices.
# the selections the labelers made were recorded in the order they were made.
# This function finds the index at which the desired signal index appears in the ordering of signal indices.
def reverse_order_search(j, order):
    for i in range(len(order)):
        if j == order[i]:
            return i

# the linear coefficients come from a linear regression model searching for the best fit line between
# signal-noise ratio and _ratio parameter used to generate signals
def ratio_to_snr_converter(_ratio):
    snr = -19.65 * (_ratio) + 9.668
    return snr

# this decodes ONE entry in the list of params used to generate one signal.
# the function returns a dictionary with named parameters for visual inspection
# this function is only used when investigating the signal with the worst f1 score.
def decode_params(param_list) -> dict | None:
    if len(param_list) != 5:
        return None
    retDict = {}
    retDict["freq"] = 2 * param_list[0]
    retDict["n_cycles"] = param_list[1]
    retDict["rise-decay asymmetry"] = param_list[2]
    retDict["aperiodic exponent"] = param_list[3]
    retDict["signal-noise ratio"] = ratio_to_snr_converter(param_list[4])
    return retDict

# this decodes ONE entry in the list of params used to generate one signal.
# the function returns a numpy array with the same parameters as decode_params.
# this function is used when preparing data for regression analysis.
# It differs from decode_params in that it returns a numpy array instead of a dictionary.
def decode_params_np(param_list) -> dict | None:
    if len(param_list) != 5:
        return None
    retArray = np.zeros(5)
    retArray[0] = 2 * param_list[0]
    retArray[1] = param_list[1]
    retArray[2] = `param_list[2]
    retArray[3] = param_list[3]
    retArray[4] = ratio_to_snr_converter(param_list[4])
    return retArray

# this function takes a list of parameters and returns a dictionary with named parameters.
# the function is used when preparing data for regression analysis.
# this function calls decode_params to ensure frequency and snr are accurate.
def param_list_to_training_data(param_list):
    num_samples = len(param_list)
    num_features = 5
    retArray = np.zeros((num_samples, num_features))
    for i in range(len(param_list)):
        row = decode_params_np(param_list[i])
        retArray[i][0:num_features] = row[:num_features]
    return retArray
