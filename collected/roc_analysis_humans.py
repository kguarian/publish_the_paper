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

from common_funcitons import decode_params
from common_funcitons import decode_params_np
from common_funcitons import reverse_order_search
from common_funcitons import param_list_to_training_data

# %%

# Load data from results json exported from firebase
with open("./voyteklabstudy-default-rtdb-export.json") as f:
    results = json.load(f)

# %% [markdown]
# Set constants

# %%


# this is the number of real recorded EEG signal we used in the study platform we hosted. The
# signals were arranged in (real signals, simulated signals) order. Thus, num_real_sigs is used
# as an array offset in this analysis.
num_real_sigs = 49


# ground truth has shape (num_participants,2)
# it shows the correct bursting classification.
# ground_truth[i] = [a,b] where a is true burst onset and b is true burst offset.
ground_truth = np.array(results["ground_truth"])

# List of names of human collaborators who labeled data
# length of which gives us number of labelings. Allows us to iterate through labelers
who = list(results["selections"].keys())
print(who)

'''
y_pred and y_true are variables for auc_roc
  we want to analyze humans as a whole.
  so y_pred is the interval the humans select (on average?)
  let's start with just one lab person.
'''
y_pred = np.zeros(
    (len(ground_truth), len(results['sigs']['sig_'+str(0)])))
y_true = np.zeros(
    (len(ground_truth), len(results['sigs']['sig_'+str(0)])))
scores = np.zeros(len(ground_truth))

# the classes are "bursting" and "non-bursting"
num_classes = 2




# %% [markdown]
# Assign one F1 Score per Signal.
# 
# We generate consensus about each timepoint in each signal.
# We predict a timepoint is bursting if and only if most participants decide it is.

# %%

# iterates through each signal to create the y_pred[i] for signal i.
for ground_truth_signal_index in range(len(ground_truth)):
    signal_index = ground_truth_signal_index+num_real_sigs
    signal = results['sigs']['sig_'+str(
        ground_truth_signal_index+num_real_sigs)]

    # this loop records the quantity of labelers who each timepoint in signal i as bursting
    for labeler_index in range(len(who)):
        order = np.array(results["selections"][who[labeler_index]]['indices'])
        selections = np.array(
            results["selections"][who[labeler_index]]["selections"])

        # this block retrieves the signal selections that labeler j made on signal i.
        # then assigns this value to selections_indexed_by_labeler.
        reverse_search_sig_idx = reverse_order_search(
            ground_truth_signal_index + num_real_sigs, order)
        selections_indexed_by_labeler = selections[reverse_search_sig_idx]

        len_curr_sig = len(signal)

        # records ground truth for signal i in y_true[i].
        # The j==0 condition ensures that y_true isn't set repetitively.
        if labeler_index == 0:
            for time_point in range(ground_truth[ground_truth_signal_index][0], ground_truth[ground_truth_signal_index][1]+1):
                y_true[ground_truth_signal_index][time_point] = 1

        for time_point in range(selections_indexed_by_labeler[0], selections_indexed_by_labeler[1]+1):
            # IN CONTEXT OF ALL ENCLOSING LOOPS, this line counts the number of people
            # who labeled this timepoint as containing a burst,
            # for all timepoints, labelers, and signals
            y_pred[ground_truth_signal_index][time_point] += 1

    # consensus
    consensus_threshold = ceil(len(who)/float(2))
    for timepoint in range(len(signal)):
        y_pred[ground_truth_signal_index][timepoint] = 1 if y_pred[ground_truth_signal_index][timepoint] >= consensus_threshold else 0

    scores[ground_truth_signal_index] = sklearn.metrics.f1_score(
        y_true=y_true[ground_truth_signal_index], y_pred=y_pred[ground_truth_signal_index])

# plt.figure("f1 scores aggregated")
# plt.boxplot(scores)
# plt.show()



# %% [markdown]
# Analysis of the 0.0 f-score signal

# %%

# identify the really bad signal.
max_val, bad_score_index = 1.0, -1
for ground_truth_signal_index in range(len(ground_truth)):
    if scores[ground_truth_signal_index] < max_val:
        max_val = scores[ground_truth_signal_index]
        bad_score_index = ground_truth_signal_index

print("worst f1 score:", scores[bad_score_index], "index", bad_score_index)
params = results["params"][bad_score_index]
print("raw params: %s" % str(params))
params = decode_params(params)
print("decoded params: %s" % params)

sig_bad_perf = results['sigs']['sig_'+str(
    bad_score_index+num_real_sigs)]
# this loop records the quantity of labelers who each timepoint in signal i as bursting
for labeler_index in range(len(who)):
    order = np.array(results["selections"][who[labeler_index]]['indices'])
    selections = np.array(results["selections"]
                          [who[labeler_index]]["selections"])

    # this block retrieves the signal selections that labeler j made on signal i.
    # then assigns this value to selections_indexed_by_labeler.
    reverse_search_sig_idx = reverse_order_search(
        bad_score_index + num_real_sigs, order)
    selections_indexed_by_labeler = selections[reverse_search_sig_idx]

    plt.subplot(len(who), 1, labeler_index+1)
    plt.plot(np.linspace(0, len_curr_sig, len_curr_sig),
             sig_bad_perf)
    plt.axvspan(
        selections_indexed_by_labeler[0], selections_indexed_by_labeler[1], color='red', alpha=0.5)
    plt.axvspan(ground_truth[bad_score_index][0],
                ground_truth[bad_score_index][1], color='blue', alpha=0.5)


# %% [markdown]
# Data preparation for regressions on data with outliers included

# %%
param_data = results['params']

X = param_list_to_training_data(param_data)
y = scores.copy()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=104, test_size=0.25, shuffle=True)

print(X_test)


# %% [markdown]
# Regressions first without standardizing data, then with

# %%

# operating on raw data

print("\nUNSCALED DATA, OUTLIERS KEPT:")
# linear regression
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
# print("linear regression params: %s" % str(regr.get_params()))
print("linear regression score: %f" % regr.score(X_test, y_test))



# Operating on scaled data
# Same models but preprocessing
# https://scikit-learn.org/1.5/modules/preprocessing.html

print("\nSCALED DATA, OUTLIERS KEPT:")
pipe1 = make_pipeline(StandardScaler(), linear_model.LinearRegression())
pipe1.fit(X_train, y_train)  # apply scaling on training data
print("linear regression score: %f" % pipe1.score(X_test, y_test))

# removing outliers.
score_copy = np.array(scores)

z_scores = np.abs(stats.zscore(score_copy))
threshold = 3
# print("z_scores", z_scores)
indices = np.array([], dtype=int)
# print(X_test.shape)
for i in range(len(score_copy)):
    if z_scores[i]<3:
        indices = np.append(indices, i)
# print("indices", indices)

X_out = X[indices]
y_out = y[indices]
X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(
    X_out, y_out, random_state=104, test_size=0.25, shuffle=True)


# %% [markdown]
# Apply same regressions as last time

# %%

print("\nUNSCALED DATA, REMOVED OUTLIERS:")

# linear regression
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print("REMOVED OUTLIERS\n")
print(regr.coef_)
print("linear regression score: %f" % regr.score(X_test_out, y_test_out))

print("\nSCALED DATA, REMOVED OUTLIERS:")
# Same models but preprocessing
# https://scikit-learn.org/1.5/modules/preprocessing.html
pipe1 = make_pipeline(StandardScaler(), linear_model.LinearRegression())
pipe1.fit(X_train_out, y_train_out)  # apply scaling on training data
print("linear regression score: %f" % pipe1.score(X_test_out, y_test_out))
