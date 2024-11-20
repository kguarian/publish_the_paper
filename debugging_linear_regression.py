# %%
# %%
from statistics import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import sklearn.metrics
import json
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

import sklearn
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
# %%

# this is the number of real recorded EEG signal we used in the study platform we hosted. The signals were arranged in (real signals, simulated signals) order. Thus, num_real_sigs is used as an array offset in this analysis.
num_real_sigs = 49

# the simulated signals were presented in randomized order to each human labeler. The ordering was recorded as a list of indices.
# the selections the labelers made were recorded in the order they were selected.
# This function finds the index at which the desired signal index appears in the ordering of signal indices.


def reverse_order(j, order):
    for i in range(len(order)):
        if j == order[i]:
            return i


def ratio_to_snr_converter(_ratio):
    snr = -19.65 * (_ratio) + 9.668
    return snr

# this decodes ONE entry in param_list


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

# this decodes ONE entry in param_list


def decode_params_np(param_list) -> dict | None:
    if len(param_list) != 5:
        return None
    retArray = np.zeros(5)
    retArray[0] = 2 * param_list[0]
    retArray[1] = param_list[1]
    retArray[2] = param_list[2]
    retArray[3] = param_list[3]
    retArray[4] = ratio_to_snr_converter(param_list[4])
    return retArray


def param_list_to_training_data(param_list):
    num_samples = len(param_list)
    num_features = 5
    retArray = np.zeros((num_samples, num_features))
    for i in range(len(param_list)):
        row = decode_params_np(param_list[i])
        retArray[i][0:num_features] = row[:num_features]
    return retArray


# Load data from results json exported from firebase
with open("./voyteklabstudy-default-rtdb-export.json") as f:
    results = json.load(f)


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


# Here we do not have burst labeling code because humans labeled the signals on a study labeling platform

# the classes are "bursting" and "non-bursting"
num_classes = 2

y_pred = np.zeros(
    (len(ground_truth), len(results['sigs']['sig_'+str(0)])))
y_true = np.zeros(
    (len(ground_truth), len(results['sigs']['sig_'+str(0)])))
scores = np.zeros(len(ground_truth))


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
        reverse_search_sig_idx = reverse_order(
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

    consensus_threshold = ceil(len(who)/float(2))
    for timepoint in range(len(signal)):
        y_pred[ground_truth_signal_index][timepoint] = 1 if y_pred[ground_truth_signal_index][timepoint] >= consensus_threshold else 0

    scores[ground_truth_signal_index] = sklearn.metrics.f1_score(
        y_true=y_true[ground_truth_signal_index], y_pred=y_pred[ground_truth_signal_index])

# plt.figure("f1 scores aggregated")
# plt.boxplot(scores)
# plt.show()


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
    reverse_search_sig_idx = reverse_order(
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
# multivariate linear regression

# %% [markdown]
# first, figure out what the ratio column from the df looks like.

# %%
param_data = results['params']

# %%
# x= group_param_list(params_for_regression)


# keys = x.keys()
# print(keys)

# print(x['signal-noise ratio'][bad_score_index])
# print(x['freq'][bad_score_index])

# X = [x[i] for i in x.keys()]
X = param_list_to_training_data(param_data)
# X = group_param_list(x)
# print(X)

y = scores


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=104, test_size=0.25, shuffle=True)

print(X_test)


# linear regression
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print(regr.coef_)
# print(regr.get_params())

print("linear regression score: %f" % regr.score(X_test, y_test))


# SVR regression code (https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html)
svr_lin = SVR(kernel="linear", C=100, gamma="auto")
svr_lin.fit(X_train, y_train)
print("linear support vector regression score: %f" % svr_lin.score(X_test, y_test))

svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_rbf.fit(X_train, y_train)
print("rbf svr score: %f" % svr_rbf.score(X_test, y_test))

svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
svr_poly.fit(X_train, y_train)
print("poly svr score: %f" % svr_poly.score(X_test, y_test))

# Same models but preprocessing
# https://scikit-learn.org/1.5/modules/preprocessing.html
# pipe1 = make_pipeline(StandardScaler(), LinearRegression())
# pipe1.fit(X_train, y_train)  # apply scaling on training data
# print(pipe1.score(X_test, y_test))

pipe2 = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))
pipe2.fit(X_train, y_train)  # apply scaling on training data
print(pipe2.score(X_test, y_test))

pipe3 = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))
pipe3.fit(X_train, y_train)  # apply scaling on training data
print(pipe3.score(X_test, y_test))

pipe4 = make_pipeline(StandardScaler(), SVR(kernel="poly", C=100, gamma="auto", degree=3))
pipe4.fit(X_train, y_train)  # apply scaling on training data
print(pipe4.score(X_test, y_test))


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


# linear regression
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print("REMOVED OUTLIERS\n")
print(regr.coef_)
print("linear regression score: %f" % regr.score(X_test_out, y_test_out))

# SVR regression code (https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html)
svr_lin = SVR(kernel="linear", C=100, gamma="auto")
svr_lin.fit(X_train_out, y_train_out)
print("linear support vector regression score: %f" % svr_lin.score(X_test_out, y_test_out))

svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_rbf.fit(X_train_out, y_train_out)
print("rbf svr score: %f" % svr_rbf.score(X_test_out, y_test_out))

svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
svr_poly.fit(X_train_out, y_train_out)
print("poly svr score: %f" % svr_poly.score(X_test_out, y_test_out))

# Same models but preprocessing
# https://scikit-learn.org/1.5/modules/preprocessing.html
# pipe1 = make_pipeline(StandardScaler(), LinearRegression())
# pipe1.fit(X_train_out, y_train_out)  # apply scaling on training data
# print(pipe1.score(X_test_out, y_test_out))

pipe2 = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))
pipe2.fit(X_train_out, y_train_out)  # apply scaling on training data
print(pipe2.score(X_test_out, y_test_out))

pipe3 = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))
pipe3.fit(X_train_out, y_train_out)  # apply scaling on training data
print(pipe3.score(X_test_out, y_test_out))

pipe4 = make_pipeline(StandardScaler(), SVR(kernel="poly", C=1, gamma="auto", degree=3))
pipe4.fit(X_train_out, y_train_out)  # apply scaling on training data
print("last", pipe4.score(X_test_out, y_test_out))

pipe4 = make_pipeline(MinMaxScaler(), SVR(kernel="poly", C=1, gamma="auto", degree=3))
pipe4.fit(X_train_out, y_train_out)  # apply scaling on training data
print("last", pipe4.score(X_test_out, y_test_out))
