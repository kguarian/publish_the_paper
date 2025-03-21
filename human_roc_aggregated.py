# %%
from re import sub
from statistics import LinearRegression
import sklearn.metrics
import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.model_selection import train_test_split

from sklearn import linear_model


# %%

num_real_sigs = 49


def reverse_order(j, order):
    for i in range(len(order)):
        if j == order[i]:
            return i


# Returns input for ROC_AUC algorithm (probabilistic attempt)
# to use:
# give X=ground_truth burst array of booleans
# give Y=labeler-predicted burst array of booleans
#
# Returns [(1 - P(x_i=False | y_i=False), P(x_i=True | y_i=True)]) given boolean lists X,Y
# OR the algorithm returns an integral Error code.
#   Code 1: len(X)!=len(Y)
#   Code 2: X[i] or Y[i] is not boolean
def probability_correct_selection(X, Y):
    num_true = 0
    num_false = 0
    selected_true = 0
    selected_false = 0

    if len(X) != len(Y):
        return 1
    for i in range(len(X)):
        xi = X[i]
        yi = Y[i]
        if type(xi) != bool or type(yi) != bool:
            print(type(xi), type(yi))
            return 2
        if xi:
            num_true += 1
            if yi:
                selected_true += 1
        else:
            num_false += 1
            if not yi:
                selected_false += 1

    P_0 = float(selected_false)/float(num_false)
    P_1 = float(selected_true)/float(num_true)

    return [1-P_0, P_1]



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

# Load data
with open("./voyteklabstudy-default-rtdb-export.json") as f:
    results = json.load(f)


# %%

# ground truth has shape (num_participants,2)
# it shows the correct bursting classification
ground_truth = np.array(results["ground_truth"])

# List of names of human collaborators who labeled data
# length of which gives us number of labelings
who = list(results["selections"].keys())
print(type(who))
name = who[0]

result_element = results["selections"][who[0]]

# Performance:
error = np.zeros((len(who), 50, 2))

print("name is %s" % name)
# the i'th element of `order` is the index in signals, corresponding to the i'th signal the labeler saw.
order = np.array(results["selections"][name]['indices'])


# %%
'''
y_pred and y_true are variables for auc_roc
  we want to analyze humans as a whole.
  so y_pred is the interval the humans select (on average?)
  let's start with just one lab person.

'''

# initial values to allow for averaging or single perrson burst selections
y_pred = [[0, 0]]*len(ground_truth)
num_pred = 0

# def find_overlap_intervals(selection_0, selection_1):

# Here we do not have burst labeling code because humans labeled the signals on a study labeling platform

# %%
y_pred = np.zeros(
    (len(ground_truth), len(results['sigs']['sig_'+str(0)])))

y_true = np.zeros(
    (len(ground_truth), len(results['sigs']['sig_'+str(0)])))

scores= np.zeros(len(ground_truth))

for i in range(len(ground_truth)):

    curr_sig_idx = i+num_real_sigs
    eeg_signal_profiled_in_this_loop = results['sigs']['sig_'+str(
        i+num_real_sigs)]
    
    for j in range(len(who)):
        order = np.array(results["selections"][who[j]]['indices'])
        selections = np.array(results["selections"][who[j]]["selections"])
        reverse_search_sig_idx = reverse_order(i + num_real_sigs, order)
        selections_indexed_by_labeler = selections[reverse_search_sig_idx]

        len_curr_sig = len(eeg_signal_profiled_in_this_loop)

        y_true_boolean = [False]*len_curr_sig
        if j == 0:
            for subIndex in range(ground_truth[i][0], ground_truth[i][1]+1):
                y_true[i][subIndex] = 1

        for subIndex in range(selections_indexed_by_labeler[0], selections_indexed_by_labeler[1]+1):
            y_pred[i][subIndex] += 1

    for subindex in range(0, len(eeg_signal_profiled_in_this_loop)):
        y_pred[i][subIndex] /= len(who)

    scores[i] = sklearn.metrics.roc_auc_score(y_true=y_true[i], y_score=y_pred[i])

plt.figure("roc scores aggregated")
plt.boxplot(scores)
plt.show()
plt.close


# snr = -19.65(_ratio) + 9.668

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

pipe5 = make_pipeline(MinMaxScaler(), linear_model.LinearRegression())
pipe5.fit(X_train, y_train)  # apply scaling on training data
print("last x2", pipe5.score(X_test, y_test))


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

pipe5 = make_pipeline(MinMaxScaler(), linear_model.LinearRegression())
pipe5.fit(X_train_out, y_train_out)  # apply scaling on training data
print("last x2", pipe5.score(X_test_out, y_test_out))


