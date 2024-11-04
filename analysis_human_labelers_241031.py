# %%
import sklearn.metrics
import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn


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

num_classes = 2
# to index: probability_pairs[labeler_index][signal_index]
probability_pairs = np.empty(
    (len(ground_truth), len(who), num_classes), dtype=object)
ground_truths = np.empty(
    (len(ground_truth), len(who), num_classes), dtype=object)

for i in range(0, len(ground_truth)):
    for j in range(0, len(who)):
        ground_truths[i][j][0:2] = [0, 1]

for i in range(len(ground_truth)):

    plt.figure("figure "+str(i))

    for j in range(len(who)):
        order = np.array(results["selections"][who[j]]['indices'])
        selections = np.array(results["selections"][who[j]]["selections"])
        reverse_search_sig_idx = reverse_order(i + num_real_sigs, order)
        selections_indexed_by_labeler = selections[reverse_search_sig_idx]

        curr_sig_idx = i+num_real_sigs
        eeg_signal_profiled_in_this_loop = results['sigs']['sig_'+str(
            i+num_real_sigs)]
        len_curr_sig = len(eeg_signal_profiled_in_this_loop)

        plt.subplot(len(who), 1, j+1)
        plt.plot(np.linspace(0, len_curr_sig, len_curr_sig),
                 eeg_signal_profiled_in_this_loop)
        plt.axvspan(
            selections_indexed_by_labeler[0], selections_indexed_by_labeler[1], color='red', alpha=0.5)
        plt.axvspan(ground_truth[i][0],
                    ground_truth[i][1], color='blue', alpha=0.5)

        y_true_boolean = [False]*len_curr_sig
        for subIndex in range(ground_truth[i][0], ground_truth[i][1]+1):
            y_true_boolean[subIndex] = True

        y_pred_boolean = [False]*len_curr_sig
        for subIndex in range(selections_indexed_by_labeler[0], selections_indexed_by_labeler[1]+1):
            y_pred_boolean[subIndex] = True

        y_pred_prob = probability_correct_selection(
            y_true_boolean, y_pred_boolean)
        # print(type(y_pred_prob))
        if type(y_pred_prob) == int:
            print(y_pred_prob)
            print("review code: the stats function should not have bombed.")
            exit(3)
        

        probability_pairs[i][j] = y_pred_prob[0]
        # y_true = [0, 1]

        # score = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_pred_prob)
        # print(y_true)
        # print(y_pred_prob)
        # print(score)


# we want subarray=ground_truths[:][0][0]

gt_roc=ground_truths[:,0,0]
pred_roc =y_score=probability_pairs[:,0,0]

print(gt_roc.shape)
score = sklearn.metrics.roc_auc_score(
     )
# print(ground_truths[0][:][0])
# print(y_pred_prob[0][:][0])
print(score)
