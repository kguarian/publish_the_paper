# %%
import sklearn.metrics
import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns


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


# to index: probability_pairs[labeler_index][signal_index]
roc_scores = np.empty(
    (len(ground_truth), len(who)), dtype=object)
for i in range(len(ground_truth)):

    
    # plt.figure("figure "+str(i)+ "plots visual")
    for j in range(len(who)):
        order = np.array(results["selections"][who[j]]['indices'])
        selections = np.array(results["selections"][who[j]]["selections"])
        reverse_search_sig_idx = reverse_order(i + num_real_sigs, order)
        selections_indexed_by_labeler = selections[reverse_search_sig_idx]

        curr_sig_idx = i+num_real_sigs
        eeg_signal_profiled_in_this_loop = results['sigs']['sig_'+str(
            i+num_real_sigs)]
        len_curr_sig = len(eeg_signal_profiled_in_this_loop)

        # plt.subplot(len(who), 1, j+1)
        # plt.plot(np.linspace(0, len_curr_sig, len_curr_sig),
        #          eeg_signal_profiled_in_this_loop)
        # plt.axvspan(
        #     selections_indexed_by_labeler[0], selections_indexed_by_labeler[1], color='red', alpha=0.5)
        # plt.axvspan(ground_truth[i][0],
        #             ground_truth[i][1], color='blue', alpha=0.5)

        y_true_boolean = [False]*len_curr_sig
        for subIndex in range(ground_truth[i][0], ground_truth[i][1]+1):
            y_true_boolean[subIndex] = True

        y_pred_boolean = [False]*len_curr_sig
        for subIndex in range(selections_indexed_by_labeler[0], selections_indexed_by_labeler[1]+1):
            y_pred_boolean[subIndex] = True

        yt_int = np.array(y_true_boolean).astype(int)
        yp_int =np.array(y_pred_boolean).astype(int)

        score = sklearn.metrics.roc_auc_score(y_true=yt_int, y_score=yp_int)
        # print(y_true)
        # print(y_pred_prob)
        # print("roc score is: " + str(score))
        roc_scores[i,j]=score
    
    # plt.figure("figure "+str(i)+ " ROC scores")
    # plt.xlim([0,8])
    # plt.ylim([0,1])
    # plt.plot(np.linspace(0,len(who),len(who)), roc_scores[i])

print(roc_scores)
# m_sd = np.array((len(ground_truth), 2)))
for i in range(len(ground_truth)):
    plt.subplot(5,int(len(ground_truth)/5),i+1)
    plt.ylim([0,1])
    plt.boxplot(roc_scores[i])


plt.show()
    
    

input("give input to kill plots cleanly.")
plt.close()

