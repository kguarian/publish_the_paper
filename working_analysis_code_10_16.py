import json
import numpy as np
import matplotlib.pyplot as plt

num_real_sigs = 49

def reverse_order(j, order):
    for i in range(len(order)):
        if j==order[i]:
            return i



# Load data
with open("./voyteklabstudy-default-rtdb-export.json") as f:
    results = json.load(f)

# ground truth has shape (num_participants,2)
# it shows the correct bursting classification
ground_truth = np.array(results["ground_truth"])

# List of names of human collaborators who labeled data
# length of which gives us number of labelings
who = list(results["selections"].keys())
print(type(who))
name = who[0]

result_element = results["selections"][who[0]]

# excessive output. Use when necessary.
# print(results)

# Performance:
error = np.zeros((len(who), 50, 2))

print("name is %s" % name)
# the i'th element of `order` is the index in signals, corresponding to the i'th signal the labeler saw.
order = np.array(results["selections"][name]['indices'])
# print order
print("order:", order)

analytics = [None]*len(ground_truth)
for i in range(len(ground_truth)):
    analytics[i] = [None]*(len(who))
    plt.figure("figure "+str(i))
    for j in range(len(who)):
        order = np.array(results["selections"][who[j]]['indices'])
        selections = np.array(results["selections"][who[j]]["selections"])
        reverse_search_sig_idx = reverse_order(i+num_real_sigs, order)
        selections_indexed_by_labeler = selections[reverse_search_sig_idx]
        # order: tp, fp, tn, fn
        print(i)
        print(order[i])
        print(selections[order[i]])
        print(ground_truth[i])
        curr_sig_idx = i+num_real_sigs
        eeg_signal_profiled_in_this_loop = results['sigs']['sig_'+str(i+num_real_sigs)]
        len_curr_sig = len(eeg_signal_profiled_in_this_loop)

        plt.subplot(len(who),1,j+1)
        plt.plot(np.linspace(0, len_curr_sig, len_curr_sig),
                 eeg_signal_profiled_in_this_loop)
        plt.axvspan(
            selections_indexed_by_labeler[0], selections_indexed_by_labeler[1], color='red', alpha=0.5)
        plt.axvspan(ground_truth[i][0],
                    ground_truth[i][1], color='blue', alpha=0.5)

        overlap = 0
        start_overlap = max(selections_indexed_by_labeler[0], ground_truth[i][0])
        end_overlap = min(selections_indexed_by_labeler[1], ground_truth[i][1])
        if start_overlap < end_overlap:
            overlap = end_overlap-start_overlap

        analytics[i][j] = np.zeros(4)
        analytics[i][j][0] = overlap
        analytics[i][j][1] = len_curr_sig



plt.show()
selections = selections[np.argsort(order)[49:]]

for i in range(50):
    plt.close(i)

# This needs to be updated for AUC-ROC and AUC-PR instead
error[ind] = (selections - ground_truth)

# MAE Results
who = [" ".join(i.split("@")[0].split("_")[0].split(" ")[:2])
       for i in list(results["selections"].keys())]

abs_error = np.abs(error)

dict(zip(who, abs_error.mean(axis=(1, 2))))
