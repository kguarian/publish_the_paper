import json
import numpy as np
import matplotlib.pyplot as plt

num_real_sigs = 49

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

# TODO: Remove
ind = 0
i = 0

selections = np.array(results["selections"][name]["selections"])
for i in range(len(ground_truth)):
    # 
    # for ind, name in enumerate(who):
    print(i)
    print(order[i])
    print(selections[order[i]])
    print(ground_truth[i])
    curr_sig_idx = order[i]
    if curr_sig_idx <= num_real_sigs:
        continue
    eeg_signal_profiled_in_this_loop = results['sigs']['sig_'+str(
        curr_sig_idx)]
    len_curr_sig = len(eeg_signal_profiled_in_this_loop)

    plt.figure("figure "+str(i))
    plt.plot(np.linspace(0, len_curr_sig, len_curr_sig),
             eeg_signal_profiled_in_this_loop)
    plt.axvspan(selections[i][0], selections[i][1], color='red', alpha=0.5)
    plt.axvspan(ground_truth[curr_sig_idx-num_real_sigs][0],
                ground_truth[curr_sig_idx-num_real_sigs][1], color='blue', alpha=0.5)
    print("no_op")

plt.show()
selections = selections[np.argsort(order)[49:]]
input("")
for i in range(50):
    plt.close(i)

# This needs to be updated for AUC-ROC and AUC-PR instead
error[ind] = (selections - ground_truth)

# MAE Results
who = [" ".join(i.split("@")[0].split("_")[0].split(" ")[:2])
       for i in list(results["selections"].keys())]

abs_error = np.abs(error)

dict(zip(who, abs_error.mean(axis=(1, 2))))
