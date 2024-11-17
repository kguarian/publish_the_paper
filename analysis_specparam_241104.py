# %% [markdown]
# First, let's see what it takes to detect bursts with specparam

# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn

import specparam

from neurodsp.spectral import compute_spectrum
from neurodsp.burst import detect_bursts_dual_threshold, compute_burst_stats
from neurodsp.plts.time_series import plot_time_series, plot_bursts

import itertools

# %%

from math import nan, isfinite
from numpy import ndarray


num_real_sigs = 49
fs = 1000


def reverse_order(j, order):
    for i in range(len(order)):
        if j == order[i]:
            return i


# Load data
with open("./voyteklabstudy-default-rtdb-export.json") as f:
    results = json.load(f)


def plot_signal(y):
    x = np.linspace(0, len(y), len(y))

    plt.plot(x, y)

# unfinished

# runtime analysis of (num_bursting_intervals + get_bursting_intervals):
# 
def num_bursting_intervals(is_burst) -> int:
    bursting_interval_count = 0
    if is_burst[0]:
        bursting_interval_count=1
    siglen = len(is_burst)
    for i in range(0, siglen-1):
        val_at_i = is_burst[i]
        val_at_i_plus_1 = is_burst[i+1]
        if (not val_at_i) and val_at_i_plus_1:
            bursting_interval_count += 1

    return bursting_interval_count

# accommodates multiple intervals right now. can be optimized (a tiny bit) by restricting one interval
def get_bursting_intervals(is_burst, num_intervals):
    bursting_intervals = [[0,0]]*num_intervals
    burst_interval_index = 0
    siglen = len(is_burst)

    if is_burst[0]:
        bursting_intervals[0][0]=0
    for i in range(0, siglen-1):
        val_at_i = is_burst[i]
        val_at_i_plus_1 = is_burst[i+1]
        if val_at_i and (not val_at_i_plus_1):
            bursting_intervals[burst_interval_index][1]=i
            burst_interval_index += 1
        elif (not val_at_i) and val_at_i:
            bursting_intervals[burst_interval_index][0]=i
    return bursting_intervals



    

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
from sklearn.isotonic import spearmanr


y_pred: list[tuple[int,int]] = [None]*len(ground_truth)
num_pred = 0

# should memeoize every neurodsp function so the ROC score loop (loop below this one) runs blazingly fast.
center_freq_bandwidth_memoization = [None]*len(ground_truth)
for i in range(len(ground_truth)):
    # Here we have code to execute the burst labeling.
    test_signal = results['sigs']['sig_'+str(i + num_real_sigs)]
    test_signal = np.array(test_signal)

    freqs, power_spectral_density = compute_spectrum(fs=fs, sig=test_signal)
    sm = specparam.SpectralModel(peak_width_limits=[1.0, 8.0], max_n_peaks=8)
    sm.fit(freqs, power_spectrum=power_spectral_density)
    [center_frequency, log_power, bandwidth] = specparam.analysis.get_band_peak(
        sm, [10, 20], select_highest=True)
    print(center_frequency)
    center_freq_bandwidth_memoization[i] = [center_frequency, bandwidth]

    is_burst = detect_bursts_dual_threshold(sig=np.array(
        test_signal), fs=fs, f_range=(9, 21), dual_thresh=(1, 2))
    plot_bursts(np.linspace(0, 1, 1000), test_signal, is_burst)

    num_bursts = num_bursting_intervals(is_burst)
    if (num_bursts > 1):
        print(3/0)
    intervals: list[list[int]] = get_bursting_intervals(num_intervals=num_bursts, is_burst=is_burst)
    # should only be one interval:
    if len(intervals)==0:
        y_pred[i]=(0,0)
    else:
        y_pred[i]=(intervals[0][0], intervals[0][1])
    print(y_pred[i])


# %%

import sklearn.metrics


# should memeoize every neurodsp function so the ROC score loop (loop below this one) runs blazingly fast.
roc_scores = np.empty(
    (len(ground_truth)), dtype=object)
for i in range(len(ground_truth)):
    plt.figure("figure "+str(i))
    curr_sig_idx = i+num_real_sigs
    center_frequency = center_freq_bandwidth_memoization[i][0]
    bandwidth = center_freq_bandwidth_memoization[i][1]
    eeg_signal_profiled_in_this_loop = results['sigs']['sig_'+str(
        i+num_real_sigs)]
    len_curr_sig = len(eeg_signal_profiled_in_this_loop)
    selections_indexed_by_labeler = y_pred[i]
    if isfinite(bandwidth) and isfinite(center_frequency):
        print("cf, bandwidth = ", center_frequency, bandwidth)
        is_burst = detect_bursts_dual_threshold(sig=np.array(
            eeg_signal_profiled_in_this_loop), fs=fs, f_range=(center_frequency-bandwidth, center_frequency+bandwidth), dual_thresh=(1, 2))
    else:
        is_burst=[False]*len(eeg_signal_profiled_in_this_loop)
    
    y_true_boolean = [False]*len_curr_sig
    for subIndex in range(ground_truth[i][0], ground_truth[i][1]+1):
        y_true_boolean[subIndex] = True


    yt_int = np.array(y_true_boolean).astype(int)
    yp_int =np.array(is_burst).astype(int)
    

    score = sklearn.metrics.roc_auc_score(y_true=yt_int, y_score=yp_int)

    roc_scores[i]=float(score)


print(roc_scores)
# %%


