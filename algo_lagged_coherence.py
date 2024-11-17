# %% [markdown]
# First, let's see what it takes to detect bursts with specparam

# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn

import specparam

from neurodsp.burst import detect_bursts_dual_threshold
from neurodsp.plts.time_series import plot_time_series, plot_bursts
from neurodsp.rhythm import compute_lagged_coherence
from neurodsp.plts.rhythm import plot_lagged_coherence

from neurodsp.rhythm import sliding_window_matching
from neurodsp.plts.rhythm import plot_swm_pattern

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

# plotting loop
for i in range(len(ground_truth)):
    # Here we have code to execute the burst labeling.
    test_signal = results['sigs']['sig_'+str(i + num_real_sigs)]
    test_signal = np.array(test_signal)

    lagged_coh_alpha, freqs = compute_lagged_coherence(test_signal, fs=fs, freqs=[2,30,.1], return_spectrum=True)
    # Set the frequency range to compute the spectrum of LC values across
    lc_range = (8, 30)

    plot_lagged_coherence(freqs, lagged_coh_alpha)
    
    windows = None
    try:
        windows, starts = sliding_window_matching(test_signal, fs=fs, win_len=1./15,
                                          win_spacing=0.05)

    except:
        print("skipped %d" % i)
        continue

    
    # Compute the average window
    avg_window = np.mean(windows, 0)

    # Plot the discovered pattern
    plot_swm_pattern(avg_window)


plt.show()
# sm.fit(freqs = freqs, power_spectrum = power_spectral_density, freq_range=[1.4,15.0])
# sm.report(freqs, power_spectrum=power_spectral_density, freq_range=[1.4,15.0])
