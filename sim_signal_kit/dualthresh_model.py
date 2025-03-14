import torch.nn as nn

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

import warnings

warnings.filterwarnings("ignore")

num_real_sigs = 49


def reverse_order(j, order):
    for i in range(len(order)):
        if j == order[i]:
            return i


# Load data
with open(
    "/Users/kenton/HOME/coding/python/publish_the_paper/voyteklabstudy-default-rtdb-export.json"
) as f:
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
        bursting_interval_count = 1
    siglen = len(is_burst)
    for i in range(0, siglen - 1):
        val_at_i = is_burst[i]
        val_at_i_plus_1 = is_burst[i + 1]
        if (not val_at_i) and val_at_i_plus_1:
            bursting_interval_count += 1

    return bursting_interval_count


# accommodates multiple intervals right now. can be optimized (a tiny bit) by restricting one interval
def get_bursting_intervals(is_burst, num_intervals):
    bursting_intervals = [[0, 0] for _ in range(num_intervals)]
    burst_interval_index = 0
    siglen = len(is_burst)

    if is_burst[0]:
        bursting_intervals[0][0] = 0
    for i in range(0, siglen - 1):
        val_at_i = is_burst[i]
        val_at_i_plus_1 = is_burst[i + 1]
        if val_at_i and (not val_at_i_plus_1):
            bursting_intervals[burst_interval_index][1] = i
            burst_interval_index += 1
        elif (not val_at_i) and val_at_i_plus_1:  # changed condition here
            bursting_intervals[burst_interval_index][0] = i
    return bursting_intervals


def merge_burst_selections(detections_for_signal):
    """
    (Mutates Parameters): Process YOLO burst detections to extract and average onsets and offsets.
    for each burst, the function calculates the average onset and offset times.

    Parameters:
        burst_detections list with length i: A list of (onset,offset) pairs
    Returns:
        Nothing
    """

    # Sort intervals by start time
    detections_for_signal.sort(key=lambda x: x[0])

    if len(detections_for_signal) == 0:
        return []
    # naive solution, O(n) runtime. Good.
    combine_table = np.full(len(detections_for_signal) - 1, False)
    len_table = len(combine_table)
    for i in range(len_table):
        last_start = detections_for_signal[i][0]
        last_end = detections_for_signal[i][1]
        next_start = detections_for_signal[i + 1][0]
        next_end = detections_for_signal[i + 1][1]
        if next_start <= last_end or last_start <= next_end:
            combine_table[i] = True

    for i in range(len_table):
        if combine_table[len_table - i - 1]:
            detections_for_signal[len_table - i - 1][0] = min(
                detections_for_signal[len_table - i - 1][0],
                detections_for_signal[len_table - i][0],
            )
            detections_for_signal[len_table - i - 1][1] = max(
                detections_for_signal[len_table - i - 1][1],
                detections_for_signal[len_table - i][1],
            )
            # we want to do longest common interval, for each interval in the set. So we want to take each interval out of the bag at some point.
            detections_for_signal.pop(len_table - i)


class DualThreshModel:
    def predict(self, signal, params, fs):
        """
        params[0] is lower frequency
        params[1] is upper frequency
        params[2] is dual_thresh-low
        params[3] is dual_thresh-high
        """

        freqs, power_spectral_density = compute_spectrum(fs=fs, sig=signal)
        sm = specparam.SpectralModel(peak_width_limits=[1.0, 8.0], max_n_peaks=8)
        sm.fit(freqs, power_spectrum=power_spectral_density)
        [center_frequency, log_power, bandwidth] = specparam.analysis.get_band_peak(
            sm, [10, 20], select_highest=True
        )
        # print(center_frequency)
        # print(params)
        # print(type(signal))

        # print("params[0]: ", params[0])
        # print(
        #     f"signal type and dimensions: {type(signal)}, {signal.ndim}",
        #     f"fs type and dimensions: {type(fs)}",
        #     f"params type and dimensions: {type(params)} {params.ndim}",
        #     f"params[0] type and dimensions: {type(params[0])}",
        #     f"params[1] type and dimensions: {type(params[1])}",
        #     f"params[2] type and dimensions: {type(params[2])}",
        #     f"params[3] type and dimensions: {type(params[3])}",
        #     sep="\n",
        # )

        is_burst = detect_bursts_dual_threshold(
            sig=signal,
            fs=fs,
            f_range=(params[0], params[1]),
            dual_thresh=(params[2], params[3]),
        )
        num_intervals = num_bursting_intervals(is_burst)
        intervals = get_bursting_intervals(is_burst, num_intervals)
        if len(intervals) != 0:
            print("intervals:", intervals)
        # print("intervals:", intervals)
        intervals = merge_burst_selections(intervals)  # now using the intervals
        return intervals
