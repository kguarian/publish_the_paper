# purpose of this file is to create a dataset of images that can beused to
# train a machine learning model to detect the presence of a burst in a
# signal.

from itertools import product, cycle
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from skopt import gp_minimize

import math
import os
import time

from neurodsp.sim.periodic import sim_oscillation
from neurodsp.sim.aperiodic import sim_powerlaw

from dualthresh_model import DualThreshModel
from neurodsp.burst import detect_bursts_dual_threshold

# from json_objects_yolov11 import AreaSelection, Annotations, TrainingSample, scale_selection, generate_ml_training_data

from parallel_data_annotation import generate_ml_training_data
from multiprocessing import freeze_support
from PIL import Image

model = DualThreshModel()
fs = 1000
        
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    if isinstance(obj, np.int64) or isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return obj.item()  # Convert NumPy scalars to Python scalars
    return obj  # Default case


def dualthresh_loss(signal, ground_truth, fs):
    """
    Loss function for dual threshold optimization.
    """

    def loss_fn(params, *args):
        """
        Loss function for dual threshold optimization.
        """

        intervals = model.predict(signal=signal, params=params, fs=fs)
        # Compute loss
        if intervals is None:
            return 1e99
        if len(intervals) == 0:
            return 1e99
        else:
            loss = (intervals[0][0] - ground_truth[0]) ** 2 + (
                intervals[0][1] - ground_truth[1]
            ) ** 2

        return loss

    return loss_fn


def generate_training_data_approximator():
    # if the file dualthresh.csv exists, load it with pd.read_csv("dualthresh.csv") and safety checks
    # if the file does not exist, generate the training data

    if os.path.exists("dualthresh.csv"):
        with open("dualthresh.csv", "r") as file:
            data = json.load(file)
            positive_examples = data[0]
            param_opt = data[1]
            ground_truth = data[2]
            return positive_examples, param_opt, ground_truth

    """
    Generate training data for the approximator.
    """

    feature_list = pd.DataFrame()

    # directory for ml training data
    training_data_directory = "dualthresh/train"
    test_data_directory = "dualthresh/test"

    # General
    n_seconds = 2
    fs = 500
    n_sims = 1000
    pos_whole_ratio = 0.8
    n_samples = int(n_seconds * fs)

    # Periodic
    freqs = [5, 10]
    n_cycles = [3, 6]
    rdsym = [0.2, 0.5]

    # Aperiodic
    exponents = [-1, -2]
    ratio = [0.2, 0.5]

    # Total number of simulations:
    #   len(list(product(freqs, n_cycles, rdsym, exponents, ratio)))

    positive_examples = np.zeros(
        (int(n_sims * pos_whole_ratio), int(n_seconds * fs)), dtype=np.float32
    )
    negative_examples = np.zeros(
        (n_sims - int(n_sims * pos_whole_ratio), int(n_seconds * fs)), dtype=np.float32
    )

    # params = enumerate(product(freqs, n_cycles, rdsym, exponents, ratio))
    params = [None] * n_sims

    ground_truth = []
    param_opt = []

    for i in range(int(pos_whole_ratio * n_sims)):

        nstime = time.time_ns()
        np.random.seed(nstime % (2**32))

        # features for an asymmetric sinusoidal burst and powerlaw noise
        _freq = math.floor((np.random.random() * (freqs[1] - freqs[0])) + freqs[0])
        _n_cycles = (
            math.floor((np.random.rand() * (n_cycles[1] - n_cycles[0]))) + n_cycles[0]
        )
        _rdsym = (np.random.rand() * (rdsym[1] - rdsym[0])) + rdsym[0]
        _ratio = (np.random.rand() * (ratio[1] - ratio[0])) + ratio[0]
        _exp = (np.random.rand() * (exponents[1] - exponents[0])) + exponents[0]
        seconds_burst = _n_cycles / _freq
        # simulate the burst
        osc = sim_oscillation(
            seconds_burst,
            fs,
            _freq,
            cycle="asine",
            rdsym=_rdsym,
            variance=(_ratio),
            mean=0,
        )
        # randomly select the start time of the burst
        burst_start = int(
            np.random.choice(
                np.arange(0, n_samples - (seconds_burst * fs) - 1), size=1
            )[0]
        )
        # generate the noise
        sig = sim_powerlaw(n_seconds, fs, exponent=_exp, variance=_ratio, mean=0)

        # Combine
        sig[burst_start : burst_start + len(osc)] += osc

        sig = sig.astype(np.float32)
        positive_examples[i] = sig
        ground_truth.append([burst_start, burst_start + len(osc)])

        for i in range(n_sims - int(n_sims * pos_whole_ratio)):

            nstime = time.time_ns()
            np.random.seed(nstime % (2**32))

            # modified
            _freq = math.floor((np.random.random() * (freqs[1] - freqs[0])) + freqs[0])
            _n_cycles = (
                math.floor((np.random.rand() * (n_cycles[1] - n_cycles[0])))
                + n_cycles[0]
            )

            _rdsym = (np.random.rand() * (rdsym[1] - rdsym[0])) + rdsym[0]
            _ratio = (np.random.rand() * (ratio[1] - ratio[0])) + ratio[0]
            _exp = (np.random.rand() * (exponents[1] - exponents[0])) + exponents[0]
            seconds_burst = _n_cycles / _freq

            # Aperiodic
            sig = sim_powerlaw(n_seconds, fs, exponent=_exp, variance=_ratio, mean=0)

            sig = sig.astype(np.float32)
            negative_examples[i] = sig

    for i in range(int(pos_whole_ratio * n_sims)):
        # self-refreshing progress bar. define function then call it with i and n_sims as arguments
        def printProgressBar(
            iteration,
            total,
            prefix="",
            suffix="",
            decimals=1,
            length=100,
            fill="█",
            printEnd="\r",
        ):
            # erase previous progress bar
            percent = ("{0:.1f}").format(100 * (i / float(n_sims * pos_whole_ratio)))
            filledLength = int(length * i // (n_sims * 0.8))
            bar = fill * filledLength + "-" * (length - filledLength)
            print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)

        printProgressBar(i, n_sims, prefix="Progress:", suffix="Complete", length=50)

        # Code contributed by Ryan
        # Function to fit
        def fit(thresh, thresh_inc, freq, freq_inc, x=None, y=None):
            y_pred = detect_bursts_dual_threshold(
                x, 500, (thresh, thresh + thresh_inc), (freq, freq + freq_inc)
            )
            loss = (
                y_pred[: y[0]].sum()
                + y_pred[y[1] + 1 :].sum()
                + (1 - y_pred[y[0] : y[1] + 1]).sum()
            )
            return loss

        # Optimize for first example:
        f = lambda params: fit(*params, x=positive_examples[i], y=ground_truth[i])

        res = gp_minimize(
            f,
            [(0.1, 2), (0.5, 3), (5, 20), (1, 5)],
            acq_func="EI",
            n_calls=15,
            n_initial_points=50,
            n_random_starts=5,
            noise=0.1**2,
            random_state=1234,
        )

        # Get prediction
        thresh, thresh_inc, freq, freq_inc = res.x
        y_pred = detect_bursts_dual_threshold(
            positive_examples[i],
            500,
            (thresh, thresh + thresh_inc),
            (freq, freq + freq_inc),
        )
        # print(y_pred)

        bounds_list = [0, 0]
        found_onset = False
        found_offset = False
        for j in range(len(y_pred)):
            if not found_onset and y_pred[j] == 1:
                bounds_list[0] = j
                found_onset = True
            if found_onset and not found_offset and y_pred[i] == 0:
                bounds_list[1] = j
            if j == len(y_pred) - 1 and found_onset and not found_offset:
                bounds_list[1] = j
        bounds = np.array(bounds_list)

        # plt.figure(figsize=(14, 3))
        # plt.plot(positive_examples[i])
        # plt.axvline(bounds[0], color='C1', label='predicted')
        # plt.axvline(bounds[1], color='C1')
        # plt.axvline(ground_truth[i][0], color='C2')
        # plt.axvline(ground_truth[i][1], color='C2', label='true')
        # plt.legend()

        # plt.show()
        # END CODE CONTRIBUTED BY RYAN

        # bounds = np.array(bounds_list)
        param_opt.append(res.x)

    data = [positive_examples.tolist(), param_opt, ground_truth]
    file = open("dualthresh.csv", "w")
    json.dump(data, file, default=convert_to_serializable)
    file.close()

    return positive_examples, param_opt, ground_truth
