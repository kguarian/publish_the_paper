# purpose of this file is to create a dataset of images that can beused to
# train a machine learning model to detect the presence of a burst in a
# signal.

from itertools import product, cycle
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

import math
import os
import time

from neurodsp.sim.periodic import sim_oscillation
from neurodsp.sim.aperiodic import sim_powerlaw

# from json_objects_yolov11 import AreaSelection, Annotations, TrainingSample, scale_selection, generate_ml_training_data

from parallel_data_annotation import generate_ml_training_data
from multiprocessing import freeze_support

from PIL import Image

if __name__ == "__main__":
    feature_list = pd.DataFrame()

    # directory for ml training data
    training_data_directory = "compact_training_data_pn"
    test_data_directory = "test_data_pn"

    # General
    n_seconds = 2
    fs = 500
    n_sims = 7000
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

    positive_examples = np.zeros((n_sims, int(n_seconds * fs)), dtype=np.float32)
    negative_examples = np.zeros((n_sims, int(n_seconds * fs)), dtype=np.float32)

    # params = enumerate(product(freqs, n_cycles, rdsym, exponents, ratio))
    params = [None] * n_sims

    ground_truth = []

    for i in range(n_sims):

        nstime = time.time_ns()
        np.random.seed(nstime % (2**32))

        #features for an asymmetric sinusoidal burst and powerlaw noise
        _freq = math.floor((np.random.random() * (freqs[1] - freqs[0])) + freqs[0])
        _n_cycles = math.floor((np.random.rand() * (n_cycles[1] - n_cycles[0]))) + n_cycles[0]
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
            variance=(1 - _ratio),
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

    for i in range(n_sims):

        nstime = time.time_ns()
        np.random.seed(nstime % (2**32))

        # modified
        _freq = math.floor((np.random.random() * (freqs[1] - freqs[0])) + freqs[0])
        _n_cycles = (
            math.floor((np.random.rand() * (n_cycles[1] - n_cycles[0]))) + n_cycles[0]
        )

        _rdsym = (np.random.rand() * (rdsym[1] - rdsym[0])) + rdsym[0]
        _ratio = (np.random.rand() * (ratio[1] - ratio[0])) + ratio[0]
        _exp = (np.random.rand() * (exponents[1] - exponents[0])) + exponents[0]
        seconds_burst = _n_cycles / _freq

        # Aperiodic
        sig = sim_powerlaw(n_seconds, fs, exponent=_exp, variance=_ratio, mean=0)

        sig = sig.astype(np.float32)
        negative_examples[i] = sig

    #     # Plot
    #     plt.figure(i, figsize=(14, 3))
    #     plt.plot(sig)
    #     # plt.title(" ".join([str(j) for j in [_freq, _n_cycles, _rdsym, _exp, _ratio]]))
    #     print("for plot %d\tfreq: %s| cycles: %d| rdsym: %f| exp: %f| ratio: %f|" %
    #           (i, _freq, _n_cycles, _rdsym, _exp, _ratio))
    #     plt.title("freq: %s| cycles: %d| rdsym: %f| exp: %f| ratio: %f|" %
    #               (_freq, _n_cycles, _rdsym, _exp, _ratio))
    #     params[i] = [_freq, _n_cycles, _rdsym, _exp, _ratio]

    # plt.show()

    sig_dict = {"sigs": {}, "n_sigs": n_sims, "users": {}, "params": {}}

    for i in range(n_sims):
        # plt.figure()
        # plt.plot(sigs[i])

        sig_dict["sigs"]["sig_"] = positive_examples[i].tolist()
        sig_dict["ground_truth"] = ground_truth
        sig_dict["params"][i] = params[i]
    print(sig_dict)

    data_annotations = {}

    # # code in original sim code to save the signals for display on webpage.
    # with open("signals_2024_22_4.json", "w") as f:
    #     json.dump(sig_dict, f)

    # code for saving the signals as images and generating json for ml training
    dpi = 100
    figsize_width = 1000.0 / float(dpi)
    figsize_height = 1.0

    # train_size = int(0.75*n_sims)
    train_size = n_sims
    generate_ml_training_data(
        training_data_directory,
        train_size,
        positive_examples[:train_size],
        negative_examples[:train_size],
        ground_truth[:train_size],
    )
