# purpose of this file is to create a dataset of images that can beused to
# train a machine learning model to detect the presence of a burst in a
# signal.

from itertools import product, cycle
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, brute

import math
import os
import time

from neurodsp.sim.periodic import sim_oscillation
from neurodsp.sim.aperiodic import sim_powerlaw

from dualthresh_model import DualThreshModel

# from json_objects_yolov11 import AreaSelection, Annotations, TrainingSample, scale_selection, generate_ml_training_data

from parallel_data_annotation import generate_ml_training_data
from multiprocessing import freeze_support
from PIL import Image

model = DualThreshModel()
fs=1000

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
    n_sims = 30
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

    positive_examples = np.zeros((int(n_sims*pos_whole_ratio), int(n_seconds * fs)), dtype=np.float32)
    negative_examples = np.zeros((n_sims-int(n_sims*pos_whole_ratio), int(n_seconds * fs)), dtype=np.float32)

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

        for i in range(n_sims-int(n_sims*pos_whole_ratio)):

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


    # for i in range(int(pos_whole_ratio * n_sims)):
    #     # print(positive_examples[i].shape)
    #     # res=brute(
    #     #     func=dualthresh_loss(signal=positive_examples[i], ground_truth=ground_truth[i], fs=fs), # this is a call to `loss_fn` in `dualthresh_loss`
    #     #     ranges=((freqs[0], freqs[1]), (freqs[0]+freqs[1], freqs[1]+50), (0, 1), (1, 50)),
    #     #     args=[fs],
    #     #     Ns=30,
    #     #     finish=None,
    #     # )
    #     res = minimize(
    #         fun=dualthresh_loss(signal=positive_examples[i], ground_truth=ground_truth[i], fs=fs), # this is a call to `loss_fn` in `dualthresh_loss`
    #         x0=[freqs[0], freqs[1], 0, 1],
    #         args=[fs],
    #         method="Newton-CG",
    #     )

    #     param_opt.append(res.x)

    return positive_examples, ground_truth
