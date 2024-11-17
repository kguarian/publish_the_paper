from itertools import product, cycle
from statistics import linear_regression
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

import math
import os
import time

from neurodsp.sim.periodic import sim_oscillation
from neurodsp.sim.aperiodic import sim_powerlaw

def signal_power(signal):
    retVal = 0
    for i in range(len(signal)):
        retVal+=signal[i]*signal[i]
    retVal/=len(signal)
    return retVal

feature_list = pd.DataFrame()

# General
n_seconds = 2
fs = 500
n_sims = 50
n_samples = int(n_seconds * fs)

# Periodic
freqs = [5, 10]
n_cycles = [3, 6]
rdsym = [.2, .5]

# Aperiodic
exponents = [-1, -2]
ratio = [.2, .5]

# Total number of simulations:
#   len(list(product(freqs, n_cycles, rdsym, exponents, ratio)))

sigs = np.zeros((50, int(n_seconds * fs)), dtype=np.float32)

# params = enumerate(product(freqs, n_cycles, rdsym, exponents, ratio))
params = [None]*n_sims

ground_truth = []
ratio_snr_pairs = [None]*n_sims
plt.figure()
for i in range(n_sims):

    nstime = time.time_ns()
    np.random.seed(nstime % (2**32))

    # modified
    _freq = math.floor((np.random.random()*(freqs[1]-freqs[0])) + freqs[0])
    _n_cycles = math.floor(
        (np.random.rand()*(n_cycles[1]-n_cycles[0]))) + n_cycles[0]

    _rdsym = (np.random.rand()*(rdsym[1]-rdsym[0])) + rdsym[0]
    _ratio = (np.random.rand()*(ratio[1]-ratio[0])) + ratio[0]
    _exp = (np.random.rand()*(exponents[1]-exponents[0]))+exponents[0]
    seconds_burst = _n_cycles / _freq

    # I don't think "mean" corresponds to a param, but it ain't broke.
    signal = sim_oscillation(seconds_burst, fs, _freq, cycle='asine',
                             rdsym=_rdsym, variance=(1-_ratio), mean=0)

    burst_start = int(
        np.random.choice(
            np.arange(0, n_samples - (seconds_burst * fs) - 1), size=1)[0]
    )

    # Aperiodic
    noise = sim_powerlaw(n_seconds, fs, exponent=_exp, variance=_ratio, mean=0)

    # Evaluate SNR
    pow_signal = signal_power(signal)
    pow_noise = signal_power(noise)
    snr = pow_signal/pow_noise
    snr = 10*math.log10(snr)
    ratio_snr_pairs[i]=(_ratio, snr)

    plt.plot(snr,_ratio,  'ro')

ratio_snr_pairs.sort(key=lambda pair: pair[0])
linreg = linear_regression([x[0] for x in ratio_snr_pairs], [x[1] for x in ratio_snr_pairs])
    
print(linreg)

plt.show()
plt.close()