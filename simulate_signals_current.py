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
    seconds_burst=_n_cycles / _freq

    # I don't think "mean" corresponds to a param, but it ain't broke.
    osc=sim_oscillation(seconds_burst, fs, _freq, cycle='asine',
                          rdsym=_rdsym, variance=(1-_ratio), mean=0)

    burst_start=int(
        np.random.choice(
            np.arange(0, n_samples - (seconds_burst * fs) - 1), size=1)[0]
    )

    # Aperiodic
    sig=sim_powerlaw(n_seconds, fs, exponent=_exp, variance=_ratio, mean=0)

    # Combine
    # I think this is really neat because we avoid indexing osc.
    sig[burst_start:burst_start+len(osc)] += osc

    sig=sig.astype(np.float32)
    sigs[i]=sig
    ground_truth.append([burst_start, burst_start+len(osc)])

    # Plot
    plt.figure(i, figsize=(14, 3))
    plt.plot(sig)
    # plt.title(" ".join([str(j) for j in [_freq, _n_cycles, _rdsym, _exp, _ratio]]))
    print("for plot %d\tfreq: %s| cycles: %d| rdsym: %f| exp: %f| ratio: %f|" %
              (i, _freq, _n_cycles, _rdsym, _exp, _ratio))
    plt.title("freq: %s| cycles: %d| rdsym: %f| exp: %f| ratio: %f|" %
              (_freq, _n_cycles, _rdsym, _exp, _ratio))
    params[i]=[_freq, _n_cycles, _rdsym, _exp, _ratio]

plt.show()
assert np.all(sigs[-1] == sig)

sig_dict={"sigs": {},
            "n_sigs": len(sigs),
            "users": {},
            "params": {}}

for i in range(len(sigs)):
    # plt.figure()
    # plt.plot(sigs[i])

    # 49 because we have 49 real signals that appear first.
    sig_dict["sigs"]["sig_"+str(i+49)]=sigs[i].tolist()
    sig_dict["ground_truth"]=ground_truth
    sig_dict["params"][i]=params[i]
print(sig_dict)
# plt.show()
with open("signals_2024_22_4.json", "w") as f:
    json.dump(sig_dict, f)
