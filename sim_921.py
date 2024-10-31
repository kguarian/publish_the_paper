from itertools import product, cycle
import json
import matplotlib.pyplot as plt
import numpy as np

from neurodsp.sim.periodic import sim_oscillation
from neurodsp.sim.aperiodic import sim_powerlaw

# General
n_seconds = 2
fs = 500
n_samples = int(n_seconds * fs)

# Periodic
freqs = [5, 20]
n_cycles = [3, 6]
rdsym = [.2, .5]

# Aperiodic
exponents = [-1, -2]
ratio = [.2, .35, .5]

# Total number of simulations:
#   len(list(product(freqs, n_cycles, rdsym, exponents, ratio)))

sigs = np.zeros((48, int(n_seconds * fs)), dtype=np.float32)
                
params = enumerate(product(freqs, n_cycles, rdsym, exponents, ratio))

ground_truth = []

for i, (_freq, _n_cycles, _rdsym, _exp, _ratio) in params:

    np.random.seed(i)

    # Periodic
    seconds_burst = _n_cycles / _freq
    
    osc = sim_oscillation(seconds_burst, fs, _freq, cycle='asine',
                          rdsym=_rdsym, variance=(1-_ratio), mean=0)


    burst_start = int(
        np.random.choice(np.arange(0, n_samples - (seconds_burst * fs) - 1), size=1)[0]
    )

    # Aperiodic
    sig = sim_powerlaw(n_seconds, fs, exponent=_exp, variance=_ratio, mean=0)
    
    # Combine
    sig[burst_start:burst_start+len(osc)] += osc

    sig = sig.astype(np.float32)
    sigs[i] = sig
    ground_truth.append([burst_start, burst_start+len(osc)])

    # Plot
    # plt.figure(i, figsize=(14, 3))
    # plt.plot(sig)
    # plt.title(" ".join([str(j) for j in [_freq, _n_cycles, _rdsym, _exp, _ratio]]))

assert np.all(sigs[-1] == sig)

sig_dict = {"sigs":{},
            "n_sigs":len(sigs),
            "users":{}}

for i in range(len(sigs)):
    # plt.figure()
    # plt.plot(sigs[i])
    sig_dict["sigs"]["sig_"+str(i)] = sigs[i].tolist()
    sig_dict["ground_truth"]=ground_truth
print(sig_dict)
# plt.show()
with open("signals.json", "w") as f:
    json.dump(sig_dict, f)