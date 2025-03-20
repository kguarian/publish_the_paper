import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from neurodsp.burst import detect_bursts_dual_threshold
from sim_dualthresh_ryan import generate_training_data_approximator
from skopt import gp_minimize

fs = 500

x_test, y_test = generate_training_data_approximator()
x_train, y_train = generate_training_data_approximator()

y_test = np.array(y_test)
y_train = np.array(y_train)

# Function to fit
def fit(thresh, thresh_inc, freq, freq_inc, x=None, y=None):
    y_pred = detect_bursts_dual_threshold(x, 500, (thresh, thresh+thresh_inc), (freq, freq+freq_inc))
    loss = y_pred[:y[0]].sum() + y_pred[y[1]+1:].sum() + (1-y_pred[y[0]:y[1]+1]).sum()
    return loss

# Optimize for first example:
f = lambda params:  fit(*params, x=x_train[0], y=y_train[0])

res = gp_minimize(
    f,
    [(0.1, 2), (0.5, 3), (5, 20), (1, 5)],
    acq_func="EI",
    n_calls=15,
    n_random_starts=5,
    noise=0.1**2,
    random_state=1234)

# Get prediction
thresh, thresh_inc, freq, freq_inc = res.x
y_pred = detect_bursts_dual_threshold(x_train[8], 500, (thresh, thresh+thresh_inc), (freq, freq+freq_inc))
print(y_pred)

bounds_list=[0,0]
found_onset = False
found_offset = False
for i in range(len(y_pred)):
    if not found_onset and y_pred[i]==1:
        bounds_list[0] = i
        found_onset = True
    if found_onset and not found_offset and y_pred[i]==0:
        bounds_list[1] = i
bounds = np.array(bounds_list)


# Plot result
plt.figure(figsize=(14, 3))
plt.plot(x_train[0])
plt.plot(y_pred, label='predicted')
plt.axvline(y_train[0][0], color='C2')
plt.axvline(y_train[0][1], color='C2', label='true')
plt.legend()

plt.show()