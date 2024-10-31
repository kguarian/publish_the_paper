# %% [markdown]
# initialize firebase admin app if not done before

# %%
import firebase_admin
from firebase_admin import credentials

# Import database module.
from firebase_admin import db
import numpy as np
import matplotlib.pyplot as plt

# from bycycle.recon import BycycleRecon
# import bycycle.recon as br
from bycycle.recon.recon_obj import BycycleRecon, DEFAULT_OPT
from bycycle.features import compute_shape_features

from bycycle.burst import detect_bursts_dual_threshold as dualthresh
from sys import exit

# %%

# boilerplate for firebase data retrieval

# if not firebase_admin._apps:
if not firebase_admin._apps:
    cred = credentials.Certificate(
        "/Users/kenton/HOME/coding/node/voytek_site_env/key_dir/bycycle_key/bycycle_key.json"
    )
    firebase_admin.initialize_app(credential=cred)
app = firebase_admin.get_app()
print(app.name)
print(app.project_id)


# %% [markdown]
# get firebase realtime database data necessary for model params optimization

# %%

# Get a database reference to our signals
sigs = db.reference(
    path="sigs", url="https://voyteklabstudy-default-rtdb.firebaseio.com/"
).get()
selections_for_signal = db.reference(
    path="selections", url="https://voyteklabstudy-default-rtdb.firebaseio.com/"
).get()
sig_quantity = db.reference(
    path="n_sigs", url="https://voyteklabstudy-default-rtdb.firebaseio.com/"
).get()
ground_truth = db.reference(
    path="ground_truth", url="https://voyteklabstudy-default-rtdb.firebaseio.com/"
).get()

sig_lengths = np.zeros(sig_quantity)
test_size = 10
sigs_np = np.array(sigs)
sels_np = np.array(selections_for_signal)
ground_truth_np = np.array(ground_truth)

print("sigs shape:         %s" % str(sigs_np.shape))
print("sigs shape:         %s" % str(sels_np.shape))
print("ground truth shape: %s" % str(ground_truth_np.shape))
for i in range(sig_quantity):
    sig_lengths[i] = len(sigs["sig_" + str(i)])


# %%
# for generated signals.
# It's methodologically difficult to say that the ground truth of a real-world signal is.
# params: ground_truth_all, selections_all
#   ground_truth_all: list of tuples (onset, offset) for each signal
#   selections_all: list of lists of tuples (onset, offset) for each user for each signal
# output: fp_all, tp_all, fn_all, tn_all
#   fp_all: list of lists of false positives for each user for each signal
#   tp_all: list of lists of true positives for each user for each signal
#   fn_all: list of lists of false negatives for each user for each signal
#   tn_all: list of lists of true negatives for each user for each signal
#   the first dimension of each of these lists is the signal index. The second dimension is the user index.
def ft_pn(ground_truth_all, selections_all):
    # for i in selections_all:
    # print(selections_all[i])

    # print(selections_all)
    # max_len = max(selections_all, key=lambda x: max([len(i) for i in selections_all[x] if isinstance(i, list)], default=0))
    max_len = 0
    for i, v in enumerate(selections_all):
        # print(i)
        # print(selections_all[i])
        curr_len = 0
        print(selections_all[i])
        for j in selections_all[i]:
            if isinstance(j, list):
                curr_len += 1
        if curr_len > max_len:
            max_len = curr_len
    print(max_len)

    fp_all = np.zeros((len(ground_truth_all), max_len))
    tp_all = np.zeros((len(ground_truth_all), max_len))
    fn_all = np.zeros((len(ground_truth_all), max_len))
    tn_all = np.zeros((len(ground_truth_all), max_len))

    for signal_index in range(len(ground_truth_all)):
        selections_for_signal = selections_all[signal_index]
        # print("sfs: %s"%selections_for_signal)
        ground_truth_for_signal = ground_truth_all[signal_index]
        fp = np.zeros(len(selections_for_signal))
        tp = np.zeros(len(selections_for_signal))
        fn = np.zeros(len(selections_for_signal))
        tn = np.zeros(len(selections_for_signal))
        print("sig_length: %s" % (sig_lengths[signal_index]))
        ground_trutharray = np.zeros(int(sig_lengths[signal_index]))
        onset_true = ground_truth_for_signal[0]
        offset_true = ground_truth_for_signal[1]
        for i in range(onset_true, offset_true):
            ground_trutharray[i] = True
        selection_index = 0
        for user_index in range(len(selections_for_signal)):
            curr_selection = selections_for_signal[user_index]
            # print(selections_for_signal)
            # this happens when the signal was never presented to the user.
            if curr_selection == None:
                continue
            # this happens when the user or algorithm thinks there was no burst.
            # TODO: Test
            if curr_selection[0] == -1:
                # end_tn = sig_lengths[signal_index] - offset_true
                # tn[selection_index] = onset_true + end_tn
                # fn[selection_index] = offset_true - onset_true
                continue
            onset_sel = curr_selection[0]
            offset_sel = curr_selection[1]
            sel_trutharray = np.zeros(int(sig_lengths[signal_index]))
            print(onset_sel, offset_sel)
            for i in range(onset_sel, offset_sel):
                sel_trutharray[i] = True

            for i in range(int(sig_lengths[signal_index])):
                a=ground_trutharray[i]
                b=sel_trutharray[i]

                if a and b:
                    tp[selection_index]+=1
                elif a and (not b):
                    fn[selection_index]+=1
                elif (not a) and b:
                    fp[selection_index]+=1
                else:
                    tn[selection_index]+=1
            selection_index+=1
        for i in range(selection_index):            
            print(fn[i],fp[i],tn[i],tp[i], sep=' then ')
        fp /= sig_lengths[signal_index]
        tp /= sig_lengths[signal_index]
        fn /= sig_lengths[signal_index]
        tn /= sig_lengths[signal_index]

        # print(fp_all[0])
        # sys.exit()
        # print(fp_all.shape)
        # print(fp.shape)
        # print(fp)
        index_add = 0
        print(fp)
        for i in range(selection_index):
            if fp[i] == 0 and tp[i] == 0 and fn[i] == 0 and tn[i] == 0:
                continue
            print(
                "length of fp_all[signal_index] = %s, whereas length of fp = %s"
                % (len(fp_all[signal_index]), len(fp))
            )
            # fp[i] /= sig_lengths[signal_index]
            # tp[i] /= sig_lengths[signal_index]
            # fn[i] /= sig_lengths[signal_index]
            # tn[i] /= sig_lengths[signal_index]

            fp_all[signal_index][i] = fp[i]
            tp_all[signal_index][i] = tp[i]
            fn_all[signal_index][i] = fn[i]
            tn_all[signal_index][i] = tn[i]
    for i in range(len(fp_all)):
        print("index %d:" % (i))
        print(fp_all[i])
        print(tp_all[i])
        print(fn_all[i])
        print(tn_all[i])
    return fp_all, tp_all, fn_all, tn_all
    # print("fp:%s\ntp:%s\nfn:%s\ntn:%s" % (fp_all, tp_all, fn_all, tn_all))


# returns list with length n_signals. each element is a 4-tuple. false positive, true positive, false negative, true negative
def group_by_signal(test_results):
    stats = np.full(len(test_results[0]), None)
    for signal_index in range(len(test_results[0])):
        stats[signal_index] = [
            test_results[0][signal_index],
            test_results[1][signal_index],
            test_results[2][signal_index],
            test_results[3][signal_index],
        ]
    return stats


# output (per user): fp (over all signals), tp (oas), fn (oas), tn (oas)
def group_by_user(test_results):
    #
    # length 4: fp, tp, fn, tn
    print(len(test_results))
    # (48,4): 48 signals, 4 users
    print(test_results[0].shape)
    stats = np.full(len(test_results[0][0]), None)
    # print(test_results[0].shape)
    for user_index in range(len(test_results[0][0])):
        stats[user_index] = [
            test_results[0][:][user_index],
            test_results[1][:][user_index],
            test_results[2][:][user_index],
            test_results[3][:][user_index],
        ]
    return stats


# returns a list with length num_users.
# each element is a 2x4 array.
# for each element, the first row is the mean of the false positive, true positive, false negative, true negative rates for the user.
# the second row is the standard deviation of the false positive, true positive, false negative, true negative rates for the user.
def test_result_user_stats(results_by_user):
    user_stats = np.zeros((len(results_by_user), 2, 4))
    for user_index in range(len(results_by_user)):
        active_indices = np.where(
            (results_by_user[user_index][0] != 0)
            | (results_by_user[user_index][1] != 0)
            | (results_by_user[user_index][2] != 0)
            | (results_by_user[user_index][3] != 0)
        )
        user_fp_avg = np.mean(results_by_user[user_index][0][active_indices])
        user_tp_avg = np.mean(results_by_user[user_index][1][active_indices])
        user_fn_avg = np.mean(results_by_user[user_index][2][active_indices])
        user_tn_avg = np.mean(results_by_user[user_index][3][active_indices])

        user_fp_sd = np.std(results_by_user[user_index][0][active_indices])
        user_tp_sd = np.std(results_by_user[user_index][1][active_indices])
        user_fn_sd = np.std(results_by_user[user_index][2][active_indices])
        user_tn_sd = np.std(results_by_user[user_index][3][active_indices])

        user_stat = np.array(
            [
                np.array([user_fp_avg, user_tp_avg, user_fn_avg, user_tn_avg]),
                np.array([user_fp_sd, user_tp_sd, user_fn_sd, user_tn_sd]),
            ]
        )
        user_stats[user_index] = user_stat
    return user_stats


# %% [markdown]
# Organize data by signal.

# %%
# this section organizes the selections by signal. They are organized by labeler in the selections_for_signal dictionary.
#
selection_dict = {}
# say n labellers labeled some signals. Then, selection_dict is a dictionary with n keys.
keys = selections_for_signal.keys()
# create a list of objects with length n for each signal index.
for i in range(sig_quantity):
    selection_dict[i] = np.full(len(keys), None)

# false/true positive/negative lists
ft_pn_list = []
for index, key in enumerate(keys):
    # print("selection key: %s" % key)
    user_selections = selections_for_signal[key]["selections"]
    indices_selected = selections_for_signal[key]["indices"]
    # print("these are the indices of the signal the labeling participant saw")
    # print(indices_selected)
    # print("these are the left and right bounds of the selection, of the signal the labeling participant saw")
    # print(user_selections)
    for i in range(len(user_selections)):
        selection_dict[indices_selected[i]][index] = user_selections[i]

for i in selection_dict:
    selection_dict[i] = [j for j in selection_dict[i] if j is not None]


# print(selection_dict)

# fp_all, tp_all, fn_all, fp_all have the same format.
# f stands for false, t stands for true, p stands for positive, n stands for negative.
# let `retval` correspond to [x]_all
# fp_all[i][j] is the [x] rate for the jth user on the ith signal.
#
fp_all, tp_all, fn_all, tn_all = ft_pn(
    ground_truth_all=ground_truth, selections_all=selection_dict
)

print(fp_all, tp_all, fn_all, tn_all, sep="\n")

by_user = group_by_user([fp_all, tp_all, fn_all, tn_all])
by_signal = group_by_signal([fp_all, tp_all, fn_all, tn_all])

print("by user: %s\nby_signal: %s" % (by_user, by_signal), sep="\n")


# %% [markdown]
# get stats on user selections

# %%

user_stats = test_result_user_stats(by_user)
print("stats by user: [[means],[stds]]: %s" % user_stats)

# %% [markdown]
# get algorithmic selections


# this function takes an array of booleans and returns an array of 2-tuples
# where the first element of each tuple is the index of the first True value
# of the corresponding burst, and the second element is the index of the last
# TODO: fix formatting from this function. The elements are double bracketed and they need to either be singled or the handling of the double brackets needs to be fixed.
def truth_table_to_intervals(truth_table):
    intervals = []
    curr_interval = None
    for i, val in enumerate(truth_table):
        if val == True:
            if curr_interval == None:
                curr_interval = [i, i]
        else:
            if curr_interval != None:
                curr_interval[1] = i - 1
                intervals.append(curr_interval)
                curr_interval = None
    return intervals


# assuming a truth table goes [false ... false true ... true false ... false], this function returns the interval of true's.
def truth_table_to_single_interval(truth_table):
    curr_interval = None
    for i, val in enumerate(truth_table):
        if val == True:
            if curr_interval == None:
                curr_interval = [i, i]
        else:
            if curr_interval != None:
                curr_interval[1] = i - 1
                return curr_interval
    # if we get here, the burst ends at the end of the signal.
    if curr_interval != None:
        curr_interval[1] = len(truth_table) - 1

    return curr_interval


# %% [markdown]
# create list of sigs

# %%
# plaves all the signals we use for online labeling in an array.
sig_array = np.full(sig_quantity, fill_value=None)
for i in range(sig_quantity):
    sig_array[i] = np.array(sigs["sig_" + str(i)])


# %% [markdown]
# demonstrate usage of dualthresh on a db-extracted signal

# %%

print("sig_array type:\t%s" % type(sig_array))
print("sig_array shape:\t%s" % sig_array.shape)
print("sig_array[0] type:\t%s" % type(sig_array[0]))
print("sig_array[0] shape:\t%s" % sig_array[0].shape)
print("fill value type:\t%s" % type(sig_array[0][0]))

burst_truth_array = dualthresh(
    sig_array[0], fs=500, dual_thresh=(1, 2), f_range=(8, 12)
)

intervals = truth_table_to_intervals(burst_truth_array)


print(intervals)


# %% [markdown]
# run bycycle.burst.detect_bursts_dual_threshold on all the signals.
# have humans select bursts on as many signals as we decide.
# for each signal, write a summary of the humans' selections.
# for each signal, run scipy.optimize.minimize to find the dualthresh parameters that give the closes results to the human selection summary
# for each signal, summarize the signal by its dualthresh burst characteristics. Save the index, signal summary, optimal dualthresh parameters in a dict
# This dict is our model's dat

# %%
from scipy.optimize import minimize
from scipy.stats import ttest_rel

# dual_thresh, f_range
popt_dualthresh = np.zeros((sig_quantity, 4))

# use scipy.optimize.minimize and roc_auc_score to find the optimal dual_thresh and f_range for each signal
# params: sig_array, ground_truth
#   sig_array: array of signals
#   ground_truth: array of tuples (onset, offset) for each signal
# output: popt_dualthresh, popt_f_range
#   popt_dualthresh: array of tuples (onset, offset) for each signal
#   popt_f_range: array of tuples (onset, offset) for each signal


# sig_array is an array of signals (1d arrays)
# params is a list of tuples (dual_thresh, f_range) for each signal (each is a 2-tuple)
def dualthresh_selections_by_signal(sig_array, params):
    selections = np.full((len(sig_array), len(params)), fill_value=None)
    for i, sig in enumerate(sig_array):
        for j, param in enumerate(params):
            burst_truth_array = dualthresh(
                sig, fs=500, dual_thresh=param[0], f_range=param[1]
            )
            intervals = truth_table_to_single_interval(burst_truth_array)
            selections[i][j] = intervals
    return selections


def split_sigarray_into_cycles(sig_array, fs=900):
    cycles = np.full(len(sig_array), fill_value=None)
    for i, sig in enumerate(sig_array):
        df_features = compute_shape_features(sig_array[i], fs=fs, f_range=(8, 12))
        last_troughs = df_features["sample_last_trough"].to_numpy()

        # we want to append that last trough to the end of the array and that's not in last_troughs
        np.append(last_troughs, df_features["sample_next_trough"].iloc[-1])
        left_bound = 0
        cycles[i] = np.full(len(last_troughs), fill_value=None)
        for j, trough in enumerate(last_troughs):
            cycles[i][j] = sig[left_bound:trough]
            left_bound = trough
        print(cycles[i])
        # exit(3)

    return cycles


def bycyclerecon_selections_by_signal(sig_array, params):
    # preprocessing:
    # def minimize_wrapper(self, x, cyc_sim, dtypes, icyc, n_cycles, *params):
    cycles = split_sigarray_into_cycles(sig_array)
    models = list(DEFAULT_OPT.keys())
    print("cycles: ")
    for i, partitioned_sig in enumerate(cycles):
        print("signal %d" % i)
        print(partitioned_sig)
        # bro means bycycle recon object
        print(DEFAULT_OPT.keys())
        # exit(3)
        # is this the correct var?
        models = list(DEFAULT_OPT.keys())
        bro = BycycleRecon(cycles=models)

        print("level 0")
        print(sig_array.shape)
        for j, param in enumerate(params):
            # print("\tlevel 1")
            print(len(param))
            for k, param2 in enumerate(param):
                # print("\t\tlevel 2")
                print(len(param2))
                print(param2)
        print(partitioned_sig.shape)
        # print("usually stops here")
        bro.fit(partitioned_sig)
        print("passed fit")
        models = bro.models
        r_squared_array = [
            models.asine.rsq,
            models.asym_harmonic.rsq,
            models.exp_cos.rsq,
            models.gaussian.rsq,
            models.skewed_gaussian.rsq,
            models.sine.rsq,
            models.sawtooth.rsq,
        ]
        for i in range(len(r_squared_array)):
            print(("length of popt[%d]: " + str(len(r_squared_array[i]))) % i)
            print(r_squared_array[i])
    print("noop")

    # # for indexing
    # base_rsq = r_squared_array[0]
    # max_rsq_length = len(base_rsq)
    # for rsq_idx in range(len(base_rsq)):
    #     for sig_idx in range(len(sig_array)):
    #         max_popt_length = max(max_rsq_length, len(r_squared_array[rsq_idx][sig_idx]))

    # maxes = np.full(len(base_rsq), [None] * max_popt_length)
    # print(r_squared_array)
    # print(maxes)
    # print("done")


# fp, tp, fn, tn are all 2d jagged arrays
# fp[i][j] is the false positive rate for the jth user on the ith signal
# TODO: inspect distributions of fp, tp, fn, tn to see if they are normal
def mean_squares_by_signal(fp, tp, fn, tn):
    fp_mean_squares = np.zeros(len(fp))
    tp_mean_squares = np.zeros(len(tp))
    fn_mean_squares = np.zeros(len(fn))
    tn_mean_squares = np.zeros(len(tn))
    num_sigs = len(fp)
    for i in range(num_sigs):
        fp_mean_squares[i] = np.mean(fp[i]) ** 2
        tp_mean_squares[i] = np.mean(tp[i]) ** 2
        fn_mean_squares[i] = np.mean(fn[i]) ** 2
        tn_mean_squares[i] = np.mean(tn[i]) ** 2
    return [fp_mean_squares, tp_mean_squares, fn_mean_squares, tn_mean_squares]


# params=[[(a,b),(c,d)] for a in range(1,3) for b in range(a+1,4) for c in range(1,3) for d in range(a+1,4)]
params = [
    [(a, b), (c, d)]
    for a in range(1, 3)
    for b in range(a + 1, 4)
    for c in range(5, 12)
    for d in range(c + 1, c + 4)
]

dualthresh_selections = dualthresh_selections_by_signal(sig_array, params)
bycylerecon_selections = bycyclerecon_selections_by_signal(sig_array, params)

# print(out_sel)
algo_ft_pn = ft_pn(ground_truth, dualthresh_selections)
human_ft_pn = ft_pn(ground_truth, selection_dict)
# algo_mean_squares = mean_squares_by_signal(algo_ft_pn[0], algo_ft_pn[1], algo_ft_pn[2], algo_ft_pn[3])
# human_mean_squares = mean_squares_by_signal(human_ft_pn[0], human_ft_pn[1], human_ft_pn[2], human_ft_pn[3])
# t_test_results = ttest_rel(human_mean_squares, algo_mean_squares)
# print(t_test_results)


# %% [markdown]
# testing results from last cell


# %%
for i in range(len(sig_array)):
    ground_truth_for_signal = ground_truth[i]
    print("ground truth for signal %s: %s" % (i, ground_truth_for_signal))
    amount_positives = ground_truth_for_signal[1] - ground_truth_for_signal[0]
    amount_negatives = sig_lengths[i] - amount_positives
    print("signal %d results:" % i)
    # [0][i][j] is the false positives for the ith signal for the jth human/algorithm
    print("human fp: %s" % human_ft_pn[0][i])
    print("dualthresh fp: %s" % algo_ft_pn[0][i])

    print("human tp: %s" % human_ft_pn[1][i])
    print("human tp rate: %s" % (human_ft_pn[1][i] / amount_positives))
    print("dualthresh tp: %s" % algo_ft_pn[1][i])
    print("dualthresh tp rate: %s" % (algo_ft_pn[1][i] / amount_positives))

    print("human fn: %s" % human_ft_pn[2][i])
    print("dualthresh fn: %s" % algo_ft_pn[2][i])

    print("human tn: %s" % human_ft_pn[3][i])
    print("human tn rate: %s" % (human_ft_pn[3][i] / amount_negatives))
    print("dualthresh tn: %s" % algo_ft_pn[3][i])
    print("dualthresh tn rate: %s" % (algo_ft_pn[3][i] / amount_negatives))

    print(
        "human_added_values: %s"
        % (
            human_ft_pn[0][i]
            + human_ft_pn[1][i]
            + human_ft_pn[2][i]
            + human_ft_pn[3][i]
        )
    )
    print(
        "dualthresh_added_values: %s"
        % (algo_ft_pn[0][i] + algo_ft_pn[1][i] + algo_ft_pn[2][i] + algo_ft_pn[3][i])
    )

mean_human_perf_by_signal = np.zeros((len(human_ft_pn[0]), 4))
mean_algo_perf_by_signal = np.zeros((len(algo_ft_pn[0]), 4))

for i in range(len(human_ft_pn[0])):
    mean_human_perf_by_signal[i] = [
        np.mean(human_ft_pn[0][i]),
        np.mean(human_ft_pn[1][i]),
        np.mean(human_ft_pn[2][i]),
        np.mean(human_ft_pn[3][i]),
    ]
    mean_algo_perf_by_signal[i] = [
        np.mean(algo_ft_pn[0][i]),
        np.mean(algo_ft_pn[1][i]),
        np.mean(algo_ft_pn[2][i]),
        np.mean(algo_ft_pn[3][i]),
    ]

print("mean human performance by signal: %s" % mean_human_perf_by_signal)
print("mean dualthresh performance by signal: %s" % mean_algo_perf_by_signal)