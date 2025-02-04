# This is heavliy biased in YOLO's favor because
# YOLO is trained on very similar training data.
# BUT we don't have more ground truth data.


import sklearn.metrics
import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from math import ceil

# YOLO11 imports
import os
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

#for filtering
from neurodsp.filt import filter_signal

# for bycycle

from bycycle import Bycycle
from neurodsp.spectral import compute_spectrum
from neurodsp.burst import detect_bursts_dual_threshold, compute_burst_stats
from neurodsp.plts.time_series import plot_time_series, plot_bursts
import pandas as pd
import warnings

#dualthresh
import specparam


num_real_sigs = 49


def reverse_order(j, order):
    for i in range(len(order)):
        if j == order[i]:
            return i


# Returns input for ROC_AUC algorithm (probabilistic attempt)
# to use:
# give X=ground_truth burst array of booleans
# give Y=labeler-predicted burst array of booleans
#
# Returns [(1 - P(x_i=False | y_i=False), P(x_i=True | y_i=True)]) given boolean lists X,Y
# OR the algorithm returns an integral Error code.
#   Code 1: len(X)!=len(Y)
#   Code 2: X[i] or Y[i] is not boolean
def probability_correct_selection(X, Y):
    num_true = 0
    num_false = 0
    selected_true = 0
    selected_false = 0

    if len(X) != len(Y):
        return 1
    for i in range(len(X)):
        xi = X[i]
        yi = Y[i]
        if type(xi) != bool or type(yi) != bool:
            print(type(xi), type(yi))
            return 2
        if xi:
            num_true += 1
            if yi:
                selected_true += 1
        else:
            num_false += 1
            if not yi:
                selected_false += 1

    P_0 = float(selected_false)/float(num_false)
    P_1 = float(selected_true)/float(num_true)

    return [1-P_0, P_1]

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

    if len(detections_for_signal)==0:
        return []
    # naive solution, O(n) runtime. Good.
    combine_table = np.full(len(detections_for_signal) - 1, False)
    len_table = len(combine_table)
    for i in range(len_table):
        last_start = detections_for_signal[i][0]
        last_end = detections_for_signal[i][1]
        next_start = detections_for_signal[i + 1][0]
        next_end = detections_for_signal[i+1][1]
        if next_start <= last_end or last_start <= next_end:
            combine_table[i] = True

    for i in range(len_table):
        if combine_table[len_table - i - 1]:
            detections_for_signal[len_table - i-1][0] = min(
                detections_for_signal[len_table - i - 1][0],
                detections_for_signal[len_table - i][0],
            )
            detections_for_signal[len_table - i-1][1] = max(
                detections_for_signal[len_table - i - 1][1],
                detections_for_signal[len_table - i][1],
            )
            # we want to do longest common interval, for each interval in the set. So we want to take each interval out of the bag at some point.
            detections_for_signal.pop(len_table - i)


def create_signal_images(signal_data, output_directory):
    """
    Save signal data as cropped images with specific requirements.

    Parameters:
        signal_data (list or array-like): A list of signals, where each signal is an array of amplitude values.
        output_directory (str): Directory where the images will be saved.
    """
    dpi = 100
    figsize_width = 1000.0 / float(dpi)
    figsize_height = 1.0

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    else:
        print(f"Directory {output_directory} already exists. Skipping image creation.")
        return

    for i, signal in enumerate(signal_data):
        if i % 100 == 0:
            print(i)

        # Normalize signal only if its full y-axis isn't in range [-3, 3]
        signal_min, signal_max = np.min(signal), np.max(signal)
        if signal_min < -3 or signal_max > 3:
            signal = (signal - signal_min) / (
                signal_max - signal_min
            ) * 5.8 - 2.9  # Normalize to [-3, 3]

        filename = f"sig_{i}.png"
        filepath = os.path.join(output_directory, filename)

        # Create the plot
        fig = plt.figure(figsize=(figsize_width, figsize_height), dpi=dpi)
        plt.ylim(-3, 3)

        # Remove axes and internal padding
        plt.gca().set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Plot the signal
        plt.plot(signal)
        plt.savefig(
            filepath,
            bbox_inches="tight",  # Crop tightly to the plot content
            pad_inches=0,  # Remove any padding
            transparent=False,  # Optional: Save with a transparent background
        )
        plt.close(fig)

        # Crop the image
        img = Image.open(filepath)
        box = (45, 0, 955, 100)  # Define the cropping box
        img = img.crop(box)
        img.save(filepath)

        print(f"Saved cropped signal image to: {filepath}")


def is_burst_to_window_bounds(is_burst):
    retVal=[]
    curr_pair = [-1,-1]
    pair_active=False
    for i in range(len(is_burst)):
        if not pair_active and is_burst[i]:
            pair_active=True
            curr_pair[0]=i
        
        elif pair_active and not is_burst[i]:
            curr_pair[1]=i
            retVal.append(curr_pair)
            pair_active=False
    
    if pair_active:
        curr_pair[1]=len(is_burst)
        retVal.append(curr_pair)
        
    return retVal


# Load data
with open("./voyteklabstudy-default-rtdb-export.json") as f:
    results = json.load(f)



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

# print("name is %s" % name)
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
y_pred = [[0, 0]]*len(ground_truth)
num_pred = 0

# def find_overlap_intervals(selection_0, selection_1):

# Here we do not have burst labeling code because humans labeled the signals on a study labeling platform

num_classes = 2
# to index: probability_pairs[labeler_index][signal_index]
probability_pairs = np.empty(
    (len(ground_truth), len(who), num_classes), dtype=object)
ground_truths = np.empty(
    (len(ground_truth), len(who), num_classes), dtype=object)

round_robin_array = np.zeros(
    (len(ground_truth), len(results['sigs']['sig_'+str(0)])))

y_true = np.array(results["ground_truth"])
human_f1_scores=np.zeros(len(ground_truth))
yolo_f1_scores = np.zeros(len(ground_truth))
bycycle_f1_scores = np.zeros(len(ground_truth))
dualthresh_f1_scores = np.zeros(len(ground_truth))

for i in range(0, len(ground_truth)):
    for j in range(0, len(who)):
        ground_truths[i][j][0:2] = [0, 1]

# Gets F1 scores for humans
for i in range(len(ground_truth)):
    human_selections = []
    curr_sig_idx = i+num_real_sigs
    eeg_signal_profiled_in_this_loop = results['sigs']['sig_'+str(
        i+num_real_sigs)]
    
    # iterates through each human labeler
    for j in range(len(who)):
        order = np.array(results["selections"][who[j]]['indices'])
        selections = np.array(results["selections"][who[j]]["selections"])
        reverse_search_sig_idx = reverse_order(i + num_real_sigs, order)
        selections_indexed_by_labeler = selections[reverse_search_sig_idx]

        len_curr_sig = len(eeg_signal_profiled_in_this_loop)

        onset = selections_indexed_by_labeler[0]
        offset = selections_indexed_by_labeler[1]
        if onset != -1 and offset != -1:
            human_selections.append([onset, offset])
    
    merge_burst_selections(human_selections)

    burst_sels = np.zeros(len_curr_sig)
    gt_sels = np.zeros(len_curr_sig)
    for burst in human_selections:
        for subIndex in range(burst[0], burst[1]+1):
            burst_sels[subIndex] = 1
    for subIndex in range(ground_truth[i][0], ground_truth[i][1]+1):
        gt_sels[subIndex] = 1


    human_f1_scores[i]=sklearn.metrics.f1_score(y_true=gt_sels, y_pred = burst_sels)
    print("f1 score for signal %d is %f" % (i, human_f1_scores[i]))
    print(y_true[i])


plt.figure("f1 scores")
# new subplot
plt.subplot(1, 4, 1)
plt.title("Human F1 Scores")
plt.ylim(0, 1)
plt.boxplot(human_f1_scores)

# makes images of signals
# loads the model
# makes predictions
# gives f1 scores for yolo
fs=1000
test_dir = "signal_images_ground_truth"

simulated_sims = [results['sigs']['sig_'+str(i+num_real_sigs)] for i in range(len(ground_truth))]
for i in range(len(simulated_sims)):
    simulated_sims[i] = filter_signal(np.array(simulated_sims[i]), fs, 'lowpass', 30, n_seconds=.2, remove_edges=False)
create_signal_images(simulated_sims, test_dir)
all_images = [
    os.path.join(test_dir, f"sig_{i}.png")
    for i in range(len(ground_truth))
]

model = YOLO(
    "/Users/kenton/HOME/coding/python/publish_the_paper/runs/detect/train50/weights/best.pt"
)

# Annotate each selected image and record the predicted onsets and offsets
annotated_images = []
for i in range(len(all_images)):
    image_path = all_images[i]
    # Predict results for the image
    prediction_results = model.predict(source=image_path, conf=0.2)

    # Load the image using PIL
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Process YOLO model predictions
    yolo_intervals = []
    for r in prediction_results:
        found = False

        for box in r.boxes.data:
            # Extract bounding box and class information
            x1, y1, x2, y2, confidence, class_id = box.tolist()
            class_name = model.names[
                int(class_id)
            ]  # Get class name using model's class names
            if class_id == 1:
                print(f"box bounds: {x1}, {x2}")
                if x1<0:
                    print(f"here {x1} {i}")
                if x2 < 0:
                    print(f"here {x2} {i}")
                yolo_intervals.append([x1,x2])
                found = True
            else:
                continue

    merge_burst_selections(yolo_intervals)
    print(yolo_intervals)
    burst_sels = np.zeros(len_curr_sig)
    gt_sels = np.zeros(len_curr_sig)
    for burst in yolo_intervals:
        for subIndex in range(int(burst[0]), int((burst[1]+1))):
            burst_sels[subIndex] = 1
    for subIndex in range(int(ground_truth[i][0]*.910), int(((ground_truth[i][1])*.910)+1)):
        gt_sels[subIndex] = 1

    yolo_f1_scores[i]=sklearn.metrics.f1_score(y_true=gt_sels, y_pred = burst_sels)
    print("f1 score for signal %d is %f" % (i, yolo_f1_scores[i]))
    print(y_true[i])


plt.subplot(1, 4, 2)
plt.title("YOLO F1 Scores")
plt.ylim(0, 1)
plt.boxplot(yolo_f1_scores)



#Bycycle
warnings.filterwarnings("ignore", category=FutureWarning)
# Frequency band of interest
f_alpha = (5, 80)

# convert results["sigs"][string_key] to numpy arrays
np_sigs = np.zeros((len(ground_truth), 1000))
for i in range(len(ground_truth)):
    np_sigs[i] = np.array(results["sigs"]["sig_"+str(i+num_real_sigs)])

# # Apply lowpass filter to each signal
for idx, value in enumerate(np_sigs):
    np_sigs[idx] = filter_signal(value, fs, 'lowpass', 30, n_seconds=.2, remove_edges=False)

#get burst bool array

# df_features.head()

for i,gt in enumerate(ground_truth):
# Tuned burst detection parameters
    thresholds = {
        'amp_fraction': .2,
        'amp_consistency': .5,
        'period_consistency': .5,
        'monotonicity': .9,
        'min_n_cycles': 2
}
    b = Bycycle(thresholds=thresholds)
    b.fit(np_sigs[i], 500, f_alpha)

    # Recompute cycles on edges of bursts with reduced thresholds
    b.recompute_edges(.01)

    # # Add group and subject ids to dataframes
    # groups = ['patient' if idx >= int(num_real_sigs/2) else 'control' for idx in range(len(ground_truth))]
    # subject_ids = [idx for idx in range(len(ground_truth))]

    # for idx, group in enumerate(groups):
    #     b.df_features[idx]['group'] = group
    #     b.df_features[idx]['subject_id'] = subject_ids[idx]

    # Concatenate the list of dataframes
    df_features = b.df_features

    max_index = b.df_features.shape[0]
    bycycle_selections = []
    for j in range(max_index):
        row = b.df_features.iloc[j]  # Change index as needed

        # Extract the last and next trough sample indices
        last_trough = row["sample_last_trough"]
        next_trough = row["sample_next_trough"]

        if row["is_burst"]:
            bycycle_selections.append([last_trough, next_trough])

    merge_burst_selections(bycycle_selections)
    burst_sels = np.zeros(len_curr_sig)
    gt_sels = np.zeros(len_curr_sig)
    for burst in bycycle_selections:
        for subIndex in range(int(burst[0]), int((burst[1]+1))):
            burst_sels[subIndex] = 1
    for subIndex in range(int(gt[0]), int(gt[1]+1)):
        gt_sels[subIndex] = 1

    bycycle_f1_scores[i]=sklearn.metrics.f1_score(y_true=gt_sels, y_pred = burst_sels)
    
    print(f"bycycle f1 score for signal {i} is {bycycle_f1_scores[i]}")

plt.subplot(1, 4, 3)
plt.title("ByCycle F1 Scores")
plt.ylim(0, 1)
plt.boxplot(bycycle_f1_scores)


# Dual Threshold
# Setup process for dual threshold burst detection
y_pred = [[]]*num_real_sigs
num_pred = 0
for i in range(num_real_sigs):
    # Here we have code to execute the burst labeling.
    #using same sigs from bycycle setup
    test_signal = np_sigs[i]

    freqs, power_spectral_density = compute_spectrum(fs=500, sig=test_signal)
    sm = specparam.SpectralModel(peak_width_limits=[1.0, 8.0], max_n_peaks=8)
    sm.fit(freqs, power_spectrum=power_spectral_density)
    [center_frequency, log_power, bandwidth] = specparam.analysis.get_band_peak(
        sm, [10, 20], select_highest=True
    )
    print(center_frequency)

    is_burst = detect_bursts_dual_threshold(
        sig=test_signal, fs=1000, f_range=(10, 20), dual_thresh=(1, 2)
    )
    # plot_bursts(np.linspace(0, 1, 1000), test_signal, is_burst)

    intervals = is_burst_to_window_bounds(is_burst)

    y_pred[i]=intervals
    burst_sels = np.zeros(len_curr_sig)
    gt_sels = np.zeros(len_curr_sig)
    for j,boolVal in enumerate(is_burst):
        if boolVal:
            burst_sels[j]=1
    for subIndex in range(int(gt[0]), int(gt[1]+1)):
        gt_sels[subIndex] = 1

    dualthresh_f1_scores[i]=sklearn.metrics.f1_score(y_true=gt_sels, y_pred = burst_sels)
    
    
    print(f"dualthresh f1 score for signal {i} is {dualthresh_f1_scores[i]}")
    


plt.subplot(1, 4, 4)
plt.title("Dualthresh F1 Scores")
plt.ylim(0, 1)
plt.boxplot(dualthresh_f1_scores)


plt.show()
plt.close()
