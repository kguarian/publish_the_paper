# %% [markdown]
# Imports

# %%
from statistics import LinearRegression
import json
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy import stats

import pandas as pd
import sklearn
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model

import os
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import random

# %% [markdown]
# analysis functions


# %%
# the simulated signals were presented in randomized order to each human labeler. The ordering was recorded as a list of indices.
# the selections the labelers made were recorded in the order they were made.
# This function finds the index at which the desired signal index appears in the ordering of signal indices.
def reverse_order_search(j, order):
    for i in range(len(order)):
        if j == order[i]:
            return i


# the linear coefficients come from a linear regression model searching for the best fit line between
# signal-noise ratio and _ratio parameter used to generate signals
def ratio_to_snr_converter(_ratio):
    snr = -19.65 * (_ratio) + 9.668
    return snr


# this decodes ONE entry in the list of params used to generate one signal.
# the function returns a dictionary with named parameters for visual inspection
# this function is only used when investigating the signal with the worst f1 score.
def decode_params(param_list) -> dict | None:
    if len(param_list) != 5:
        return None
    retDict = {}
    retDict["freq"] = 2 * param_list[0]
    retDict["n_cycles"] = param_list[1]
    retDict["rise-decay asymmetry"] = param_list[2]
    retDict["aperiodic exponent"] = param_list[3]
    retDict["signal-noise ratio"] = ratio_to_snr_converter(param_list[4])
    return retDict


# this decodes ONE entry in the list of params used to generate one signal.
# the function returns a numpy array with the same parameters as decode_params.
# this function is used when preparing data for regression analysis.
# It differs from decode_params in that it returns a numpy array instead of a dictionary.
def decode_params_np(param_list) -> dict | None:
    if len(param_list) != 5:
        return None
    retArray = np.zeros(5)
    retArray[0] = 2 * param_list[0]
    retArray[1] = param_list[1]
    retArray[2] = param_list[2]
    retArray[3] = param_list[3]
    retArray[4] = ratio_to_snr_converter(param_list[4])
    return retArray


# this function takes a list of parameters and returns a dictionary with named parameters.
# the function is used when preparing data for regression analysis.
# this function calls decode_params to ensure frequency and snr are accurate.
def param_list_to_training_data(param_list):
    num_samples = len(param_list)
    num_features = 5
    retArray = np.zeros((num_samples, num_features))
    for i in range(len(param_list)):
        row = decode_params_np(param_list[i])
        retArray[i][0:num_features] = row[:num_features]
    return retArray


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


# %% [markdown]
# Import File Data

# %%

# Load data from results json exported from firebase
with open("../../voyteklabstudy-default-rtdb-export.json") as f:
    results = json.load(f)

# %% [markdown]
# Set constants

# %%


# this is the number of real recorded EEG signal we used in the study platform we hosted. The
# signals were arranged in (real signals, simulated signals) order. Thus, num_real_sigs is used
# as an array offset in this analysis.
num_real_sigs = 49

# List of names of human collaborators who labeled data
# length of which gives us number of labelings. Allows us to iterate through labelers
who = list(results["selections"].keys())
print(who)

"""
y_pred and y_true are variables for auc_roc
  we want to analyze humans as a whole.
  so y_pred is the interval the humans select (on average?)
  let's start with just one lab person.
"""
y_pred = np.zeros((num_real_sigs, len(results["sigs"]["sig_" + str(0)])))
y_true = np.zeros((num_real_sigs, len(results["sigs"]["sig_" + str(0)])))

# the classes are "non-bursting" and "bursting"
num_classes = 2


threshold = 0.7

onsets = np.zeros(num_real_sigs)
offsets = np.zeros(num_real_sigs)

# Here we want to find the average selection onset and offset. We can handle outliers and such later -- now we need to scaffold.

for curr_sig_idx in range(num_real_sigs):
    eeg_signal_profiled_in_this_loop = results["sigs"]["sig_" + str(curr_sig_idx)]
    no_labels = 0

    for j in range(len(who)):
        order = np.array(results["selections"][who[j]]["indices"])
        selections = np.array(results["selections"][who[j]]["selections"])
        reverse_search_sig_idx = reverse_order_search(curr_sig_idx, order)
        selections_indexed_by_labeler = selections[reverse_search_sig_idx]

        len_curr_sig = len(eeg_signal_profiled_in_this_loop)

        onsets[curr_sig_idx] += selections_indexed_by_labeler[0]
        offsets[curr_sig_idx] += selections_indexed_by_labeler[1]
        if selections_indexed_by_labeler[0] == -1:
            print("nosel made")
            no_labels += 1

    onsets[curr_sig_idx] = onsets[curr_sig_idx] / (len(who)-no_labels)
    offsets[curr_sig_idx] = offsets[curr_sig_idx] / (len(who)-no_labels)



print(onsets)
print(offsets)

# Here we want to generate an image dataset from the signals.

test_dir = "signal_images"
collection_real_sigs = [results["sigs"]["sig_" + str(i)] for i in range(num_real_sigs)]
create_signal_images(collection_real_sigs, test_dir)

# now we want to predict the onset and offset of the signals with yolo.

output_collage = f"collage_with_boxes_and_borders_conf_{threshold}.png"

# Load the model
model = YOLO(
    "/Users/kenton/HOME/coding/python/publish_the_paper/runs/detect/train50/weights/best.pt"
)

# Get all image files in the directory
all_images = [
    os.path.join(test_dir, f)
    for f in os.listdir(test_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

# Optional: Load a font for better text rendering
try:
    font = ImageFont.truetype(
        "arial.ttf", size=16
    )  # Use a font installed on your system
except IOError:
    font = ImageFont.load_default()

# Annotate each selected image and record the predicted onsets and offsets
pred_onsets = np.zeros(num_real_sigs)
pred_offsets = np.zeros(num_real_sigs)
annotated_images = []

rands = np.random.randint(0, len(who), 5)
for i in rands:
    image_path = all_images[i]
    # Predict results for the image
    results = model.predict(source=image_path, conf=threshold)

    # Load the image using PIL
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Process model predictions
    for r in results:
        for j in range(len(r.boxes.data)):
            box = r.boxes.data[j]
            # Extract bounding box and class information
            x1, y1, x2, y2, confidence, class_id = box.tolist()
            class_name = model.names[
                int(class_id)
            ]  # Get class name using model's class names
            if class_id == 1:
                print(f"box bounds: {x1}, {x2}")
                pred_onsets[i] = x1
                pred_offsets[i] = x2

            # Draw the bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

            # Create a label
            label = f"{class_name} ({confidence:.2f})"

            # Draw label inside the bounding box
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Position text inside the bounding box, adjusted to fit
            label_x = min(max(x1, 0) + 2, 910 - text_width)
            label_y = (text_bbox[3] - text_bbox[1])*j + 2

            # Draw label background and text
            draw.rectangle(
                [label_x, label_y, label_x + text_width, label_y + text_height],
                fill="red",
            )
            draw.text((label_x, label_y), label, fill="white", font=font)

    # Add a black border around the image
    border_size = 5
    bordered_image = Image.new(
        "RGB",
        (image.width + 2 * border_size, image.height + 2 * border_size),
        color="black",
    )
    bordered_image.paste(image, (border_size, border_size))

    # Ensure the image is resized to 910x100
    resized_image = bordered_image.resize((910, 100))  # Natural resolution
    annotated_images.append(resized_image)

# Determine collage dimensions
collage_width = 910  # Each image's width
collage_images_per_row = 1  # Number of images per row
collage_rows = ceil(len(annotated_images) / collage_images_per_row)
collage_height = collage_rows * 100  # 100 pixels per image height

# Create the blank collage canvas
collage = Image.new(
    "RGB", (collage_width * collage_images_per_row, collage_height), color="white"
)

# Paste each image into the collage
for i, annotated_image in enumerate(annotated_images):
    row = i // collage_images_per_row
    col = i % collage_images_per_row
    x_offset = col * 910
    y_offset = row * 100
    collage.paste(annotated_image, (x_offset, y_offset))

# Save the collage
collage.save(output_collage)
print(f"Collage saved to {output_collage}")
collage.show()


threshold = 0.2
annotated_images = []
output_collage = f"collage_with_boxes_and_borders_conf_{threshold}.png"
for i in rands:
    image_path = all_images[i]
    # Predict results for the image
    results = model.predict(source=image_path, conf=threshold)

    # Load the image using PIL
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Process model predictions
    for r in results:
        for j in range(len(r.boxes.data)):
            box = r.boxes.data[j]
            # Extract bounding box and class information
            x1, y1, x2, y2, confidence, class_id = box.tolist()
            class_name = model.names[
                int(class_id)
            ]  # Get class name using model's class names
            if class_id == 1:
                print(f"box bounds: {x1}, {x2}")
                pred_onsets[i] = x1
                pred_offsets[i] = x2

            # Draw the bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

            # Create a label
            label = f"{class_name} ({confidence:.2f})"

            # Draw label inside the bounding box
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Position text inside the bounding box, adjusted to fit
            label_x = min(max(x1, 0) + 2, 910 - text_width)
            label_y = (text_bbox[3] - text_bbox[1])*j + 2

            # Draw label background and text
            draw.rectangle(
                [label_x, label_y, label_x + text_width, label_y + text_height],
                fill="red",
            )
            draw.text((label_x, label_y), label, fill="white", font=font)

    # Add a black border around the image
    border_size = 5
    bordered_image = Image.new(
        "RGB",
        (image.width + 2 * border_size, image.height + 2 * border_size),
        color="black",
    )
    bordered_image.paste(image, (border_size, border_size))

    # Ensure the image is resized to 910x100
    resized_image = bordered_image.resize((910, 100))  # Natural resolution
    annotated_images.append(resized_image)

# Determine collage dimensions
collage_width = 910  # Each image's width
collage_images_per_row = 1  # Number of images per row
collage_rows = ceil(len(annotated_images) / collage_images_per_row)
collage_height = collage_rows * 100  # 100 pixels per image height

# Create the blank collage canvas
collage = Image.new(
    "RGB", (collage_width * collage_images_per_row, collage_height), color="white"
)

# Paste each image into the collage
for i, annotated_image in enumerate(annotated_images):
    row = i // collage_images_per_row
    col = i % collage_images_per_row
    x_offset = col * 910
    y_offset = row * 100
    collage.paste(annotated_image, (x_offset, y_offset))

# Save the collage
collage.save(output_collage)
print(f"Collage saved to {output_collage}")
collage.show()
