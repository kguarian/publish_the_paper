import json
from typing import Collection

import matplotlib
import numpy as np
import os

from PIL import Image
from matplotlib import pyplot as plt

import yaml

CLASS_ID_NON_BURST = 0
CLASS_ID_BURST = 1


class AreaSelection:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class Annotations:
    def __init__(self, class_id: str, coordinates: AreaSelection):
        self.label = class_id
        self.coordinates = coordinates

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class TrainingSample:
    def __init__(self, filename: str, annotation: object):
        self.image: str = filename
        if isinstance(annotation, Collection):
            self.annotations = annotation
        elif isinstance(annotation, Annotations):
            self.annotations = [annotation]
        else:
            self.annotations = []
            print("Error: annotation is not a list or Annotations object")
            print(type(annotation))

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


def write_training_sample_yolo(ts: TrainingSample):
    if not os.path.exists(ts.image):
        print(f"Image {ts.image} does not exist")
        return

    if not ts.annotations:
        print(f"No annotations for image {ts.image}")
        return

    label_file = ts.image.replace("/images/", "/labels/").replace(".png", ".txt")
    label_directory = os.path.dirname(label_file)
    if not os.path.exists(label_directory):
        os.makedirs(label_directory)

    with open(label_file, "w") as f:
        for annotation in ts.annotations:
            f.write(
                f"{annotation.label} {annotation.coordinates.x} {annotation.coordinates.y} {annotation.coordinates.width} {annotation.coordinates.height}\n"
            )


def selection_to_training_sample(selection, signal):
    FULL_WIDTH = float(955)
    annots = []
    if selection != None:

        sel0_float_normalized = float(selection[0]) / FULL_WIDTH
        sel1_float_normalized = float(selection[1]) / FULL_WIDTH

        center_sel0_normalized = sel0_float_normalized / 2
        width_sel0_normalized = sel0_float_normalized

        center_sel1_normalized = (sel0_float_normalized + sel1_float_normalized) / 2.0
        width_sel1_normalized = sel1_float_normalized - sel1_float_normalized

        center_sel2_normalized = (sel1_float_normalized + 1.0) / 2.0
        width_sel2_normalized = 1.0 - sel1_float_normalized

        annots = [
            Annotations(
                class_id=CLASS_ID_NON_BURST,
                coordinates=AreaSelection(
                    x=center_sel0_normalized,
                    y=0.5,
                    width=width_sel0_normalized,
                    height=1.0,
                ),
            ),
            Annotations(
                class_id=CLASS_ID_BURST,
                coordinates=AreaSelection(
                    x=center_sel1_normalized,
                    y=0.5,
                    width=width_sel1_normalized,
                    height=1.0,
                ),
            ),
            Annotations(
                class_id=CLASS_ID_NON_BURST,
                coordinates=AreaSelection(
                    x=center_sel2_normalized,
                    y=0.5,
                    width=width_sel2_normalized,
                    height=1.0,
                ),
            ),
        ]
    else:
        center_sel0_normalized = 0.5
        width_sel0_normalized = 1.0
        annots = [
            Annotations(
                class_id=CLASS_ID_NON_BURST,
                coordinates=AreaSelection(
                    x=center_sel0_normalized,
                    y=0.5,
                    width=width_sel0_normalized,
                    height=1.0,
                ),
            )
        ]

    training_sample = TrainingSample(filename=signal, annotation=annots)
    return training_sample


def scale_selection(selection):
    return [int((910.0 / 1000.0) * timepoint) for timepoint in selection]


def generate_ml_training_data(
    directory,
    n_sims,
    positive_examples,
    negative_examples,
    ground_truth,
    train_size=0.7,
    val_size=0.2,
    test_size=0.1,
):

    training_samples = []
    val_samples = []
    test_samples = []

    # Create the directories
    for subset in ["train/images", "train/labels", "val/images", "val/labels", "test/images", "test/labels"]:
        os.makedirs(os.path.join(directory, subset), exist_ok=True)

    dpi = 100
    figsize_width = 1000.0 / float(dpi)
    figsize_height = 1.0
    for i in range(n_sims):
        if i % 100 == 0:
            print(i)

        filename = f"sig_{i}.png"
        if i < (n_sims * train_size):
            filepath = f"{directory}/train/images/{filename}"
        elif i < (n_sims * (train_size + val_size)):
            filepath = f"{directory}/val/images/{filename}"
        else:
            filepath = f"{directory}/test/images/{filename}"

        fig = plt.figure(figsize=(figsize_width, figsize_height), dpi=100)
        plt.ylim(-3, 3)

        plt.gca().set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.plot(positive_examples[i])
        plt.savefig(filepath, bbox_inches="tight", pad_inches=0, transparent=False)
        plt.close(fig)

        img = Image.open(filepath)
        box = (45, 0, 955, 100)
        img = img.crop(box)
        img.save(filepath)

        scaled_selection = scale_selection(ground_truth[i])
        training_sample = selection_to_training_sample(scaled_selection, filepath)
        training_samples.append(training_sample)

        write_training_sample_yolo(training_sample)

        filename = f"sig_{2*i + 1}.png"

        if i < (n_sims * train_size):
            filepath = f"{directory}/train/images/{filename}"
        elif i < (n_sims * (train_size + val_size)):
            filepath = f"{directory}/val/images/{filename}"
        else:
            filepath = f"{directory}/test/images/{filename}"

        fig = plt.figure(figsize=(figsize_width, figsize_height), dpi=100)
        plt.ylim(-3, 3)

        plt.gca().set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.plot(negative_examples[i])
        plt.savefig(filepath, bbox_inches="tight", pad_inches=0, transparent=False)
        plt.close(fig)

        img = Image.open(filepath)
        box = (45, 0, 955, 100)
        img = img.crop(box)
        img.save(filepath)

        training_sample = selection_to_training_sample(None, filepath)
        training_samples.append(training_sample)

        write_training_sample_yolo(training_sample)

    # Generate YAML file
    yaml_path = os.path.join(directory, "dataset.yaml")
    yaml_data = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 2,  # Number of classes
        "names": ["non-burst", "burst"],  # Class names
    }

    with open(yaml_path, "w") as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)

    print(f"Training data and YAML file generated at {directory}")