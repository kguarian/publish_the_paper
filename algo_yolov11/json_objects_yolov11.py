import json
from typing import Collection

import matplotlib
import numpy as np
import os

from PIL import Image
from matplotlib import pyplot as plt

import yaml
from multiprocessing import Pool

CLASS_ID_NON_BURST = 0
CLASS_ID_BURST = 1


class AreaSelection:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    # https://stackoverflow.com/questions/3768895/how-to-make-a-class-json-serializable
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
    # write the training sample to a file in the format required by YOLO
    # https://docs.ultralytics.com/modes/train/
    # https://docs.ultralytics.com/modes/train/#train-annotations

    # check if the image exists
    if not os.path.exists(ts.image):
        print(f"Image {ts.image} does not exist")
        return

    # check if the annotations exist
    if not ts.annotations:
        print(f"No annotations for image {ts.image}")
        return

    # create a directory for the image
    image_directory = os.path.dirname(ts.image)
    
    # create a directory for the labels
    label_directory = ts.image.replace("images", "labels")
    label_directory = os.path.dirname(label_directory)
    if not os.path.exists(label_directory):
        print("SCREAM 2")
        print(label_directory)
        os._exit(1)

    label_file = os.path.join(
        label_directory, os.path.basename(ts.image).replace(".png", ".txt")
    )
    with open(label_file, "w") as f:
        for annotation in ts.annotations:
            f.write(
                f"{annotation.label} {annotation.coordinates.x} {annotation.coordinates.y} {annotation.coordinates.width} {annotation.coordinates.height}\n"
            )


# type =0 means negative example
# type =1 means positive example


def selection_to_training_sample(selection, signal):
    FULL_WIDTH = float(955)
    annots = []
    if selection != None:

        sel0_float_normalized = float(selection[0]) / FULL_WIDTH
        sel1_float_normalized = float(selection[1]) / FULL_WIDTH

        # center of sel0 (non-burst) is half of sel0 because sel0 is the left bound of a burst, so right bound of a non-burst.
        # Then we divide by full width to normalize.
        center_sel0_normalized = sel0_float_normalized / 2
        width_sel0_normalized = sel0_float_normalized

        # center of sel1 (burst) is the average of sel1 and sel2 because sel1 is the left bound of a burst, and sel2 is the right bound of a burst.
        # Then we divide by full width to normalize.
        center_sel1_normalized = (sel0_float_normalized + sel1_float_normalized) / 2.0
        width_sel1_normalized = sel1_float_normalized - sel0_float_normalized

        # center of sel2 (non-burst) is the average of sel1 and 955 because sel2 is the right bound of a burst and the right bound of the signal is 955 after cropping.
        # Then we divide by full width to normalize.
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


# creates 2*n_sims training samples. half of them are positive examples, the other half are negative examples


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

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create the images directory if it doesn't exist
    if not os.path.exists(f"{directory}/images"):
        os.makedirs(f"{directory}/images")
    # Create the labels directory if it doesn't exist
    if not os.path.exists(f"{directory}/labels"):
        os.makedirs(f"{directory}/labels")

    # Create the images directories if they don't exist
    if not os.path.exists(f"{directory}/images/train"):
        os.makedirs(f"{directory}/images/train")
    if not os.path.exists(f"{directory}/images/val"):
        os.makedirs(f"{directory}/images/val")
    if not os.path.exists(f"{directory}/images/test"):
        os.makedirs(f"{directory}/images/test")

    # Create the labels directories if they don't exist
    if not os.path.exists(f"{directory}/labels/train"):
        os.makedirs(f"{directory}/labels/train")
    if not os.path.exists(f"{directory}/labels/val"):
        os.makedirs(f"{directory}/labels/val")
    if not os.path.exists(f"{directory}/labels/test"):
        os.makedirs(f"{directory}/labels/test")

    dpi = 100
    figsize_width = 1000.0 / float(dpi)
    figsize_height = 1.0
    p = Pool()
    for i in range(n_sims):
        if i % 100 == 0:
            print(i)
        # for i in range(1):
        # refactor this into another function, and call it twice. This will be difficult to read.
        filename = f"sig_{i}.png"
        if i < (n_sims * train_size):
            filepath = f"{directory}/images/train/{filename}"
        elif i < (n_sims * (train_size + val_size)):
            filepath = f"{directory}/images/val/{filename}"
        else:
            filepath = f"{directory}/images/test/{filename}"
        fig = plt.figure(figsize=(figsize_width, figsize_height), dpi=100)
        plt.ylim(-3, 3)

        # Remove axes AFTER plotting
        plt.gca().set_axis_off()  # Turn off axis for the current plot
        # Remove internal figure padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.plot(positive_examples[i])


        plt.savefig(
            filepath,
            bbox_inches="tight",  # Crop tightly to the plot content
            pad_inches=0,  # Remove any padding around the saved image
            transparent=False,  # Optional: Save with a transparent background
        )

        plt.close(fig)
        img = Image.open(filepath)
        box = (45, 0, 955, 100)
        img = img.crop(box)
        img.save(filepath)

        # Create the annotation
        scaled_selection = scale_selection(ground_truth[i])# Create the annotation
        training_sample = selection_to_training_sample(scaled_selection, filename)
        if i < (n_sims * train_size):
            training_sample.image = f"{directory}/images/train/{filename}"
            training_samples.append(training_sample)
        elif i < (n_sims * (train_size + val_size)):
            training_sample.image = f"{directory}/images/val/{filename}"
            val_samples.append(training_sample)
        else:
            training_sample.image = f"{directory}/images/test/{filename}"
            test_samples.append(training_sample)

        # write the training sample to a file in the format required by YOLO
        write_training_sample_yolo(training_sample)

        filename = f"sig_{2*i + 1}.png"
        
        if i < (n_sims * train_size):
            filepath = f"{directory}/images/train/{filename}"
        elif i < (n_sims * (train_size + val_size)):
            filepath = f"{directory}/images/val/{filename}"
        else:
            filepath = f"{directory}/images/test/{filename}"
        fig = plt.figure(figsize=(figsize_width, figsize_height), dpi=100)
        plt.ylim(-3, 3)

        # Remove axes AFTER plotting
        plt.gca().set_axis_off()  # Turn off axis for the current plot
        # Remove internal figure padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.plot(negative_examples[i])
        plt.savefig(
            filepath,
            bbox_inches="tight",  # Crop tightly to the plot content
            pad_inches=0,  # Remove any padding around the saved image
            transparent=False,  # Optional: Save with a transparent background
        )
        plt.close(fig)
        img = Image.open(filepath)
        box = (45, 0, 955, 100)
        img = img.crop(box)
        # img.save(filepath)
        if i < (n_sims * train_size):
            img.save(f"{directory}/images/train/{filename}")
        elif i < (n_sims * (train_size + val_size)):
            img.save(f"{directory}/images/val/{filename}")
        else:
            img.save(f"{directory}/images/test/{filename}")

        # Create the annotation
        training_sample = selection_to_training_sample(None, filename)
        if i < (n_sims * train_size):
            training_sample.image = f"{directory}/images/train/{filename}"
            training_samples.append(training_sample)
        elif i < (n_sims * (train_size + val_size)):
            training_sample.image = f"{directory}/images/val/{filename}"
            val_samples.append(training_sample)
        else:
            training_sample.image = f"{directory}/images/test/{filename}"
            test_samples.append(training_sample)

        # write the training sample to a file in the format required by YOLO
        write_training_sample_yolo(training_sample)


    # Generate YAML file
    yaml_path = os.path.join(directory, "dataset.yaml")
    yaml_data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,  # Number of classes
        "names": ["non-burst", "burst"],  # Class names
    }

    with open(yaml_path, "w") as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)

    # json.dump(
    #     training_samples,
    #     open(directory + "/annotation.json", "w"),
    #     default=lambda o: o.__dict__,
    #     sort_keys=True,
    #     indent=4,
    # )
