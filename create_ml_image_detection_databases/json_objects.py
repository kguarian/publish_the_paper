import json
from typing import Collection

import matplotlib
import numpy as np
import os

from PIL import Image
from matplotlib import pyplot as plt


class AreaSelection:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    # https://stackoverflow.com/questions/3768895/how-to-make-a-class-json-serializable
    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4)


class Annotations:
    def __init__(self, label: str, coordinates: AreaSelection):
        self.label = label
        self.coordinates = coordinates

    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4)


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
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4)


def save_training_samples(training_samples, filename):
    with open(filename, 'w') as file:
        out = json.dumps(
            training_samples,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4
        )
        file.write(out)

# type =0 means negative example
# type =1 means positive example


def selection_to_training_sample(selection, signal):
    annots = []
    if selection!=None:
        annots = [
            Annotations(
                label="non-burst",
                coordinates=AreaSelection(
                    x=selection[0]/2, y=50, width=selection[0], height=100
                )
            ),
            Annotations(
                label="burst",
                coordinates=AreaSelection(
                    x=(selection[0]+selection[1])/2, y=50, width=selection[1]-selection[0], height=100
                )
            ),
            Annotations(
                label="non-burst",
                coordinates=AreaSelection(
                    x=(selection[1]+955)/2, y=50, width=955-selection[1], height=100
                )
            )
        ]
    else :
        annots = [
            Annotations(
                label="non-burst",
                coordinates=AreaSelection(
                    x=955/2, y=50, width=955, height=100
                )
            )
        ]

    training_sample = TrainingSample(
        filename=signal, annotation=annots)
    return training_sample


def scale_selection(selection):
    return [int((910./1000.)*timepoint) for timepoint in selection]

# creates 2*n_sims training samples. half of them are positive examples, the other half are negative examples


def generate_ml_training_data(directory, n_sims, positive_examples, negative_examples, ground_truth, training_samples):

    dpi = 100
    figsize_width = (1000./float(dpi))
    figsize_height = 1.0
    for i in range(n_sims):
        if i%100==0:
            print(i)
        # for i in range(1):
        # refactor this into another function, and call it twice. This will be difficult to read.
        filename = f"sig_{i}.png"
        filepath = os.path.join(directory, filename)
        fig = plt.figure(figsize=(figsize_width, figsize_height), dpi=100)
        plt.ylim(-3, 3)

        # Remove axes AFTER plotting
        plt.gca().set_axis_off()  # Turn off axis for the current plot
        # Remove internal figure padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.plot(positive_examples[i])
        plt.savefig(
            filepath,
            bbox_inches='tight',  # Crop tightly to the plot content
            pad_inches=0,         # Remove any padding around the saved image
            transparent=False      # Optional: Save with a transparent background
        )

        plt.close(fig)
        img = Image.open(filepath)
        box = (45, 0, 955, 100)
        img = img.crop(box)
        img.save(filepath)

        # Create the annotation
        scaled_selection = scale_selection(ground_truth[i])
        training_sample = selection_to_training_sample( scaled_selection, filename)
        training_samples.append(training_sample)

        filename = f"sig_{2*i + 1}.png"
        filepath = os.path.join(directory, filename)
        fig = plt.figure(figsize=(figsize_width, figsize_height), dpi=100)
        plt.ylim(-3, 3)

        # Remove axes AFTER plotting
        plt.gca().set_axis_off()  # Turn off axis for the current plot
        # Remove internal figure padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.plot(negative_examples[i])
        plt.savefig(
            filepath,
            bbox_inches='tight',  # Crop tightly to the plot content
            pad_inches=0,         # Remove any padding around the saved image
            transparent=False      # Optional: Save with a transparent background
        )
        plt.close(fig)
        img = Image.open(filepath)
        box = (45, 0, 955, 100)
        img = img.crop(box)
        img.save(filepath)

        # Create the annotation
        training_sample = selection_to_training_sample(None, filename)
        training_samples.append(training_sample)

    json.dump(training_samples, open(directory+'/annotation.json', 'w'),
              default=lambda o: o.__dict__, sort_keys=True, indent=4)
