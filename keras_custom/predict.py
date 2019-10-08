import argparse
import os
import sys

import numpy as np
from keras.models import load_model

from .data_loaders.SpectrogramGenerator import SpectrogramGenerator

# Language labels from ISO 639-1 Code
# https://www.loc.gov/standards/iso639-2/php/code_list.php
class_labels = ["en", "de", "fr", "es", "zh", "ru"]


def predict(cli_args, verbose=False):

    config = {"pixel_per_second": 50, "input_shape": [129, 500, 1], "num_classes": 6}
    data_generator = SpectrogramGenerator(
        cli_args.input_file, config, shuffle=False, run_only_once=True
    ).get_generator()
    data = [np.divide(image, 255.0) for image in data_generator]
    # data = [np.divide(image, 255.0) for i, image in enumerate(data_generator) if i % 2 == 0]
    data = np.stack(data)

    # Model Generation
    model = load_model(cli_args.model_dir)

    probabilities = model.predict(data)

    classes = np.argmax(probabilities, axis=1)
    average_prob = np.mean(probabilities, axis=0)
    average_class = int(np.argmax(average_prob))

    if verbose:
        print(classes, class_labels[average_class], average_prob)

    return average_prob, probabilities


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model_dir", required=True)
    parser.add_argument("--input", dest="input_file", required=True)
    cli_args = parser.parse_args()

    if not os.path.isfile(cli_args.input_file):
        sys.exit("Input is not a file.")

    predict(cli_args)
