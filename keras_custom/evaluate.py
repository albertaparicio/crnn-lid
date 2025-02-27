import argparse

import data_loaders
import numpy as np
import yaml
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


def equal_error_rate(y_true, probabilities):

    y_one_hot = to_categorical(y_true)
    fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), probabilities.ravel())
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    return eer


def metrics_report(y_true, y_pred, probabilities, label_names=None):

    available_labels = range(0, len(label_names))

    print("Accuracy %s" % accuracy_score(y_true, y_pred))
    print("Equal Error Rate (avg) %s" % equal_error_rate(y_true, probabilities))
    print(classification_report(y_true, y_pred, labels=available_labels, target_names=label_names))
    print(confusion_matrix(y_true, y_pred, labels=available_labels))


def evaluate(cli_args):

    config = yaml.load(open(cli_args.config, "rb"), Loader=yaml.FullLoader)

    # Load Data + Labels
    dataset_dir = (
        config["test_data_dir"] if cli_args.use_test_set else config["validation_data_dir"]
    )

    DataLoader = getattr(data_loaders, config["data_loader"])
    data_generator = DataLoader(dataset_dir, config)

    # Model Generation
    model: Model = load_model(cli_args.model_dir)
    # print(model.summary())

    print("Before predict_generator")
    probabilities = model.predict_generator(
        data_generator.get_data(should_shuffle=False, is_prediction=True),
        steps=data_generator.get_num_files(),
        workers=1,  # parallelization messes up data order. careful!
        max_queue_size=config["batch_size"],
        use_multiprocessing=True,
    )
    print("After predict_generator")

    print("Presenting metrics report")
    y_pred = np.argmax(probabilities, axis=1)
    y_true = data_generator.get_labels()[: len(y_pred)]
    metrics_report(y_true, y_pred, probabilities, label_names=config["label_names"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model_dir", required=True)
    parser.add_argument("--config", dest="config", required=True)
    parser.add_argument("--testset", dest="use_test_set", default=False, type=bool)
    cli_args = parser.parse_args()

    evaluate(cli_args)
