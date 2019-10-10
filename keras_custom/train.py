import argparse
import os
import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping
from tensorflow.keras.metrics import Recall, Precision, Accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import data_loaders as data_loaders
import models as models


# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("error", category=RuntimeWarning)
# from tensorflow.compat.v1 import logging
# logging.set_verbosity(logging.ERROR)


def train(cli_args, log_dir):

    config = yaml.load(open(cli_args.config, "rb"), Loader=yaml.FullLoader)
    if config is None:
        print("Please provide a config.")

    # Load Data + Labels
    DataLoader = getattr(data_loaders, config["data_loader"])

    train_data_generator = DataLoader(config["train_data_dir"], config)
    validation_data_generator = DataLoader(config["validation_data_dir"], config)

    # Training Callbacks
    checkpoint_filename = os.path.join(log_dir, "weights.{epoch:02d}.model")
    model_checkpoint_callback = ModelCheckpoint(
        checkpoint_filename, save_best_only=True, verbose=1, monitor="val_accuracy"
    )

    tensorboard_callback = TensorBoard(log_dir=log_dir, write_images=True)
    csv_logger_callback = CSVLogger(os.path.join(log_dir, "log.csv"))
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="min"
    )

    # Model Generation
    model_class = getattr(models, config["model"])
    model: Model = model_class.create_model(train_data_generator.get_input_shape(), config)
    # print(model.summary())

    optimizer = Adam(lr=config["learning_rate"], decay=1e-6)
    # optimizer = RMSprop(lr=config["learning_rate"], rho=0.9, epsilon=1e-08, decay=0.95)
    # optimizer = SGD(lr=config["learning_rate"], decay=1e-6, momentum=0.9, clipnorm=1, clipvalue=10)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy", "recall", "precision"])

    if cli_args.weights:
        model.load_weights(cli_args.weights)

    # Training
    # config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    # off = rewriter_config_pb2.RewriterConfig.OFF
    # config_proto.graph_options.rewrite_options.arithmetic_optimization = off
    #
    # sess = tf.Session(config=config_proto)
    # tf.keras.backend.set_session(sess)

    history = model.fit_generator(
        train_data_generator.get_data(),
        steps_per_epoch=train_data_generator.get_num_files(),
        epochs=config["num_epochs"],
        verbose=1,
        callbacks=[
            model_checkpoint_callback,
            tensorboard_callback,
            csv_logger_callback,
            early_stopping_callback,
        ],
        validation_data=validation_data_generator.get_data(should_shuffle=False),
        validation_steps=validation_data_generator.get_num_files(),
        max_queue_size=config["batch_size"],
        workers=8,
        use_multiprocessing=True,
    )

    # Do evaluation on model with best validation accuracy
    best_epoch = np.argmax(history.history["val_accuracy"])
    print("Log files: ", log_dir)
    print("Best epoch: ", best_epoch + 1)
    model_file_name = checkpoint_filename.replace("{epoch:02d}", "{:02d}".format(best_epoch + 1))

    return model_file_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", dest="weights")
    parser.add_argument("--config", dest="config", default="config.yaml")
    cli_args = parser.parse_args()

    log_dir = os.path.join("logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    print("Logging to {}".format(log_dir))

    # copy models & config for later
    shutil.copytree("models", log_dir)  # creates the log_dir
    shutil.copy(cli_args.config, log_dir)

    model_file_name = train(cli_args, log_dir)
    print(f'Best model saved at "{model_file_name}"')

    # DummyCLIArgs = namedtuple("DummyCLIArgs", ["model_dir", "config", "use_test_set"])
    # evaluate(DummyCLIArgs(model_file_name, cli_args.config, False))

    exit()
