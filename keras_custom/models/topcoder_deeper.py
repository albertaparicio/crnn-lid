from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Dropout,
    Convolution2D,
    BatchNormalization,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

NAME = "Topcoder_CNN"


def create_model(input_shape, config, is_training=True):

    weight_decay = 0.001

    model = Sequential()

    model.add(
        Convolution2D(
            32, 7, 7, W_regularizer=l2(weight_decay), activation="relu", input_shape=input_shape
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(64, 5, 5, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, W_regularizer=l2(weight_decay), activation="relu"))

    model.add(Dense(config["num_classes"], activation="softmax"))

    return model
