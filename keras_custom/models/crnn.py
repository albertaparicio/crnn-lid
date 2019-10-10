from tensorflow.keras.layers import (
    Dense,
    Permute,
    Reshape,
    Convolution2D,
    BatchNormalization,
    MaxPooling2D,
    Bidirectional,
    LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

NAME = "CRNN"


def create_model(input_shape, config, is_training=True):

    weight_decay = 0.001

    model = Sequential()

    model.add(
        Convolution2D(
            64, 3, 3, W_regularizer=l2(weight_decay), activation="relu", input_shape=input_shape
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # (bs, y, x, c) --> (bs, x, y, c)
    model.add(Permute((2, 1, 3)))

    # (bs, x, y, c) --> (bs, x, y * c)
    bs, x, y, c = model.layers[-1].output_shape
    model.add(Reshape((x, y * c)))

    model.add(Bidirectional(LSTM(256, return_sequences=False), merge_mode="concat"))
    model.add(Dense(config["num_classes"], activation="softmax"))

    return model
