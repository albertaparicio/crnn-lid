from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model

NAME = "resnet"


def create_model(input_shape, config):

    input_tensor = Input(shape=input_shape)  # this assumes K.image_dim_ordering() == 'tf'
    resnet_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)
    print(resnet_model.summary())

    x = resnet_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(config["num_classes"], activation="softmax")(x)

    return Model(input=resnet_model.input, output=predictions)
