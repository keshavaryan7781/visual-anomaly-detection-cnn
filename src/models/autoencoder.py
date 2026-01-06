import tensorflow
from tensorflow.keras import layers as tfl
from tensorflow.keras import models


def build_encoder(input_shape = (256, 256, 3)):
    inputs = tfl.Input(shape = input_shape)

    x = tfl.Conv2D(32, kernel_size = 3, padding = "same", strides = 1)(inputs)
    x = tfl.BatchNormalization()(x)
    x = tfl.ReLU()(x)
    x = tfl.MaxPooling2D(pool_size = 2)(x)

    x = tfl.Conv2D(64, kernel_size = 3, padding = "same", strides = 1)(x)
    x = tfl.BatchNormalization()(x)
    x = tfl.ReLU()(x)
    x = tfl.MaxPooling2D(pool_size = 2)(x)

    x = tfl.Conv2D(128, kernel_size = 3, padding = "same", strides = 1)(x)
    x = tfl.BatchNormalization()(x)
    x = tfl.ReLU()(x)
    latent = tfl.MaxPooling2D(pool_size = 2)(x)

    encoder = models.Model(inputs, latent, name = "encoder")
    return encoder

def build_decoder(latent_shape = (32, 32, 128)):
    latent = tfl.Input(shape = latent_shape)

    y = tfl.UpSampling2D(size= 2)(latent)
    y = tfl.Conv2D(128, kernel_size = 3, padding = "same", strides = 1)(y)
    y = tfl.BatchNormalization()(y)
    y = tfl.ReLU()(y)

    y = tfl.UpSampling2D(size= 2)(y)
    y = tfl.Conv2D(64, kernel_size = 3, padding = "same", strides = 1)(y)
    y = tfl.BatchNormalization()(y)
    y = tfl.ReLU()(y)

    y = tfl.UpSampling2D(size= 2)(y)
    y = tfl.Conv2D(32, kernel_size = 3, padding = "same", strides = 1)(y)
    y = tfl.BatchNormalization()(y)
    y = tfl.ReLU()(y)

    outputs = tfl.Conv2D(
        filters = 3,
        kernel_size = 3,
        padding = "same",
        activation = "sigmoid"
    )(y)

    decoder = models.Model(latent, outputs, name = "decoder")
    return decoder

def build_autoencoder(input_shape = (256, 256, 3)):
    encoder = build_encoder(input_shape= input_shape)
    decoder = build_decoder(latent_shape= encoder.output_shape[1:])

    inputs = encoder.input
    latent = encoder(inputs)
    outputs = decoder(latent)

    autoencoder = models.Model(inputs, outputs, name = "cnn_autoencoder")
    return autoencoder