import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, Concatenate, Dropout,
    LeakyReLU
)
from tensorflow_addons.layers.spatial_pyramid_pooling import (
    SpatialPyramidPooling2D
)
from tensorflow_addons.layers import InstanceNormalization
from model.layers import conv2d, upsample, sft, condition, sa


def discriminator(img_shape, spectr_norm=False):
    filters = 64

    input_layer = Input(shape=img_shape)
    y = conv2d(input_layer, filters=filters, kernel_size=4,
               stride=1, padding='same', sn=spectr_norm)
    y = InstanceNormalization(axis=-1, center=True, scale=True)(y)
    y = LeakyReLU(0.1)(y)

    y = conv2d(y, filters=filters*2, kernel_size=4,
               stride=1, padding='same', sn=spectr_norm)
    y = InstanceNormalization(axis=-1, center=True, scale=True)(y)
    y = LeakyReLU(0.1)(y)

    y = conv2d(y, filters=filters*4, kernel_size=4,
               stride=1, padding='same', sn=spectr_norm)
    y = InstanceNormalization(axis=-1, center=True, scale=True)(y)
    y = LeakyReLU(0.1)(y)

    y = conv2d(y, filters=filters*8, kernel_size=4,
               stride=1, padding='same', sn=spectr_norm)
    y = InstanceNormalization(axis=-1, center=True, scale=True)(y)
    y = LeakyReLU(0.1)(y)

    y = SpatialPyramidPooling2D(bins=[1, 2, 3], data_format='channels_last')(y)
    y = Dense(1024)(y)
    y = LeakyReLU(0.1)(y)
    output = Dense(1)(y)

    return Model(input_layer, output)


def generator(img_shape, spectr_norm=False, gen_out=None):
    input_layer = Input(shape=img_shape)
    # Part 1. Feature Extraction Network
    filter_outputs = [64, 54, 48, 43, 39, 35, 31, 28, 25, 22,
                      18, 16, 24, 8, 8, 32, 16]
    fe_layers = []
    for i in range(12):
        if i == 0:
            fe = conv2d(input_layer, filters=filter_outputs[i], kernel_size=3,
                        stride=2, padding='same', sn=spectr_norm)
            fe = LeakyReLU(0.1)(fe)
            fe = Dropout(0.2)(fe)
        else:
            fe = conv2d(fe, filters=filter_outputs[i], kernel_size=3,
                        stride=1, padding='same', sn=spectr_norm)
            fe = LeakyReLU(0.1)(fe)
            fe = Dropout(0.2)(fe)

        fe_layers.append(fe)

    fe_final_layer = Concatenate()(fe_layers)

    # Part 2.1 Reconstruction Network

    a1 = conv2d(fe_final_layer, filters=filter_outputs[12], kernel_size=1,
                stride=1, padding='same', sn=spectr_norm)
    a1 = LeakyReLU(0.1)(a1)
    a1 = Dropout(0.2)(a1)

    b1 = conv2d(fe_final_layer, filters=filter_outputs[13], kernel_size=1,
                stride=1, padding='same', sn=spectr_norm)
    b1 = LeakyReLU(0.1)(b1)
    b1 = Dropout(0.2)(b1)

    b2 = conv2d(b1, filters=filter_outputs[14], kernel_size=3,
                stride=1, padding='same', sn=spectr_norm)
    b2 = LeakyReLU(0.1)(b2)
    b2 = Dropout(0.2)(b2)

    reconstructed = Concatenate()([a1, b2])

    # Part 2.2 Upsampling
    c1 = conv2d(reconstructed, filters=filter_outputs[15], kernel_size=3,
                stride=1, padding='same', sn=spectr_norm)
    c1 = LeakyReLU(0.1)(c1)

    c2 = upsample(c1, filters=filter_outputs[16], kernel_size=4,
                  stride=2, padding='same')
    c2 = LeakyReLU(0.1)(c2)

    output = conv2d(c2, filters=1, kernel_size=3,
                    stride=1, padding='same', bias=False,
                    activation=gen_out)

    return Model(input_layer, output)


def sft_generator(img_shape, hu_min, hu_max, spectr_norm=False, gen_out=None):
    input_layer = Input(shape=img_shape)

    # Part 1. Feature Extraction Network
    filter_outputs = [64, 54, 48, 43, 39, 35, 31, 28, 25, 22,
                      18, 16, 24, 8, 8, 32, 16]
    fe_layers = []
    for i in range(12):
        if i == 0:
            # segmentation map
            indices = tf.histogram_fixed_width_bins(
                input_layer, [float(hu_min), float(hu_max)], 10)
            seg_map = tf.one_hot(indices, 10)
            seg_map = tf.squeeze(seg_map, axis=3)
            sm = condition()(seg_map)

            # feature map
            fe = conv2d(input_layer, filters=filter_outputs[i], kernel_size=3,
                        stride=2, padding='same', sn=spectr_norm)
            fe = LeakyReLU(0.1)(fe)
            fe = Dropout(0.2)(fe)
        else:
            fe = sft(units=[32, fe.shape[-1]])([fe, sm])
            fe = conv2d(fe, filters=filter_outputs[i], kernel_size=3,
                        stride=1, padding='same', sn=spectr_norm)
            fe = LeakyReLU(0.1)(fe)
            fe = Dropout(0.2)(fe)

        fe_layers.append(fe)

    fe_final_layer = Concatenate()(fe_layers)

    # Part 2.1 Reconstruction Network
    a1 = sft(units=[32, fe_final_layer.shape[-1]])([fe_final_layer, sm])
    a1 = conv2d(a1, filters=filter_outputs[12], kernel_size=1,
                stride=1, padding='same', sn=spectr_norm)
    a1 = LeakyReLU(0.1)(a1)
    a1 = Dropout(0.2)(a1)

    b1 = sft(units=[32, fe_final_layer.shape[-1]])([fe_final_layer, sm])
    b1 = conv2d(fe_final_layer, filters=filter_outputs[13], kernel_size=1,
                stride=1, padding='same', sn=spectr_norm)
    b1 = LeakyReLU(0.1)(b1)
    b1 = Dropout(0.2)(b1)

    b2 = sft(units=[32, b1.shape[-1]])([b1, sm])
    b2 = conv2d(b1, filters=filter_outputs[14], kernel_size=3,
                stride=1, padding='same', sn=spectr_norm)
    b2 = LeakyReLU(0.1)(b2)
    b2 = Dropout(0.2)(b2)

    reconstructed = Concatenate()([a1, b2])

    # Part 2.2 Upsampling
    c1 = conv2d(reconstructed, filters=filter_outputs[15], kernel_size=3,
                stride=1, padding='same', sn=spectr_norm)
    c1 = LeakyReLU(0.1)(c1)

    c2 = upsample(c1, filters=filter_outputs[16], kernel_size=4,
                  stride=2, padding='same')
    c2 = LeakyReLU(0.1)(c2)

    output = conv2d(c2, filters=1, kernel_size=3,
                    stride=1, padding='same', bias=False,
                    activation=gen_out)

    return Model(input_layer, output)


def sa_generator(img_shape, spectr_norm=False, gen_out=None):
    input_layer = Input(shape=img_shape)

    # Part 1. Feature Extraction Network
    filter_outputs = [64, 54, 48, 43, 39, 35, 31, 28, 25, 22,
                      18, 16, 24, 8, 8, 32, 16]
    fe_layers = []
    for i in range(12):

        if i == 0:
            # feature map
            fe = conv2d(input_layer, filters=filter_outputs[i], kernel_size=3,
                        stride=2, padding='same', sn=spectr_norm)
            fe = LeakyReLU(0.1)(fe)
            fe = Dropout(0.2)(fe)
        else:
            fe = sa(filters=fe.shape[-1])(fe)
            fe = conv2d(fe, filters=filter_outputs[i], kernel_size=3,
                        stride=1, padding='same', sn=spectr_norm)
            fe = LeakyReLU(0.1)(fe)
            fe = Dropout(0.2)(fe)

        fe_layers.append(fe)

    fe_final_layer = Concatenate()(fe_layers)

    # Part 2.1 Reconstruction Network
    a1 = conv2d(fe_final_layer, filters=filter_outputs[12], kernel_size=1,
                stride=1, padding='same', sn=spectr_norm)
    a1 = LeakyReLU(0.1)(a1)
    a1 = Dropout(0.2)(a1)

    b1 = conv2d(fe_final_layer, filters=filter_outputs[13], kernel_size=1,
                stride=1, padding='same', sn=spectr_norm)
    b1 = LeakyReLU(0.1)(b1)
    b1 = Dropout(0.2)(b1)

    b2 = conv2d(b1, filters=filter_outputs[14], kernel_size=3,
                stride=1, padding='same', sn=spectr_norm)
    b2 = LeakyReLU(0.1)(b2)
    b2 = Dropout(0.2)(b2)

    reconstructed = Concatenate()([a1, b2])

    # Part 2.2 Upsampling
    c1 = conv2d(reconstructed, filters=filter_outputs[15], kernel_size=3,
                stride=1, padding='same', sn=spectr_norm)
    c1 = LeakyReLU(0.1)(c1)

    c2 = upsample(c1, filters=filter_outputs[16], kernel_size=4,
                  stride=2, padding='same')
    c2 = LeakyReLU(0.1)(c2)

    output = conv2d(c2, filters=1, kernel_size=3,
                    stride=1, padding='same', bias=False,
                    activation=gen_out)

    return Model(input_layer, output)
