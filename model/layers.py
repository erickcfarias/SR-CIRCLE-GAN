import tensorflow as tf
import numpy as np
from tensorflow_addons.layers import InstanceNormalization, SpectralNormalization
from tensorflow.keras.layers import (
    Layer, Lambda, BatchNormalization, LeakyReLU,
    Conv2D, Conv2DTranspose)
from tensorflow.keras.initializers import RandomNormal


def conv2d(layer_input, filters, kernel_size, stride, padding='same',
           activation=None, bias=True, sn=False):
    m = (kernel_size**2) * filters
    weights_ini = RandomNormal(mean=0., stddev=np.sqrt(2/m))
    bias_ini = tf.constant_initializer(0.0)
    if sn:
        x = SpectralNormalization(
                Conv2D(
                    filters=filters, kernel_size=kernel_size,
                    strides=stride, padding=padding,
                    activation=activation,
                    bias_initializer=bias_ini,
                    kernel_initializer=weights_ini,
                    use_bias=bias
                )
        )(layer_input)
    else:
        x = Conv2D(
            filters=filters, kernel_size=kernel_size,
            strides=stride, padding=padding,
            activation=activation,
            bias_initializer=bias_ini,
            kernel_initializer=weights_ini,
            use_bias=bias
        )(layer_input)

    return x


def downsample(layer_input, filters, f_size=4):
    m = (f_size**2) * filters
    weights_ini = RandomNormal(mean=0., stddev=np.sqrt(2/m))
    bias_ini = tf.constant_initializer(0.0)
    d = Conv2D(filters, kernel_size=f_size,
               strides=2, padding='same',
               bias_initializer=bias_ini,
               kernel_initializer=weights_ini)(layer_input)
    d = InstanceNormalization(axis=-1, center=True, scale=True)(layer_input)
    d = LeakyReLU(0.1)(d)

    return d


def upsample(layer_input, filters, kernel_size=4, stride=1, padding='same',
             activation=None):
    m = (kernel_size**2) * filters
    weights_ini = RandomNormal(mean=0., stddev=np.sqrt(2/m))
    bias_ini = tf.constant_initializer(0.0)

    u = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                        strides=stride, padding=padding,
                        activation=activation, kernel_initializer=weights_ini,
                        bias_initializer=bias_ini)(layer_input)

    return u


def b_norm(layer_input, eps=1e-5):
    b = BatchNormalization(epsilon=eps, momentum=0.9, scale=True)
    return b


def resize(layer_input, new_size):
    r = Lambda(
        lambda image: tf.image.resize(
            image,
            (new_size[0],
                new_size[1]),
            method=tf.image.ResizeMethod.BICUBIC,
            preserve_aspect_ratio=True
        )
    )(layer_input)
    return r


def hw_flatten(x):
    shape = tf.TensorShape([x.shape[0], -1, x.shape[-1]])
    return tf.reshape(x, shape=shape)


# SFT Implementation
class sft(Layer):

    def __init__(self, units, **kwargs):
        super(sft, self).__init__(**kwargs)
        self.units_0 = units[0]
        self.unit_1 = units[1]

        self.conv2d_scale_0 = Conv2D(filters=units[0], kernel_size=1,
                                     strides=1, padding='same',
                                     activation=None,
                                     use_bias=True)
        self.leaky_scale = LeakyReLU(0.01)
        self.conv2d_scale_1 = Conv2D(filters=units[1], kernel_size=1,
                                     strides=1, padding='same',
                                     activation=None,
                                     use_bias=True)

        self.conv2d_shift_0 = Conv2D(filters=units[0], kernel_size=1,
                                     strides=1, padding='same',
                                     activation=None,
                                     use_bias=True)
        self.leaky_shift = LeakyReLU(0.01)
        self.conv2d_shift_1 = Conv2D(filters=units[1], kernel_size=1,
                                     strides=1, padding='same',
                                     activation=None,
                                     use_bias=True)

    def call(self, inputs):

        # inputs[0] = fea, inputs[1] = segmentation map
        scale = self.conv2d_scale_0(inputs[1])
        scale = self.leaky_scale(scale)
        scale = self.conv2d_scale_1(scale)

        shift = self.conv2d_shift_0(inputs[1])
        shift = self.leaky_shift(shift)
        shift = self.conv2d_scale_1(shift)

        return (inputs[0] * scale) + shift

    def get_config(self):
        config = super(sft, self).get_config()
        config.update({"units_0": self.units_0, "units_1": self.units_1})
        return config


class condition(Layer):

    def __init__(self, **kwargs):
        super(condition, self).__init__(**kwargs)
        self.conv2d_0 = Conv2D(filters=128, kernel_size=1,
                               strides=2, padding='same',
                               activation=None,
                               use_bias=True)
        self.leaky_0 = LeakyReLU(0.1)
        self.conv2d_1 = Conv2D(filters=128, kernel_size=1,
                               strides=1, padding='same',
                               activation=None,
                               use_bias=True)
        self.leaky_1 = LeakyReLU(0.1)
        self.conv2d_2 = Conv2D(filters=128, kernel_size=1,
                               strides=1, padding='same',
                               activation=None,
                               use_bias=True)
        self.leaky_2 = LeakyReLU(0.1)
        self.conv2d_3 = Conv2D(filters=32, kernel_size=1,
                               strides=1, padding='same',
                               activation=None,
                               use_bias=True)
        self.leaky_3 = LeakyReLU(0.1)

    def call(self, inputs):
        shared_sft_cond = self.conv2d_0(inputs)
        shared_sft_cond = self.leaky_0(shared_sft_cond)
        shared_sft_cond = self.conv2d_1(shared_sft_cond)
        shared_sft_cond = self.leaky_1(shared_sft_cond)
        shared_sft_cond = self.conv2d_2(shared_sft_cond)
        shared_sft_cond = self.leaky_2(shared_sft_cond)
        shared_sft_cond = self.conv2d_3(shared_sft_cond)
        shared_sft_cond = self.leaky_3(shared_sft_cond)

        return shared_sft_cond

    def get_config(self):
        config = super(condition, self).get_config()
        config.update({"binning_range": self.binning_range,
                       "nbins": self.nbins})
        return config


class sa(Layer):
    def __init__(self, filters, num_categories=1, **kwargs):
        super(sa, self).__init__(**kwargs)
        self.filters = filters
        self.conv_f = Conv2D(
            filters=self.filters // 8, kernel_size=1, strides=1, padding='same',
            activation=None, use_bias=True
        )
        self.conv_g = Conv2D(
            filters=self.filters // 8, kernel_size=1, strides=1, padding='same',
            activation=None, use_bias=True
        )
        self.conv_h = Conv2D(
            filters=self.filters, kernel_size=1, strides=1, padding='same',
            activation=None, use_bias=True
        )
        self.gamma = tf.Variable(lambda: tf.ones(
            shape=[num_categories]), name="gamma")

    def call(self, x):

        f = self.conv_f(x)
        g = self.conv_g(x)
        h = self.conv_h(x)

        attention_map = \
            tf.nn.softmax(
                tf.matmul(g, f, transpose_b=True)
            )

        o = tf.matmul(attention_map, h)

        output = self.gamma * o

        return output

    def get_config(self):
        config = super(sft, self).get_config()
        config.update(
            {"filters": self.filters, "gamma": self.gamma}
        )
        return config
