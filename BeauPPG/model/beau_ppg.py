from argparse import Namespace

import keras.backend as K
import tensorflow as tf
from keras.layers import (Activation, Attention, Conv1D, Conv2D, Dense,
                          Dropout, Flatten, Lambda, MaxPooling1D, UpSampling1D)

from BeauPPG.model.positional_encoding import PositionalEncoding
from BeauPPG.model.timedomain_backbone import get_timedomain_backbone


def attention_block_1d(x, g, inter_channel):
    """
    helper function for attention block used in U-Net (upsampling path)
    Based on https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-
    :param x: branch 1 (from bottleneck)
    :param g: branch 2 (from down)
    :param inter_channel: nb channels to use internally
    :return: joined branches, same shape
    """
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv1D(inter_channel, 1, strides=1, data_format="channels_last")(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv1D(inter_channel, 1, strides=1, data_format="channels_last")(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation("relu")(tf.math.add(theta_x, phi_g))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv1D(1, 1, strides=1, data_format="channels_last")(f)

    rate = Activation("sigmoid")(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = tf.math.multiply(x, rate)

    return att_x


def attention_up_and_concate(
    down_layer, layer, data_format="channels_first", down_fac=2
):
    """
    helper function to define an upsampling block
    Based on https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-
    :param down_layer: skip connections
    :param layer: upsampling branch
    :param data_format: where the channels are
    :param down_fac: how much to down/upsample each depth level
    :return: output of depth level
    """
    if data_format == "channels_first":
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[-1]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling1D(size=down_fac)(down_layer)

    layer = attention_block_1d(x=layer, g=up, inter_channel=in_channel // 4)

    if data_format == "channels_first":
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))

    concate = my_concat([up, layer])
    return concate


def double_attn(inp, n_frames, n_bins, channels):
    """
    Constructs a self-attention block that applies one-dimensional attention across frames and bins in parallel.
    :param inp: tf.tensor of shape (batch, n_frames, n_bins, _)
    :param n_frames: first dimension
    :param n_bins: second dimension
    :param channels: last dimension
    :return: tuple containing two tf.tensor's (time_result, channel_result)
                each of shape (batch, n_frames, n_bins, channels)
    """
    x1 = Conv2D(channels, (3, 3), padding="same", activation="leaky_relu")(inp)
    x1 = Dropout(0.1)(x1)
    x1 = Conv2D(channels, (3, 3), padding="same", activation="leaky_relu")(x1)
    x1 = Dropout(0.1)(x1)

    # self-attention over time
    x1_freq = PositionalEncoding(seqlen=n_bins, d_model=channels)(x1)
    freq_attn = Attention()([Dense(channels)(x1_freq), Dense(channels)(x1_freq)])

    # self-attention over frequency space
    x1_time = PositionalEncoding(seqlen=n_frames, d_model=channels)(
        tf.transpose(x1, [0, 2, 1, 3])
    )
    time_attn = Attention()([Dense(channels)(x1_time), Dense(channels)(x1_time)])
    time_attn = tf.transpose(time_attn, [0, 2, 1, 3])

    return time_attn, freq_attn


def hybrid_unet(
    spec_input,
    time_input,
    depth=3,
    attn_channels=32,
    init_channels=12,
    down_fac=4,
    use_time_backbone=True,
):
    """
    Builds the architecture at the core of the model, involving
    1) an attention block followed by the reduction of the time dimension
    2) an U-Net performing frequency-to-frequency estimation
    3) a time-domain backbone consisting of a CNN-LSTM that is attached to the U-Net bottleneck.
    Note that the number of frequency bins needs to be compatible with the downsampling factor, i.e. divisible by 4
    :param spec_input: tf.tensor of shape (batch_size, n_frames, n_bins, 2)
    :param time_input: tf.tensor of shape (batch_size, n_timesteps , 1)
    :param depth: number of contraction steps in the U-Net
    :param attn_channels: inner dimension for initial attention block
    :param init_channels: channels to start from in first layer of U-Net. Are doubled in each contraction step.
    :param down_fac: downsampling factor of the U-Net
    :param use_time_backbone: whether to attach the time-domain backbone to the NN or not
    :return: tf.tensor of shape (batch_size, n_bins)
    """
    time_attn, freq_attn = double_attn(
        spec_input, spec_input.shape[1], spec_input.shape[2], attn_channels
    )
    # reduce mean over 7 time steps
    x = Dense(attn_channels)(spec_input) + time_attn + freq_attn
    x = tf.reduce_mean(x, axis=1)
    # resulting dimension (batch, freq_bins, channels)

    # U-net code until bottleneck
    skips = []
    channels = init_channels  # channels to start with

    # down
    for i in range(depth):
        x = Conv1D(
            channels, 3, activation="relu", padding="same", data_format="channels_last"
        )(x)
        x = Dropout(0.2)(x)
        x = Conv1D(
            channels, 3, activation="relu", padding="same", data_format="channels_last"
        )(x)
        skips.append(x)
        x = MaxPooling1D(down_fac, strides=down_fac, data_format="channels_last")(x)
        channels = channels * 2

    if use_time_backbone:
        # get backbone branches
        weight_branch, feat_branch = get_timedomain_backbone(time_input, x.shape[-1])

        # concatenate them with bottleneck
        weight_branch = tf.concat([tf.expand_dims(weight_branch, -2), x], axis=-2)
        feat_branch = tf.concat([tf.expand_dims(feat_branch, -2), x], axis=-2)

        # convolve weight back to shape (1,n_channels), then normalize
        weight_branch = Conv1D(x.shape[-1], 2, activation="tanh")(weight_branch)
        weight_branch = Dropout(0.2)(weight_branch)

        # also convolve feat
        feat_branch = Conv1D(x.shape[-1], 2, activation="relu")(feat_branch)
        feat_branch = Dropout(0.2)(feat_branch)

        # combine
        x = x + weight_branch * feat_branch

    # up
    for i in reversed(range(depth)):
        channels = channels // 2
        x = attention_up_and_concate(
            x, skips[i], data_format="channels_last", down_fac=down_fac
        )
        x = Conv1D(
            channels, 3, activation="relu", padding="same", data_format="channels_last"
        )(x)
        x = Dropout(0.2)(x)
        x = Conv1D(
            channels, 3, activation="relu", padding="same", data_format="channels_last"
        )(x)

    conv6 = Conv1D(1, 1, padding="same", data_format="channels_last")(x)

    # out
    x = Flatten()(conv6)
    return x


def build_beau_ppg(args: Namespace):
    """
    Wrapper function constructing the BeauPPG architecture
    :param args: Namespace object containing config
    :return: tf.kers.models.Model the uncompiled functional model
    """
    spec_input = tf.keras.layers.Input(shape=(args.n_frames, args.n_bins, 2))
    time_input = tf.keras.layers.Input(
        shape=(args.freq * (args.n_frames - 1) * 2 + args.freq * 8, 1)
    )

    logits = hybrid_unet(spec_input, time_input)

    out = Activation("softmax")(logits)
    model = tf.keras.models.Model(inputs=[spec_input, time_input], outputs=out)
    return model
