from argparse import Namespace

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (
    Add,
    Activation,
    Attention,
    Concatenate,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling1D,
    Multiply,
    Permute,
    Reshape,
    UpSampling1D,
    )

from beliefppg.model.config import InputConfig
from beliefppg.model.positional_encoding import PositionalEncoding
from beliefppg.model.timedomain_backbone import get_timedomain_backbone


class AveragePooling1D(Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.dim)

    def get_config(self):
        config = super().get_config()
        config["dim"] = self.dim
        return config


class FlexibleAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention = Attention()

    def build(self, input_shape):
        # Check if the input is a list of tensors
        if isinstance(input_shape, list):
            # Assume each item in the list has the same shape
            self.use_reshaping = len(input_shape[0]) > 3
            if self.use_reshaping:
                # Compute intermediate and original shapes for the first tensor in the list
                self.intermediate_shape = (-1, input_shape[0][-2], input_shape[0][-1])
                self.original_shape = [-1,] + [i for i in input_shape[0][1:-2]] + [input_shape[0][-2], input_shape[0][-1]]
        else:
            self.use_reshaping = len(input_shape) > 3
            if self.use_reshaping:
                self.intermediate_shape = (-1, input_shape[-2], input_shape[-1])
                self.original_shape = [-1,] + [i for i in input_shape[1:-2]] + [input_shape[-2], input_shape[-1]]
        super().build(input_shape)

    def call(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs, inputs]  # For self-attention if a single tensor is provided

        if self.use_reshaping:
            reshaped_inputs = [tf.reshape(input_tensor, self.intermediate_shape) for input_tensor in inputs]
            attention_output = self.attention(reshaped_inputs)
            output = tf.reshape(attention_output, self.original_shape)
        else:
            output = self.attention(inputs)

        return output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape


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

    f = Activation("relu")(Add()([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv1D(1, 1, strides=1, data_format="channels_last")(f)

    rate = Activation("sigmoid")(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)
    att_x = Multiply()([x, rate])

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
        in_channel = down_layer.shape[1]
    else:
        in_channel = down_layer.shape[-1]

    up = UpSampling1D(size=down_fac)(down_layer)

    layer = attention_block_1d(x=layer, g=up, inter_channel=in_channel // 4)

    if data_format == "channels_first":
        combined = Concatenate(axis=1)([up, layer])
    else:
        combined = Concatenate(axis=-1)([up, layer])

    return combined


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
    freq_attn = FlexibleAttention()([Dense(channels)(x1_freq), Dense(channels)(x1_freq)])

    # self-attention over frequency space
    x1_time = PositionalEncoding(seqlen=n_frames, d_model=channels)(
        Permute((2, 1, 3))(x1)
    )
    time_attn = FlexibleAttention()([Dense(channels)(x1_time), Dense(channels)(x1_time)])
    time_attn = Permute((2, 1, 3))(time_attn)

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
    x = AveragePooling1D(1)(x)
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
        shape = weight_branch.shape
        reshape_layer = Reshape([1, shape[-1]])

        weight_branch = Concatenate(axis=-2)([reshape_layer(weight_branch), x])
        feat_branch = Concatenate(axis=-2)([reshape_layer(feat_branch), x])

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


def build_belief_ppg(n_frames: int, n_bins: int, freq: int, use_time_backbone=True) -> tf.keras.models.Model:
    """
    Wrapper function constructing the BeliefPPG architecture

    :param n_frames: The number of frames to consider in the spectrogram input.
    :param n_bins: The number of frequency bins in each frame of the spectrogram.
    :param freq: The frequency resolution of the time input.
    :return: tf.kers.models.Model the uncompiled functional model
    """
    spec_input = tf.keras.layers.Input(shape=(n_frames, n_bins, 2))
    time_input = tf.keras.layers.Input(
        shape=(freq * (n_frames - 1) * InputConfig.STRIDE + freq * InputConfig.WINSIZE, 1)
    )

    logits = hybrid_unet(spec_input, time_input, use_time_backbone=use_time_backbone)

    out = Activation("softmax")(logits)
    model = tf.keras.models.Model(inputs=[spec_input, time_input], outputs=out)
    return model
