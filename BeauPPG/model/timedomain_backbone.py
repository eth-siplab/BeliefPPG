from keras.layers import (LSTM, Activation, BatchNormalization, Conv1D, Dense,
                          Dropout, MaxPooling1D)


def get_timedomain_backbone(inp, output_shape):
    """
    generates the time backbone, whose architecture corresponds to CorNetLight
    :param model_in: time-domain input
    :param output_shape: what last dimension the bottleneck has
    :return: two branches, one for features and one for weight
    """
    x = inp
    # Block 1
    x = Conv1D(16, kernel_size=10, strides=1, dilation_rate=2, padding="causal")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Activation("leaky_relu")(x)
    x = MaxPooling1D(4, strides=4)(x)
    # Block 2
    x = Conv1D(16, 10, 1, dilation_rate=2, padding="causal")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Activation("leaky_relu")(x)
    x = MaxPooling1D(4, strides=4)(x)

    x = LSTM(64, activation="tanh", dropout=0.1, return_sequences=True)(
        x, training=True
    )
    x = LSTM(64, activation="tanh", dropout=0.1, return_sequences=False)(
        x, training=True
    )

    feat_branch = Dense(output_shape, activation="leaky_relu")(x)
    value_branch = Dense(output_shape, activation="leaky_relu")(x)
    return feat_branch, value_branch
