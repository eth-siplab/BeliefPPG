import tensorflow as tf
import tensorflow_probability as tfp


def get_rescale_fn(dim, min_hz, max_hz):
    """
    Composes the augmentation function for shifts along the time axis.
    That is, generates a tf.function which
    1) scales the label y by a stretch_factor
    2) shifts the spectral features accordingly
    3) resamples the time-domain features to stretch_factor*old_frequency
    :param dim: number of input frequency bins
    :param min_hz: minimal frequency in spectrogram
    :param max_hz: maximal frequency in spectrogram
    :return:
    """
    min_bpm = min_hz * 60
    max_bpm = max_hz * 60

    @tf.function
    def stretch_time(time, stretch_factor):
        """
        stretches time-domain features by stretch_factor by resampling them linearly
        :param time: features
        :param stretch_factor: how much to stretch. New frequency is stretch_factor*old_frequency
        :return: stretched feat
        """
        time = tf.reshape(time, (-1,))
        length = tf.cast(len(time), tf.float32)
        new_length = length / stretch_factor
        new_x = tfp.math.interp_regular_1d_grid(
            tf.range(0, length) - length / 2,
            x_ref_min=-new_length / 2,
            x_ref_max=new_length / 2,
            y_ref=time,
        )
        return tf.expand_dims(new_x, -1)

    @tf.function
    def stretch_spec(spec, stretch_factor, y):
        """
        rolls the spectrogram along the y axis according to a relative factor
        :param spec:
        :param stretch_factor: factor to "stretch" by. New spectrogram will have frequencies f' = f * stretch_factor.
        :param y: ground truth (helps scaling the rolling)
        :return:
        """
        diff = y * stretch_factor - y
        roll = tf.cast(diff / (max_bpm - min_bpm) * dim, tf.int32)
        return tf.roll(spec, roll, axis=1)

    @tf.function
    def joint_random_rescale(x, y):
        """
        Stretches both time and freq features randomly. Time-domain features are resampled whereas
        the spectrogram is rolled along the frequency axis. The label is adjusted.
        :param x: (spec, time) feature representations
        :param y: continuous HR label
        :return: stretched input and label
        """
        stretch_factor = tf.random.uniform(
            (),
            tf.math.maximum(min_bpm / y, 0.75),
            tf.math.minimum(max_bpm / y, 1.25),
            dtype=tf.float32,
        )
        spec, time = stretch_spec(x[0], stretch_factor, y), stretch_time(
            x[1], stretch_factor
        )
        return (spec, time), y * stretch_factor

    return joint_random_rescale


@tf.function
def add_gaussian_noise(x, strength=1):
    """
    Corrupts the input with (strong) Gaussian noise
    :param x: tf.tensor
    :return: tf.tensor of same shape as x
    """
    noise = tf.random.normal(
        tf.shape(x),
        mean=0.0,
        stddev=0.25,
        dtype=tf.dtypes.float32,
    )
    return x + strength * noise


@tf.function
def corrupt_one_representation(x, y):
    """
    Corrupts one of (spec_input, time_input) with [strong] gaussian noise
    :param x: tf.tensor of shape ((n_frames, n_bins, 2), (n_steps, 1)) representing (spec_feat, time_feat)
    :param y: tf.tensor of shape (1,)
    :return: tuple of same shape as (x,y)
    """
    s = tf.round(tf.random.uniform((), 0, 1, dtype=tf.float32))
    freq, time = add_gaussian_noise(x[0], strength=s), add_gaussian_noise(
        x[1], strength=1 - s
    )
    return (freq, time), y


def add_augmentations(ds, args):
    """
    Adds train-time augmentations to a tf.data.Dataset yielding features and labels
    :param ds: tf.data.Dataset yielding tuples of shape (((n_frames, n_bins, 2), (n_steps, 1)), 1)
                representing (features, labels) where features=(spec_feat, time_feat) and the label is the BPM HR value.
    :param args: Namespace object containing config
    :return:
    """
    # very performance-sensitive
    rescale_fn = get_rescale_fn(args.n_bins, args.min_hz, args.max_hz)
    ds = ds.map(rescale_fn, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE
    )
    ds = ds.map(
        corrupt_one_representation, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    return ds
