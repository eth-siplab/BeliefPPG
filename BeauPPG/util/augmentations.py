import tensorflow as tf
import tensorflow_probability as tfp


def get_rescale_fn(dim, min_hz, max_hz):
    min_bpm = min_hz * 60
    max_bpm = max_hz * 60

    @tf.function
    def stretch(time, stretch_factor):
        """
        stretches time-domain features by stretch_factor
        :param time: features
        :param stretch_factor: how much to stretch
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
        diff = y * stretch_factor - y
        roll = tf.cast(diff / (max_bpm - min_bpm) * dim, tf.int32)
        return tf.roll(spec, roll, axis=1)

    @tf.function
    def joint_random_rescale(x, y):
        """
        stretches both time and freq features randomly
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
        spec, time = stretch_spec(x[0], stretch_factor, y), stretch(
            x[1], stretch_factor
        )
        return (spec, time), y * stretch_factor

    return joint_random_rescale


@tf.function
def add_gaussian_noise(x, strength=1):
    noise = tf.random.normal(
        tf.shape(x),
        mean=0.0,
        stddev=0.25,
        dtype=tf.dtypes.float32,
    )
    return x + strength * noise


@tf.function
def corrupt_one_representation(x, y):
    s = tf.round(tf.random.uniform((), 0, 1, dtype=tf.float32))
    freq, time = add_gaussian_noise(x[0], strength=s), add_gaussian_noise(
        x[1], strength=1 - s
    )
    return (freq, time), y


def add_augmentations(ds, args, rescale=True, add_noise=True):
    # very performance-sensitive
    if rescale:
        rescale_fn = get_rescale_fn(args.n_bins, args.min_hz, args.max_hz)
        ds = ds.map(rescale_fn, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE
        )
    if add_noise:
        ds = ds.map(
            corrupt_one_representation, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
    return ds
