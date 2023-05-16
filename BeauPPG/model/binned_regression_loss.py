import tensorflow as tf
import tensorflow_probability as tfp

class BinnedRegressionLoss(tf.keras.losses.Loss):
    def __init__(self, dim, min_hz, max_hz, sigma_y, name="binned_log_likelihood", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sigma_y = sigma_y
        self.min_bpm = min_hz * 60
        self.max_bpm = max_hz * 60
        self.dim = dim

    @tf.function
    def y_to_bins(self, y):
        """
        creates binned representation of continuous label
        -> Gaussian-Bump representation around true HR
        :param hr: label
        :return: ihr in binned representation
        """
        distr = tfp.distributions.Normal(y, self.sigma_y)
        prob = distr.prob(tf.range(self.min_bpm, self.max_bpm, (self.max_bpm - self.min_bpm) / self.dim))
        return prob / tf.reduce_sum(prob)

    def call(self, y_true, y_pred):
        binned_true = self.y_to_bins(y_true)
        losses = tf.keras.metrics.categorical_crossentropy(binned_true, y_pred, axis=-1)
        return tf.reduce_sum(losses)

    def get_config(self):
        config = {
            'sigma_y': self.sigma_y,
            'min_bpm': self.min_bpm,
            'max_bpm': self.max_bpm,
            'dim': self.dim
        }
        base_config = super().get_config()
        return {**base_config, **config}