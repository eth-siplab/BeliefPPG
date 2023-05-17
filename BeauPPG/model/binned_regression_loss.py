import tensorflow as tf
import tensorflow_probability as tfp


class BinnedRegressionLoss(tf.keras.losses.Loss):
    """
    Implements the discretized regression loss function.
    """

    def __init__(
        self,
        dim: int,
        min_hz: float,
        max_hz: float,
        sigma_y: float,
        name="binned_log_likelihood",
        **kwargs
    ):
        """
        Initializes the layer from config
        :param dim: number of output bins of the model
        :param min_hz: lowest predictable HR value in Hz
        :param max_hz: highest predictable HR value in Hz
        :param sigma_y: standard deviation of gaussian we model ground truth HR [distribution] with
        :param name: name of the node
        :param kwargs:
        """
        super().__init__(name=name, **kwargs)
        self.sigma_y = sigma_y
        self.min_bpm = min_hz * 60
        self.max_bpm = max_hz * 60
        self.dim = dim
        self.bin_edges = tf.range(
            self.min_bpm, self.max_bpm, (self.max_bpm - self.min_bpm) / self.dim
        )

    @tf.function
    def y_to_bins(self, y):
        """
        creates binned representation of continuous label
        -> Gaussian-Bump representation around true HR
        :param y: label in BPM of shape (n_steps, )
        :return: target bin probabilities of shape (n_steps, self.dim)
        """
        distr = tfp.distributions.Normal(y, self.sigma_y)
        prob = distr.prob(self.bin_edges)
        return prob / tf.reduce_sum(prob)

    def call(self, y_true, y_pred):
        """
        performs a (symbolic) forward pass:
         1) Converts y to binned probability density representation (a normal distr. around the target)
         2) Calculates the cross-entropy between predicted and target distribution
        :param y_true: label in BPM of shape (n_steps,)
        :param y_pred: predictions for each bin o shape (n_steps, self.dim)
        :return:
        """
        binned_true = self.y_to_bins(y_true)
        losses = tf.keras.metrics.categorical_crossentropy(binned_true, y_pred, axis=-1)
        return tf.reduce_sum(losses)

    def get_config(self):
        """
        Helper function necessary to save model
        :return: dict config
        """
        config = {
            "sigma_y": self.sigma_y,
            "min_bpm": self.min_bpm,
            "max_bpm": self.max_bpm,
            "dim": self.dim,
        }
        base_config = super().get_config()
        return {**base_config, **config}
