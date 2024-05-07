import numpy as np
import tensorflow as tf
from scipy.stats import laplace, norm


class TransitionPriorNotSetError(Exception):
    def __init__(self, message="Transition prior is not set. Fit the layer before calling it."):
        self.message = message
        super().__init__(self.message)


@tf.keras.utils.register_keras_serializable('PriorLayer')
class PriorLayer(tf.keras.layers.Layer):
    """
    Implements functionality for the belief propagation / decoding framework.
    The layer is intended to be added as last layer to a trained network with instantaneous HR
     bin probabilities as output. It should be fit separately on training data before doing so.
     When added, the model produces either contextualized probabilities or BPM HR estimates as output.
    """

    def __init__(self, dim, min_hz, max_hz, is_online, return_probs,
                 transition_prior=None, uncert="entropy", **kwargs):
        """
        Construct the Prior Layer translating instantaneous bin probabilities into contextualized HR predictions.
        :param dim: number of bins
        :param min_hz: minimal predictable frequency
        :param max_hz: maximal predictable frequency
        :param is_online: whether sum-product message passing (True) or viterbi decoding (False) should be applied
        :param return_probs: returns contextualized bin probabilities if set to True, HR estimates in BPM otherwise.
        :param transition_prior: transition prior matrix, if not set, it will have to be learned from data
        :param uncert: The uncertainty measure to use. One of ["entropy", "std"].
        :param kwargs: passed to parent class
        """
        kwargs.setdefault('name', 'prior_layer') 
        super(PriorLayer, self).__init__(**kwargs)
        self.state = tf.Variable(tf.convert_to_tensor(np.ones(dim) / dim, "float32"))
        self.trainable = False
        self.dim = dim
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.is_online = tf.Variable(is_online, trainable=False, dtype=tf.bool)
        self.return_probs = return_probs
        if uncert=="entropy":
            self.is_entropy_uncert = tf.Variable(True, trainable=False, dtype=tf.bool)
        elif uncert=="std":
            self.is_entropy_uncert = tf.Variable(False, trainable=False, dtype=tf.bool)
        else:
            raise NotImplementedError(f"Unknown uncertainty measure: {uncert}")
        self.bins = tf.constant([self.hr(i) for i in range(0, dim)], "float32")
        if transition_prior is not None:
            self.transition_prior = tf.constant(transition_prior, "float32")
        else:
            self.transition_prior = transition_prior

    def set_online(self, is_online):
        """
        Set the online mode of the layer.
        :param is_online: whether to use sum-product message passing (True) or Viterbi decoding (False)
        """
        self.is_online.assign(is_online)

    def set_uncertainty(self, uncert):
        """
        Set the uncertainty measure to use.
        :param uncert: The uncertainty measure to use. One of ["entropy", "std"].
        """
        if uncert == "entropy":
            self.is_entropy_uncert.assign(True)
        elif uncert == "std":
            self.is_entropy_uncert.assign(False)
        else:
            raise NotImplementedError(f"Unknown uncertainty measure: {uncert}")

    def hr(self, i):
        """
        Helper function to calculate heart rate based on bin index
        :param i: bin index
        :return: heart rate in bpm
        """
        return self.min_hz * 60.0 + (self.max_hz - self.min_hz) * 60.0 * i / self.dim

    def _fit_distr(self, diffs, distr):
        """
        Fits a Gaussian/Laplacian prior to heart rate changes.
        :param diffs: heart rate changes in BPM for consecutive 8s windows with 2s shift
        :param distr: uses Laplacian distribution when set to true. Default is Gaussian. For the Gaussian,
        differences are assumed to be log differences.
        :return: mean and stddev of fitted Gaussian/Laplacian
        """
        if distr == "laplace":
            mu, sigma = laplace.fit(diffs)
        else:
            mu, sigma = norm.fit(diffs)
        return mu, sigma

    def fit_layer(self, ys, distr="laplace", sparse=False):
        """
        Precomputes a prior matrix based on heart rate changes.
        :param ys: list of ground truth HR values with same strides as labels
        :param distr: whether to fit a Laplacian or Gaussian on the (log) diffs
        :param sparse: whether to round very low probabilities down to zero.
                    Results in theoretically higher efficiency.
        """
        if distr == "laplace":
            diffs = np.concatenate([y[1:] - y[:-1] for y in ys], axis=0)
        elif distr == "gauss":
            diffs = np.concatenate([np.log(y[1:]) - np.log(y[:-1]) for y in ys], axis=0)
        else:
            raise NotImplementedError(r"Unknown prior %s" % distr)

        mu, sigma = self._fit_distr(diffs, distr)
        trans_prob = np.zeros((self.dim, self.dim))

        for i in range(self.dim):
            for j in range(self.dim):
                if (
                    sparse and abs(i - j) > 10 * 60 / self.dim
                ):  # cut off large transitions
                    trans_prob[i][j] = 0.0
                else:
                    if distr == "laplace":
                        trans_prob[i][j] = laplace.cdf(
                            abs(i - j) + 1, mu, sigma
                        ) - laplace.cdf(abs(i - j) - 1, mu, sigma)
                    elif distr == "gauss":
                        log_diffs = [
                            np.log(self.hr(i1)) - np.log(self.hr(i2))
                            for i1 in (i - 0.5, i + 0.5)
                            for i2 in (j - 0.5, j + 0.5)
                        ]
                        max_logdiff = np.max(log_diffs)
                        min_logdiff = np.min(log_diffs)
                        trans_prob[i][j] = norm.cdf(max_logdiff, mu, sigma) - norm.cdf(
                            min_logdiff, mu, sigma
                        )

        # no need for normalization, probability leaks are handled during forward propagation
        self.transition_prior = tf.constant(trans_prob, "float32")

    @tf.function
    def _propagate_sumprod(self, ps):
        """
        Performs online belief propagation i.e. sum-product message passing
        New probabilities are calculated as T.p_{old} * p_{pred} and normalized, where * is the Hadamard product.
        :param ps: ps: tf.tensor of shape (n_samples, n_bins) containing probabilities
        :return: tf.tensor of same shape containing updated probabilities
        """
        if self.transition_prior is None:
            raise TransitionPriorNotSetError()

        i = tf.constant(0)
        output = tf.TensorArray(tf.float32, size=tf.shape(ps)[0])
        while i < tf.shape(ps)[0]:
            # propagate (blurred) last observations
            p_prior = tf.linalg.matvec(self.transition_prior, self.state)
            # add current observations
            p_new = p_prior * ps[i]
            self.state = p_new / tf.reduce_sum(p_new)
            output = output.write(i, self.state)
            i += 1

        return output.stack()

    @tf.function
    def _update_prob(self, prev_maxprod, curr):
        """
        Applies one timestep of the forward pass of Viterbi Decoding.
        :param prev_maxprod: tf.Tensor of shape (n_bins,) containing probabilities of most probable path per HR bin
        :param curr: tf.Tensor of shape (n_bins,) containing the raw model beliefs for the current timestep
        :return: tuple (curr_maxprod, ixes) containing new best path scores and backpointers respectively
        """
        trans_prod = prev_maxprod[tf.newaxis, :] * self.transition_prior  # Broadcasting prev_maxprod to compute all transitions
        values, indices = tf.math.top_k(trans_prod, k=1, sorted=False, index_type=tf.int32)
        curr_maxprod = tf.squeeze(values, axis=1)
        ixes = tf.squeeze(indices, axis=1)

        curr_maxprod *= curr  # Element-wise multiplication with current probabilities
        curr_maxprod /= tf.reduce_sum(curr_maxprod)  # Normalize the probabilities
        return curr_maxprod, ixes

    @tf.function
    def _decode_viterbi(self, raw_pred):
        """
        Performs Viterbi Decoding on the output probabilities using TensorFlow operations.
        :param raw_pred: tf.Tensor of shape (n_timesteps, n_bins) of raw HR probabilities
        :return: tf.Tensor of shape (n_timesteps, ) of predictions for each step
        """
        if self.transition_prior is None:
            raise TransitionPriorNotSetError()
        dim = tf.shape(self.transition_prior, out_type=tf.int32)[0]

        # Forward pass
        prev_maxprod = tf.fill([dim], 1.0 / tf.cast(dim, tf.float32))  # Initialize with uniform distribution
        paths = tf.TensorArray(tf.int32, size=tf.shape(raw_pred)[0], clear_after_read=False)

        for j in tf.range(tf.shape(raw_pred)[0], dtype=tf.int32):
            prev_maxprod, ixes = self._update_prob(prev_maxprod, raw_pred[j])
            paths = paths.write(j, ixes)

        # Backward pass (traceback)
        paths = paths.stack()  # Convert tensor array to tensor for backward indexing
        best_path = tf.TensorArray(tf.int32, size=tf.shape(raw_pred)[0], clear_after_read=False)
        curr_ix = tf.argmax(prev_maxprod, output_type=tf.int32)
        for i in tf.range(tf.shape(paths)[0]-1, -1, -1, dtype=tf.int32):
            best_path = best_path.write(i, curr_ix)
            curr_ix = paths[i, curr_ix]
        best_path = best_path.stack()

        hr_values = tf.gather(self.bins, best_path)

        return hr_values

    @tf.function
    def call(self, ps):
        """
        Calculates a stateful forward pass applying the transition prior (symbolically).
        Assumes batch consists of consecutive samples. Overrides parent function.
        :param ps: tf.tensor of shape (n_samples, n_bins) containing probabilities
        :return: probs : tf.tensor of same shape, only returned if return_probs=True
                 E_x : tf.tensor of shape (n_samples,) containing the expected HR, only if return_probs=False
                 uncert : tf.tensor of shape (n_samples,) containing est. uncertainty of the prediction
        """
        if self.is_online.value():
            probs = self._propagate_sumprod(ps)
            E_x = tf.reduce_sum(probs * self.bins[None, :], axis=1)

            if self.is_entropy_uncert.value():
                uncert = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1)
            else:
                E_x2 = tf.reduce_sum(probs * self.bins[None, :] ** 2, axis=1)
                uncert = tf.sqrt(E_x2 - E_x**2)

            if self.return_probs:
                return probs, uncert
            else:
                return E_x, uncert
        else:
            MLE = self._decode_viterbi(ps)
            return MLE, tf.zeros_like(MLE)

    def get_config(self):
        """
        Helper function necessary to save model
        :return: dict config
        """
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "min_hz": self.min_hz,
                "max_hz": self.max_hz,
                "is_online": self.is_online.numpy(),
                "return_probs": self.return_probs,
                "transition_prior": self.transition_prior.numpy().tolist(),
                "uncert": "entropy" if self.is_entropy_uncert.numpy() else "std",
            }
        )
        return config
