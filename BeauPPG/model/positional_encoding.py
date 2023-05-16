import tensorflow as tf
import numpy as np

class PositionalEmbedding(tf.keras.layers.Layer):
  # https://www.tensorflow.org/text/tutorials/transformer
  # the original function computes an embedding from a (one-hot) vocabulary too

  def __init__(self, seqlen, d_model, num_dims=3):
      super().__init__()
      self.seqlen = seqlen
      self.d_model = d_model
      self.num_dims = num_dims
      self.pos_encoding = self._positional_encoding(length=seqlen, depth=d_model)

  def _positional_encoding(self, length, depth):
      depth = depth / 2

      positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
      depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

      angle_rates = 1 / (10000 ** depths)  # (1, depth)
      angle_rads = positions * angle_rates  # (pos, depth)

      pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

      return tf.cast(pos_encoding, dtype=tf.float32)

  def call(self, x):
      # This factor sets the relative scale of the embedding and positonal_encoding.
      x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
      if self.num_dims == 3:
          x = x + self.pos_encoding[tf.newaxis, tf.newaxis, :, :]
      else:
          x = x + self.pos_encoding[tf.newaxis, :, :]
      return x

  def get_config(self):
      """
        Helper function necessary to save model
        :return: dict config
      """
      config = super().get_config()
      config.update({
          "seqlen": self.seqlen,
          "d_model": self.d_model,
          "num_dims": self.num_dims
      })
      return config