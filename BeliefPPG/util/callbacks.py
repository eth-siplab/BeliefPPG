import tensorflow as tf
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


def get_callbacks(patience, use_wandb):
    """
    Generates a list of callbacks for tf.keras training loop
    :param patience: int, number of iterations without improvement for early stopping
    :param use_wandb: bool, whether to use Weights&Biases for monitoring. Make sure to be logged in if True
    :return: List[tf.keras.callbacks.Callback] of callbacks
    """
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5, min_lr=1e-6, monitor="loss", patience=3
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=patience, restore_best_weights=True, monitor="val_loss"
    )
    cbs = [reduce_lr, early_stopping]
    if use_wandb:
        cbs += [WandbMetricsLogger()]
    return cbs
