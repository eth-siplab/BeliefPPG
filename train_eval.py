import logging
import os
from itertools import groupby

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from keras import Model
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

import beliefppg
from beliefppg.datasets.pipeline_generator import get_sessions, join_sessions
from beliefppg.model.belief_ppg import build_belief_ppg
from beliefppg.model.binned_regression_loss import BinnedRegressionLoss
from beliefppg.model.prior_layer import PriorLayer
from beliefppg.util.args import parse_args
from beliefppg.util.augmentations import add_augmentations
from beliefppg.util.callbacks import get_callbacks


def generate_loso_split(sequences):
    """
    Generates a leave-one-out split of the input array based on the
    original DaLiA paper [1], subdividing the sequences into train (n-3), validation (2) and test (1) sets
    such that every session is the test session exactly once.
    Characteristic to the split is that it ensures that every sequence is used as
    validation set (roughly) equally often by first creating a coarse split and then rotating the test set.
    This reduces variance on the very small IEEE datasets, where the split has an impact on performance.
    With this in mind, any results on IEEE should be viewed critically.

    [1]: Reiss, A.; Indlekofer, I.; Schmidt, P.; Van Laerhoven, K. Deep PPG:
        Large-Scale Heart Rate Estimation with Convolutional Neural Networks. Sensors 2019, 19, 3079.
        https://doi.org/10.3390/s19143079

    :param sequences: list[object]
    :return: list[tuple[list[object], list[object], list[object]]] i.e. a list of tuples of lists of objects.
            Each list item consists of three lists representing a split. A split is a 3-tuple of lists representing
            train, val & test set respectively.
    """
    # process per dataset
    sequences = np.array(sequences)
    folds = []
    ixs = KFold(n_splits=4, shuffle=True, random_state=7).split(sequences)
    for j, (train_ixes, rest_ixes) in enumerate(ixs):
        rest = sequences[rest_ixes]
        for i in range(len(rest)):
            folds.append(
                (list(sequences[train_ixes]) + list(rest[:-3]), rest[-3:-1], rest[-1:])
            )
            rest = np.roll(rest, 1)

    # assert consistency
    for f in folds:
        # we use them all
        assert set(sequences) == set(f[0]).union(set(f[1])).union(set(f[2]))
        # no duplicates
        assert sum(map(len, f)) == len(sequences)
        # they are pairwise disjoint
        assert set(f[0]).isdisjoint(set(f[1]))
        assert set(f[2]).isdisjoint(set(f[1]))
        assert set(f[0]).isdisjoint(set(f[2]))
    return folds


def generate_train_split(names):
    """
    Generates one fold for training a model with a session from each dataset for validation and testing,
    while using the remaining sessions for training.

    :param sequences: list[object]
    :return: list[tuple[list[object], list[object], list[object]]] i.e. a list of tuples of lists of objects.
            Each list item consists of three lists representing a split. A split is a 3-tuple of lists representing
            train, val & test set respectively.
    """

    def key_func(x):
        return x.split("_", 1)[0]

    sorted_names = sorted(names, key=key_func)
    grouped_names = {key: list(group) for key, group in groupby(sorted_names, key=key_func)}
    grouped_indexes = {key: [names.index(name) for name in group] for key, group in grouped_names.items()}

    train = []
    val = []
    test = []

    for key, indexes in grouped_indexes.items():
        # take random element from each group for validation and one for testing
        indexes_ = np.random.choice(indexes, 2, replace=False)
        val.append(indexes_[0])
        test.append(indexes_[1])
        train.extend([index for index in indexes if index not in indexes_])

    folds = [(train, val, test)]

    sequences = list(range(len(names)))
    # assert consistency
    for f in folds:
        # we use them all
        assert set(sequences) == set(f[0]).union(set(f[1])).union(set(f[2]))
        # no duplicates
        assert sum(map(len, f)) == len(sequences)
        # they are pairwise disjoint
        assert set(f[0]).isdisjoint(set(f[1]))
        assert set(f[2]).isdisjoint(set(f[1]))
        assert set(f[0]).isdisjoint(set(f[2]))
    return folds


def limit_memory_growth():
    """
    Helper function to limit tensorflow GPU memory usage. May allow for multiple runs on single GPU.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            logging.info(f"{len(gpus)} Physical GPUs {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.warning(e)


def set_seed(seed):
    """
    Sets the seed for numpy, keras, tensorflow, tensorflow_probability, etc. for full determinism.
    :param seed: .
    """
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    np.random.seed(seed)


def evaluate_and_log(y_pred, y_pred_viterbi, y_true, uncertainty, sess_name, out_path, use_wandb):
    """
    Evaluates test predictions and logs to out_path as well as wandb
    :param y_pred: np.ndarray of shape (n_pred,), result obtained from sum-product message passing
    :param y_pred_viterbi: np.ndarray of shape (n_pred,), result obtained from max-product message passing
    :param y_true: np.ndarray of shape (n_pred,)
    :param uncertainty: np.ndarray of shape (n_pred,)
    :param sess_name: string descriptor
    :param out_path: relative or absolute path to store results in
    :return: float test mae
    """
    test_mae = mean_absolute_error(y_true, y_pred)
    test_r2 = r2_score(y_true, y_pred)
    spearman_res = spearmanr(np.abs(y_true - y_pred), uncertainty)

    viterbi_mae = mean_absolute_error(y_true, y_pred_viterbi)

    # log results
    logging.info(r"Session %s MAE %f" % (sess_name, test_mae))
    if use_wandb:
        wandb.log({f"test-mae": test_mae})
        wandb.log({f"test-r2": test_r2})
        wandb.log({f"spearman-corr": spearman_res.correlation})
        wandb.log({f"prediction": y_pred})
        wandb.log({f"test-mae-offline": viterbi_mae})
        wandb.log({f"prediction-offline": y_pred_viterbi})
        aerr = np.abs(y_true - y_pred)
        for j in range(len(y_true)):
            wandb.log({f"truth": y_true[j]})
            wandb.log({f"pred": y_pred[j]})
            wandb.log({f"pred-offline": y_pred_viterbi[j]})
            wandb.log({f"certainty": -uncertainty[j]})
            wandb.log({f"abs-error": aerr[j]})

    # save to output dir
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    np.savetxt(os.path.join(out_path, "y_pred.csv"), y_pred, delimiter=",")
    np.savetxt(os.path.join(out_path, "y_pred_offline.csv"), y_pred_viterbi, delimiter=",")
    np.savetxt(os.path.join(out_path, "y_true.csv"), y_true, delimiter=",")
    np.savetxt(os.path.join(out_path, "uncertainty.csv"), uncertainty, delimiter=",")

    return test_mae


def train_eval(args):
    """
    Performs a training-validation scheme specified by the mode parameter on a specified dataset. That is, it creates folds,
    then builds a model for each fold and trains it on the training set until it converges w.r.t the validation set.
    Then it fits & adds a prior layer before predicting & evaluating on the test set.
    Monitors progress on W&B if configured. Saves predicions, stats, and model weights in the output dir.

    :param args: Namespace arguments from command line
    """
    limit_memory_growth()
    set_seed(args.seed)

    sessions, names = get_sessions(args)

    if args.mode == "loso":
        folds = generate_loso_split(np.arange(0, len(sessions)))
    elif args.mode == "train":
        folds = generate_train_split(names)
    else:
        raise NotImplementedError(f"Mode {args.mode} not implemented")

    maes = {}
    for i, (train_ixes, val_ixes, test_ixes) in enumerate(folds):
        train_split, val_split, test_split = (
            sessions[train_ixes],
            sessions[val_ixes],
            sessions[test_ixes],
        )

        test_split_name = "_".join([names[idx] for idx in test_ixes])
        out_path = os.path.join(args.out_dir, f"{args.dataset}-{test_split_name}")
        logging.info(f"Split {i+1}/{len(folds)}")
        logging.info(f"Test session: {args.dataset}-{test_split_name}")
        logging.info(f"Saving output under {out_path}")

        if args.use_wandb:
            wandb.init(
                project="BeliefPPG",
                name=f"{args.dataset}-{test_split_name}",
                config=vars(args),
                dir=args.out_dir,
            )

        # prepare tf.data pipeline
        train_ds = join_sessions(
            train_split, shuffle=True
        )
        val_ds = join_sessions(val_split, shuffle=False)
        if args.cache_pipeline:
            train_ds = train_ds.cache().shuffle(5000)
            val_ds = val_ds.cache()
        if args.augmentations:
            train_ds = add_augmentations(train_ds, args)

        # build model
        model = build_belief_ppg(args.n_frames, args.n_bins, args.freq)
        loss_fn = BinnedRegressionLoss(
            args.n_bins, args.min_hz, args.max_hz, args.sigma_y
        )
        optimizer = tf.optimizers.Adam(learning_rate=args.init_lr)
        model.compile(optimizer=optimizer, loss=loss_fn)

        # fit
        callbacks = get_callbacks(args.patience, args.use_wandb)
        history = model.fit(
            train_ds.batch(args.batch_size),
            validation_data=val_ds.batch(args.batch_size),
            epochs=10000,  # rely on early stopping, set this to imaginary value
            verbose=2,
            callbacks=callbacks,
        )

        # fit prior
        train_ys = [
            np.array(list(ds.map(lambda x, y: y))) for ds in train_split
        ]  # we could include val_split here, as long as we don't use test sesh
        prior_layer = PriorLayer(
            args.n_bins, args.min_hz, args.max_hz, is_online=True, return_probs=False, uncert=args.uncertainty
        )
        prior_layer.fit_layer(train_ys, distr=args.prior)

        # prepare model for inference
        out = prior_layer(model.output)
        inference_model = Model(model.input, out)

        # predict
        test_ds = join_sessions(test_split, shuffle=False)
        y_pred, uncertainty = inference_model.predict(test_ds.batch(args.batch_size))
        inference_model.get_layer('prior_layer').set_online(False)
        y_pred_viterbi, _ = inference_model.predict(test_ds.batch(args.batch_size))
        y_true = np.array(list(test_ds.map(lambda x, y: y)))

        # evaluate & log
        test_mae = evaluate_and_log(
            y_pred, y_pred_viterbi, y_true, uncertainty, names[test_ixes[0]], out_path, args.use_wandb
        )
        maes[test_split_name] = [test_mae]

        # save model
        try:
            model.save(os.path.join(out_path, "raw_model.keras"))
            inference_model.get_layer('prior_layer').set_online(True)
            inference_model.save(os.path.join(out_path, "inference_model.keras"))
        except ImportError:
            logging.warning(
                "Failed to save model. ImportError: check your h5py installation."
            )

        if args.use_wandb:
            wandb.finish()

    # compile summary
    result_df = pd.DataFrame(maes)
    result_df["mean"] = result_df.mean(axis=1)
    result_df["std"] = result_df.std(axis=1)

    result_df.to_csv(os.path.join(args.out_dir, f"results_{args.dataset}.csv"))
    logging.info(result_df.to_string())


if __name__ == "__main__":
    args = parse_args()
    train_eval(args)
