import os

import pandas as pd
from keras import Model
from sklearn.model_selection import train_test_split, KFold

import BeauPPG
from BeauPPG.datasets.pipeline_generator import get_sessions, join_sessions
from BeauPPG.model.binned_regression_loss import BinnedRegressionLoss
from BeauPPG.model.prior_layer import PriorLayer
from BeauPPG.util.augmentations import add_augmentations
from BeauPPG.model.beau_ppg import build_beau_ppg

from BeauPPG.util.callbacks import get_callbacks
from BeauPPG.util.args import parse_args
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
import tensorflow as tf
import wandb

def generate_split(sequences):
    # process per dataset
    sequences = np.array(sequences)
    folds = []
    ixs = KFold(n_splits=4, shuffle=True, random_state=7).split(sequences)
    for j, (train_ixes, rest_ixes) in enumerate(ixs):
        rest = sequences[rest_ixes]
        for i in range(len(rest)):
            folds.append((list(sequences[train_ixes]) + list(rest[:-3]), rest[-3:-1], rest[-1:]))
            rest = np.roll(rest, 1)
            print(list(map(len, folds[-1])))

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



def train_eval(args):

    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    np.random.seed(args.seed)

    sessions, names = get_sessions(args)

    maes = {}

    folds = generate_split(np.arange(0, len(sessions)))

    for i, (train_ixes, val_ixes, test_ixes) in enumerate(folds):
        # if i < 6:
        #     continue
        train_split, val_split, test_sesh = sessions[train_ixes], sessions[val_ixes], sessions[test_ixes[0]]

        out_path = os.path.join(args.out_dir, f"{args.dataset}-{names[test_ixes[0]]}")
        logging.info(f"Split {i+1}/{len(folds)}")
        logging.info(f"Test session: {args.dataset}-{names[test_ixes[0]]}")
        logging.info(f"Saving output under {out_path}")

        if args.use_wandb:
            wandb.init(
                project="BeauPPG",
                name=f"{args.dataset}-{names[test_ixes[0]]}",
                config=vars(args),
                dir=args.out_dir,
            )

        # prepare tf.data pipeline
        train_ds = BeauPPG.datasets.pipeline_generator.join_sessions(train_split, shuffle=True)
        val_ds = join_sessions(val_split, shuffle=False)
        if args.cache_pipeline:
            train_ds = train_ds.cache().shuffle(5000)
            val_ds = val_ds.cache()
        if args.augmentations:
           train_ds = add_augmentations(train_ds, args)

        # build model
        model = build_beau_ppg(args)
        loss_fn = BinnedRegressionLoss(args.n_bins, args.min_hz, args.max_hz, args.sigma_y)
        optimizer = tf.optimizers.Adam(learning_rate=args.init_lr)
        model.compile(optimizer=optimizer, loss=loss_fn)

        # fit
        callbacks = get_callbacks(args.patience, args.use_wandb)
        history = model.fit(
            train_ds.batch(args.batch_size),
            validation_data = val_ds.batch(args.batch_size),
            epochs=10000, # rely on early stopping, set this to imaginary value
            verbose=2,
            callbacks=callbacks
        )

        # fit prior
        train_ys = [np.array(list(ds.map(lambda x, y: y))) for ds in train_split] # we could include val_split here, as long as we don't use test sesh
        prior_layer = PriorLayer(args.n_bins, args.min_hz, args.max_hz, is_online=True, return_probs=False)
        prior_layer.fit_layer(train_ys, distr=args.prior)

        # prepare model for inference
        out = prior_layer(model.output)
        inference_model = Model(model.input, out)

        # evaluate
        y_pred, uncertainty = inference_model.predict(test_sesh.batch(args.batch_size))
        y_true = np.array(list(test_sesh.map(lambda x, y: y)))

        test_mae = mean_absolute_error(y_true, y_pred)
        test_r2 = r2_score(y_true, y_pred)
        spearman_res = spearmanr(np.abs(y_true - y_pred), uncertainty)

        # log results
        maes[names[test_ixes[0]]] = [test_mae]
        logging.info(r"Session %s-%s MAE %f" % (args.dataset, names[test_ixes[0]], test_mae))
        if args.use_wandb:
            wandb.log({f"test-mae": test_mae})
            wandb.log({f"test-r2": test_r2})
            wandb.log({f"spearman-corr": spearman_res.correlation})
            wandb.log({f"prediction": y_pred})
            for j in range(len(y_true)):
                wandb.log({f"truth": y_true[j]})
                wandb.log({f"pred": y_pred[j]})
                wandb.log({f"certainty": -uncertainty[j]})
                wandb.log({f"error": np.abs(y_true-y_pred)})

            wandb.finish()

        # save to out dir
        try:
            model.save(os.path.join(out_path, "raw_model.h5"))
            inference_model.save(os.path.join(out_path, "inference_model.h5"))
        except ImportError:
            logging.warn("Failed to save model. ImportError: check your h5py installation.")
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        np.savetxt(os.path.join(out_path ,"y_pred.csv"), y_pred, delimiter=",")
        np.savetxt(os.path.join(out_path ,"y_true.csv"), y_true, delimiter=",")
        np.savetxt(os.path.join(out_path ,"uncertainty.csv"), uncertainty, delimiter=",")

    # compile summary
    result_df = pd.DataFrame(maes)
    result_df["mean"] = result_df.mean(axis=1)
    result_df["std"] = result_df.std(axis=1)

    result_df.to_csv(os.path.join(args.out_dir, "results"))
    logging.info(result_df.to_string())


if __name__ == "__main__":
    args = parse_args()
    train_eval(args)