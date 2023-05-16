""""""
import multiprocessing
from argparse import Namespace
import tensorflow as tf
import numpy as np

from BeauPPG.datasets.file_reader import load_dalia, load_ieee, load_wesad, load_bami_1, load_bami_2
from BeauPPG.util.preprocessing import process_window_time, get_strided_windows, \
    process_window_spec_acc, process_window_spec_ppg



def prepare_session_spec(ppg, acc, ppg_freq, acc_freq, win_size, stride, n_bins, min_hz, max_hz):
    fft_winsize = 535 if n_bins == 64 else (4*535 - 5)
    ppgs = []
    ppg_wsize = win_size * ppg_freq
    for i in range(0, len(ppg) - ppg_wsize + 1, ppg_freq * stride):
        ppgs.append(process_window_spec_ppg(ppg[i:i + ppg_wsize], ppg_freq, fft_winsize, min_hz, max_hz))

    accs = []
    acc_wsize = win_size * acc_freq
    for i in range(0, len(acc) - acc_wsize + 1, acc_freq * stride):
        accs.append(process_window_spec_acc(acc[i:i + acc_wsize], acc_freq, fft_winsize, min_hz, max_hz))

    sig = np.stack([ppgs, accs], axis=-1)

    # normalize
    sig = (sig-sig.mean()) / (sig.std() + 1e-10)

    assert not np.isnan(sig).any()
    return sig.astype(np.float32)


def prepare_session_time(ppg, ppg_freq, target_freq, filter_lowcut, filter_highcut):
    # only feed ppg signals as time-domain features
    sig = process_window_time(ppg, ppg_freq, target_freq, filter_lowcut, filter_highcut)

    # normalize
    sig = (sig-sig.mean()) / (sig.std() + 1e-10)

    assert not np.isnan(sig).any()
    return sig.astype(np.float32)


def prepare_session_labels(hr, n_frames):
    offset = n_frames - 1
    assert not np.isnan(hr).any()
    return hr[offset:].astype(np.float32)


def get_sessions(args: Namespace):
    if args.dataset == "dalia":
        sessions, names, ppg_freq, acc_freq = load_dalia(args.data_dir)
    elif args.dataset == "wesad":
        sessions, names, ppg_freq, acc_freq = load_wesad(args.data_dir)
    elif args.dataset == "bami-1":
        sessions, names, ppg_freq, acc_freq = load_bami_1(args.data_dir)
    elif args.dataset == "bami-2":
        sessions, names, ppg_freq, acc_freq = load_bami_2(args.data_dir)
    elif args.dataset == "ieee":
        sessions, names, ppg_freq, acc_freq = load_ieee(args.data_dir)
    else:
        raise NotImplementedError(r'Dataset %s not supported (yet)' % args.dataset)

    window_len = 8 # hard-code window length in seconds to match the labels
    stride = 2 # window stride in seconds

    pool = multiprocessing.Pool()

    # compute time-frequency features
    spectral_feat = pool.starmap(
        prepare_session_spec,
        ((ppg, acc, ppg_freq, acc_freq, window_len, stride, args.n_bins, args.min_hz, args.max_hz) for (ppg, acc, hr) in sessions)
    )
    spectral_dss = map(tf.data.Dataset.from_tensor_slices, spectral_feat)
    spectral_dss = [get_strided_windows(sesh, win_size=args.n_frames, stride=1) for sesh in spectral_dss]

    # compute time-domain features
    time_feat = pool.starmap(
        prepare_session_time,
        ((ppg, ppg_freq, args.freq, args.filter_lowcut, args.filter_highcut) for (ppg, acc, hr) in sessions)
    )
    time_winsize = window_len + (args.n_frames - 1)*stride
    time_dss = map(tf.data.Dataset.from_tensor_slices, time_feat)
    time_dss = [get_strided_windows(ds, win_size=time_winsize * args.freq, stride=stride * args.freq) for ds in time_dss]

    # align labels
    labels = [prepare_session_labels(hr, args.n_frames) for (ppg, acc, hr) in sessions]
    label_dss = map(tf.data.Dataset.from_tensor_slices, labels)

    # create datasets yielding tuples of form ((X_spec, X_time), y)
    X_dss = map(tf.data.Dataset.zip, zip(spectral_dss, time_dss))
    joint_dss = map(tf.data.Dataset.zip, zip(X_dss, label_dss))

    # joint_dss = list(joint_dss)
    # import matplotlib.pyplot as plt
    # for i in range(len(joint_dss)):
    #     first = list(joint_dss[i])[60]
    #     first_x, first_y = first
    #     spec, time = first_x
    #     fig = plt.figure()
    #     fig.suptitle(r'New Spec %d' % i)
    #     plt.plot(spec[-1])
    #     fig.show()
    #     fig = plt.figure()
    #     fig.suptitle(r'New Time %d, Label = %f' % (i, first_y))
    #     plt.plot(time[-500:])
    #     fig.show()

    sessions = np.asarray([ds.prefetch(tf.data.AUTOTUNE) for ds in joint_dss])
    return sessions[np.argsort(names)], sorted(names)


def join_sessions(dss, shuffle):
    """
    Combines a list of per-session tf datasets. Interleaves them (otherwise we would need a large shuffle buffer)
    :param dss: list of tf.data.Datasets
    :param shuffle: whether to shuffle
    :return: concatenated or [randomly interleaved] cached tf dataset
    """
    if shuffle:
        dss = [ds.shuffle(1000) for ds in dss]
        # interleave datasets
        comb = tf.data.Dataset.from_tensor_slices(dss).interleave(lambda x: x, cycle_length=len(dss),
                                                           num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    else:
        comb = dss[0]
        for i in range(1, len(dss)):
            comb = comb.concatenate(dss[i])

    return comb
