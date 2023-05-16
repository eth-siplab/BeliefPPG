""""""
import glob
import logging
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_dalia(data_dir):
    """
    Functionality to load PPG-DaLiA dataset. Assumes it is stored under data_dir/DaLiA/*
    Downloadable under https://archive.ics.uci.edu/ml/machine-learning-databases/00495/data.zip
    :param data_dir: base directory to data folder
    :return: tuple of (signals, names, ppg_freq, acc_freq
        signals: list of signal tuples (ppg, acc, hr) each representing one session/subject.
                 Signals are assumed to be in channels-last format.
                 Ground truth hr is assumed to be the instantaneous HR in an 8-second window with 2s stride.
        names:   list of strings describing the sessions
        ppg_freq, acc_freq: respective sampling frequencies
    """
    signals, names = [], []
    ppg_freq = 64
    acc_freq = 32
    nsamples = 0
    tpl = os.path.join(*[data_dir, "DaLia", "S*", "S*.pkl"])
    for fname in sorted(glob.glob(tpl)):
        # load
        ds = np.load(fname, allow_pickle=True, encoding="bytes")
        # parse
        acc = ds[b"signal"][b"wrist"][b"ACC"]
        ppg = ds[b"signal"][b"wrist"][b"BVP"]
        hr = ds[b"label"]
        signals.append((ppg, acc, hr))
        names.append(os.path.split(fname)[1][:-4])
        nsamples += len(ppg) / (60 * ppg_freq)

    assert nsamples > 0, r"Did not find any files matching path %s" % tpl

    logging.info(
        r"Loaded DaLiA (signal length: %d min, number of sessions: %d)"
        % (nsamples, len(signals))
    )
    return signals, names, ppg_freq, acc_freq


def load_wesad(data_dir):
    """
    Functionality to load WESAD dataset. Assumes it is stored under data_dir/WESAD/*
    Downloadable under https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download
    :param data_dir: base directory to data folder
    :return: tuple of (signals, names, ppg_freq, acc_freq
        signals: list of signal tuples (ppg, acc, hr) each representing one session/subject.
                 Signals are assumed to be in channels-last format.
                 Ground truth hr is assumed to be the instantaneous HR in an 8-second window with 2s stride.
        names:   list of strings describing the sessions
        ppg_freq, acc_freq: respective sampling frequencies
    """
    signals, names = [], []
    ppg_freq = 64
    acc_freq = 32
    nsamples = 0
    tpl = os.path.join(*[data_dir, "WESAD", "S*", "S*.pkl"])
    for fname in sorted(glob.glob(tpl)):
        # load
        ds = np.load(fname, allow_pickle=True, encoding="bytes")
        name = os.path.split(fname)[1][:-4]
        # parse
        ppg = ds[b"signal"][b"wrist"][b"BVP"]
        acc = ds[b"signal"][b"wrist"][b"ACC"]
        hr = np.loadtxt(
            os.path.join(*[data_dir, "WESAD", "GeneratedLabels", f"HR_{name}.csv"]),
            delimiter=",",
        )
        signals.append((ppg, acc, hr))
        names.append(name)
        nsamples += len(ppg) / (60 * ppg_freq)

    assert nsamples > 0, r"Did not find any files matching path %s" % tpl

    logging.info(
        r"Loaded WESAD (signal length: %d min, number of sessions: %d)"
        % (nsamples, len(signals))
    )
    return signals, names, ppg_freq, acc_freq


def load_bami_1(data_dir):
    """
    Functionality to load the BAMI-1 dataset. Assumes it is stored under data_dir/BAMI/BAMI-1*
    Downloadable jointly with BAMI-2 under
    https://github.com/HeewonChung92/CNN_LSTM_HeartRateEstimation/archive/refs/heads/master.zip
    :param data_dir: base directory to data folder
    :return: tuple of (signals, names, ppg_freq, acc_freq
        signals: list of signal tuples (ppg, acc, hr) each representing one session/subject.
                 Signals are assumed to be in channels-last format.
                 Ground truth hr is assumed to be the instantaneous HR in an 8-second window with 2s stride.
        names:   list of strings describing the sessions
        ppg_freq, acc_freq: respective sampling frequencies
    """
    signals, names = [], []
    ppg_freq = 50
    acc_freq = 50
    nsamples = 0
    tpl = os.path.join(*[data_dir, "BAMI", "BAMI-1", "BAMI1_*.mat"])
    for fname in sorted(glob.glob(tpl)):
        # load
        mat = loadmat(fname)
        # parse
        ppg = np.array(mat["rawPPG"]).T
        acc = np.array(mat["rawAcc"]).T
        hr = np.array(mat["bpm_ecg"]).flatten()
        length = int((len(ppg[0]) - 6 * 50) / (2 * 50))
        hr = hr[-length:]  # there is a sequence with one label too much... wonder why?

        signals.append((ppg, acc, hr))
        names.append(os.path.split(fname)[1][:-4])
        nsamples += len(ppg) / (60 * ppg_freq)

    assert nsamples > 0, r"Did not find any files matching path %s" % tpl

    logging.info(
        r"Loaded BAMI-1 (signal length: %d min, number of sessions: %d)"
        % (nsamples, len(signals))
    )
    return signals, names, ppg_freq, acc_freq


def load_bami_2(data_dir):
    """
    Functionality to load the BAMI-2 dataset. Assumes it is stored under data_dir/BAMI/BAMI-2*
    Downloadable jointly with BAMI-1 under
    https://github.com/HeewonChung92/CNN_LSTM_HeartRateEstimation/archive/refs/heads/master.zip
    :param data_dir: base directory to data folder
    :return: tuple of (signals, names, ppg_freq, acc_freq
        signals: list of signal tuples (ppg, acc, hr) each representing one session/subject.
                 Signals are assumed to be in channels-last format.
                 Ground truth hr is assumed to be the instantaneous HR in an 8-second window with 2s stride.
        names:   list of strings describing the sessions
        ppg_freq, acc_freq: respective sampling frequencies
    """
    signals, names = [], []
    ppg_freq = 50
    acc_freq = 50
    nsamples = 0
    tpl = os.path.join(*[data_dir, "BAMI", "BAMI-2", "BAMI2_*.mat"])
    for fname in sorted(glob.glob(tpl)):
        # load
        mat = loadmat(fname)
        # parse
        ppg = np.array(mat["rawPPG"]).T
        acc = np.array(mat["rawAcc"]).T
        hr = np.array(mat["bpm_ecg"]).flatten()
        length = int((len(ppg[0]) - 6 * 50) / (2 * 50))
        length2 = len(hr) * (2 * 50) + 6 * 50
        hr = hr[-length:]  # there is a sequence with one label too much... wonder why?
        ppg = np.array(
            [p[-length2:] for p in ppg]
        )  # one sequence is one label short... wonder why?
        signals.append((ppg, acc, hr))
        names.append(os.path.split(fname)[1][:-4])
        nsamples += len(ppg) / (60 * ppg_freq)

    assert nsamples > 0, r"Did not find any files matching path %s" % tpl

    logging.info(
        r"Loaded BAMI-2 (signal length: %d min, number of sessions: %d)"
        % (nsamples, len(signals))
    )
    return signals, names, ppg_freq, acc_freq


def load_ieee(data_dir, load_train=True, load_extra=False, load_test=False):
    """
    Functionality to load IEEE train and test datasets. Assumes it is stored under data_dir/IEEE/*
    Original link became unreachable. Downloadable in a NEW FORMAT under https://zenodo.org/record/3902710#.ZGKi-3ZBy3A
    :param data_dir: base directory to data folder
    :param load_train: whether to load the training sequences
    :param load_extra: whether to load the 13th training sequence
    :param load_test: whether to load the test sequences
    :return: tuple of (signals, names, ppg_freq, acc_freq
        signals: list of signal tuples (ppg, acc, hr) each representing one session/subject.
                 Signals are assumed to be in channels-last format.
                 Ground truth hr is assumed to be the instantaneous HR in an 8-second window with 2s stride.
        names:   list of strings describing the sessions
        ppg_freq, acc_freq: respective sampling frequencies
    """
    ppg_freq, acc_freq = 125, 125
    signals, names = [], []
    nsamples = 0
    # prepare filenames
    train_tpl = os.path.join(
        *[data_dir, "IEEE", "Training_data", "DATA_*_TYPE[0-9][0-9].mat"]
    )
    extra_tpl = os.path.join(
        *[data_dir, "IEEE", "Extra_TrainingData", "DATA_*_T[0-9]*.mat"]
    )
    test_tpl = os.path.join(*[data_dir, "IEEE", "IEEE", "TestData", "TEST_S*_T*.mat"])

    train_names = (glob.glob(train_tpl) if load_train else []) + (
        glob.glob(extra_tpl) if load_extra else []
    )
    test_names = glob.glob(test_tpl) if load_test else []
    fnames = sorted(train_names) + sorted(test_names)

    for fname in fnames:
        mat = loadmat(fname)
        # get the labels
        if "Extra_TrainingData" in fname:
            bpm_file = fname.replace("DATA", "BPM")
        elif "Training" in fname:
            bpm_file = fname.replace(".mat", "_BPMtrace.mat")
        else:
            bpm_file = fname.replace("TEST", "True")

        hr = loadmat(bpm_file)["BPM0"].flatten()
        df = pd.DataFrame(mat["sig"].swapaxes(0, 1))

        df.columns = ["ecg", "ppg1", "ppg2", "acc1", "acc2", "acc3"]

        ppg = np.stack([df["ppg1"], df["ppg2"]], axis=-1)
        acc = np.stack([df["acc1"], df["acc2"], df["acc3"]], axis=-1)

        signals.append((ppg, acc, hr))
        names.append(os.path.split(fname)[1][:-4])
        nsamples += len(ppg) / (60 * ppg_freq)

    assert nsamples > 0, r"Did not find any files matching paths %s %s %s" % (
        train_tpl,
        extra_tpl,
        test_tpl,
    )

    logging.info(
        r"Loaded IEEE (signal length: %d min, number of sessions: %d)"
        % (nsamples, len(signals))
    )
    logging.info((r"The following files were read: %s" % ("\n".join(["\n"] + fnames))))
    return signals, names, ppg_freq, acc_freq
