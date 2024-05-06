import os
from typing import Tuple

import numpy as np
import tensorflow as tf

from beliefppg.datasets.pipeline_generator import (
    prepare_session_spec,
    prepare_session_time,
)
from beliefppg.model.config import InputConfig
from beliefppg.model.load import load_inference_model
from beliefppg.util.preprocessing import get_strided_windows


def infer_hr(ppg: np.array, acc: np.array, ppg_freq: int, acc_freq: int, decoding: str = "sumproduct",
             batch_size: int = 128, filter_lowcut: float = 0.1, filter_highcut: float = 18.0,
             use_gpu: bool = False, model_path: str = None, use_time_backbone=True) -> Tuple[np.array, np.array]:
    """
    Infers heart rate from PPG and accelerometer data using the specified decoding method.
    :param ppg: PPG signal data with shape (n_samples, n_channels).
    :param acc: Accelerometer signal data with shape (n_samples, n_channels).
    :param ppg_freq: Sampling frequency of the PPG signal in Hz
    :param acc_freq: Sampling frequency of the accelerometer signal in Hz
    :param decoding: Decoding method to use, either "sumproduct" or "viterbi"
    :param batch_size: Batch size for inference
    :param filter_lowcut: Lowcut frequency for filtering (per default set to 0.1 Hz which is used for training the default model)
    :param filter_highcut: Highcut frequency for filtering (per default set to 18.0 Hz which is used for training the default model)
    :param use_gpu: Whether to use GPU for inference or not
    :param model_path: Path to the inference model. If None, the default model will be loaded.
    :param use_time_backbone: Whether to use the time-domain backbone or not
    :return: Tuple of predicted heart rates and uncertainties
    """

    # Set TensorFlow to use GPU or CPU based on the parameter
    if ppg.ndim != 2:
        raise ValueError("PPG signal data must have shape (n_samples, n_channels)")
    if acc.ndim != 2:
        raise ValueError("Accelerometer signal data must have shape (n_samples, n_channels)")
    
    if not isinstance(ppg_freq, int):
        print("Warning: ppg_freq is not an integer, converting to integer.")
        ppg_freq = round(ppg_freq)
    if not isinstance(acc_freq, int):
        print("Warning: acc_freq is not an integer, converting to integer.")
        acc_freq = round(acc_freq)

    if use_gpu:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            print("No GPU devices found, switching to CPU.")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Load default model if model_path is not provided
    if model_path is None:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        if use_time_backbone:
            model_path = os.path.join(model_dir, "inference_model.keras")
        else:
            model_path = os.path.join(model_dir, "inference_model_notimebackbone.keras")

    # Load the inference model
    inference_model = load_inference_model(model_path)
    prior_layer = inference_model.layers[-1]
    min_hz = prior_layer.min_hz
    max_hz = prior_layer.max_hz
    n_bins = prior_layer.dim
    n_frames = inference_model.input_spec[0].shape[1]
    win_size = InputConfig.WINSIZE
    stride = InputConfig.STRIDE
    time_winsize = win_size + (n_frames - 1) * stride
    freq = inference_model.input_spec[1].shape[1] // time_winsize

    # Set up the decoding method
    if decoding == "sumproduct":
        prior_layer = inference_model.get_layer("prior_layer")
        prior_layer.set_online(True)
    elif decoding == "viterbi":
        prior_layer = inference_model.get_layer("prior_layer")
        prior_layer.set_online(False)
    else:
        raise NotImplementedError(f"Decoding method {decoding} not implemented")

    # Prepare spectral features
    spectral_feat = prepare_session_spec(ppg, acc, ppg_freq, acc_freq,
                                         win_size, stride, n_bins, min_hz,
                                         max_hz)
    spectral_dss = tf.data.Dataset.from_tensor_slices(spectral_feat)
    spectral_dss = get_strided_windows(spectral_dss, win_size=n_frames, stride=1)

    # Prepare time-domain features
    time_feat = prepare_session_time(ppg, ppg_freq, freq,
                                     filter_lowcut, filter_highcut)
    time_dss = tf.data.Dataset.from_tensor_slices(time_feat)
    time_dss = get_strided_windows(time_dss, win_size=time_winsize*freq,
                                   stride=stride*freq)

    # Combine spectral and time-domain features into a single dataset
    X_dss = tf.data.Dataset.zip(spectral_dss, time_dss)

    # Create dummy labels for the dataset
    labels = [-1 for _ in time_dss]
    label_dss = tf.data.Dataset.from_tensor_slices(labels)
    joint_dss = tf.data.Dataset.zip(X_dss, label_dss)

    # Batch and prefetch the dataset for inference
    batches = joint_dss.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    y_pred, uncertainty = inference_model.predict(batches)

    # Compute time intervals
    time_start = np.arange(0, len(y_pred) * stride, stride, dtype=np.float32)
    time_intervals = np.column_stack((time_start, time_start + stride))

    return y_pred, uncertainty, time_intervals
