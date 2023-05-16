import numpy as np
from scipy.signal import butter, lfilter
from scipy.fft import rfft, rfftfreq
import scipy


def get_strided_windows(ds, win_size, stride):
    """
    Generates a sliding window view of a tf.data.Dataset along the first axis
    :param win_size: number of samples in window
    :param stride: number of skipped samples between consecutive windows
    :return: tf.data.Dataset yielding tensors of shape (win_size, ) + old_shape
    """
    ds = ds.window(size=win_size, shift=stride, drop_remainder=True)
    return ds.flat_map(lambda window: window.batch(win_size))

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Helper function for butter_bandpass_filter
    Creates a scipy filter
    https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter

    :param lowcut: lower cut-off value for butterworth filter (Hz)
    :param highcut: upper cut-off value for butterworth filter (Hz)
    :param fs: frequency of signal in Hz
    :param order: order of filter
    :return:(ndarray, ndarray)  Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.scipy filter object
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Filters the signal with specified butter-bandpass filter
    :param data: single-channel signal (one-dimensional)
    :param lowcut: . of the filter
    :param highcut: . of the filter
    :param fs: input sampling frequency
    :param order: . of the filter
    :return: filtered signal of same shape
    """
    data = scipy.signal.detrend(data)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def process_window_spec_ppg(sig, freq, resolution, min_hz, max_hz):
    """
    Processes a window of PPG signal into spectral representation. Custom implementation of STFT.
    The channels are averaged, the signal is downsampled and the FFT is computed. Only amplitudes within relevant
    range are returned.
    :param sig: PPG signal of shape (n_samples, n_channels)
    :param freq: signal frequency (Hz)
    :param resolution: Number of points for the FFT algorithm
    :param min_hz: Lower cutoff frequency for spectrum
    :param max_hz: Higher cutoff frequency for spectrum
    :return: Spectrogram of shape (n_steps, n_freq_bins)
    """
    filt = lambda x : butter_bandpass_filter(x, 0.4, 4, fs=freq, order=4)
    sig = np.apply_along_axis(filt, 0, sig)
    sig = (sig - sig.mean(axis=0)[None, :]) / (sig.std(axis=0)[None, :] + 1e-10)
    sig = sig.mean(axis=-1) # average over channels if multiple present

    sig = scipy.signal.resample(sig, int(len(sig) * 25 / freq)) # resample to 25 hz (optional but faster)
    # do FFT
    if resolution > len(sig):
        sig = np.pad(sig, (0, resolution - len(sig)))
    y = np.abs(rfft(sig, axis=0))
    freq = rfftfreq(len(sig), 1 / 25)
    # extract relevant frequencies
    y = y[(freq > min_hz) & (freq < max_hz)]
    return y


def process_window_spec_acc(sig, freq, resolution: int, min_hz: float, max_hz: float):
    """
    Processes a window of accelerometer signal into spectral representation. Custom implementation of STFT.
    The signal is downsampled, FFT is computed and the channels are averaged. Only amplitudes within relevant
    range are returned.
    :param sig: ACC signal of shape (n_samples, n_channels)
    :param freq: signal frequency (Hz)
    :param resolution: Number of points for the FFT algorithm
    :param min_hz: Lower cutoff frequency for spectrum
    :param max_hz: Higher cutoff frequency for spectrum
    :return: Spectrogram of shape (n_steps, n_freq_bins)
    """
    filt = lambda x: butter_bandpass_filter(x, 0.4, 4, fs=freq, order=4)
    sig = np.apply_along_axis(filt, 0, sig)
    sig = (sig - sig.mean(axis=0)[None, :]) / (sig.std(axis=0)[None, :] + 1e-10)

    sig = scipy.signal.resample(sig, int(len(sig) * 25 / freq)) # resample to 25 hz (optional but faster)
    # do FFT
    if resolution > len(sig):
        sig = np.pad(sig, ((0, resolution - len(sig)), (0,0)))
    y = np.abs(rfft(sig, axis=0))
    freq = rfftfreq(len(sig), 1 / 25)
    # extract relevant frequencies
    y = y[(freq > min_hz) & (freq < max_hz), :]
    return y.mean(axis=-1)


def process_window_time(ppg, ppg_freq, target_freq, filter_lowcut, filter_highcut):
    """
    Prepares time-domain PPG signal as input. Preprocessing consists of filtering, averaging & resampling.
    :param ppg: PPG signal of shape (n_samples, n_channels)
    :param ppg_freq: frequency of signal
    :param target_freq: desired frequency for input
    :param filter_lowcut: lower cutoff frequency for bandpass filtering
    :param filter_highcut: upper cutoff frequency for bandpass filtering
    :return: time-domain signal of shape (n_samples_new,)
    """
    import matplotlib.pyplot as plt
    #
    # plt.title("raw")
    # plt.plot(ppg[:1500])
    # plt.show()
    filt = lambda x : butter_bandpass_filter(x, filter_lowcut, filter_highcut, ppg_freq)
    ppg = np.apply_along_axis(filt, 0, ppg)
    # ppg = ppg.flatten()
    # ppg = np.expand_dims(ppg,-1)
    # plt.plot(ppg[:1500])
    # plt.title("filt")
    # plt.show()
    # print(ppg.mean(axis=0)[None, :], ppg.std(axis=0)[None, :])
    # ppg = (ppg - ppg.mean(axis=0)[None, :]) / (ppg.std(axis=0)[None, :] + 1e-10)
    ppg = ppg.mean(axis=-1) # average over channels if multiple present
    # plt.plot(ppg[:1500])
    # plt.title("avg")
    # plt.show()
    ppg = scipy.signal.resample_poly(ppg, target_freq, ppg_freq)
    ppg = np.expand_dims(ppg, -1)
    # plt.plot(ppg[:500])
    # plt.title("after")
    # plt.show()
    # make sure each channel contributes equally
    return ppg
