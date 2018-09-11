import numpy as np
import data
import compose
from scipy import signal
from sklearn import preprocessing

def medfilt(axis):
    return signal.medfilt(signal.medfilt(axis, 5), 3)

def hann(axis):
    hanning_window = signal.hann(5)[1:4]
    hanning_window = hanning_window / np.sum(hanning_window)
    return signal.fftconvolve(axis, hanning_window, mode='same')

def standardize(axis):
    return preprocessing.scale(axis)

def butter(axis):
    nyq = 50 * 0.5
    cutoff = 0.3 / nyq
    b, a = signal.butter(3, cutoff, btype='high', output='ba')
    return signal.lfilter(b, a, axis)

def preprocess(preprocessfs, data):
    return np.apply_along_axis(compose.compose(preprocessfs), 0, data)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import plot

    labels = data.get_windows_per_label()

    exp, user, start, stop = labels[1][0]

    acc_file, gyro_file = data.get_raw_data_files(exp, user)

    acc_data = data.get_data(acc_file)[start : stop]
    # gyro_data = data.get_data(gyro_file, start, stop)

    plt.figure(facecolor="white", figsize=(15,7))

    original = preprocess([], acc_data)

    plt.subplot(221)
    plot.plot_data(original, 'Acc')

    plt.subplot(222)
    plot.plot_freq_spec(original, 'Acc')

    # Standardize using global means
    preprocessed = preprocess([hann, medfilt], acc_data)
    plt.subplot(223)
    plot.plot_data(preprocessed, 'Acc_pre')

    plt.subplot(224)
    plot.plot_freq_spec(preprocessed, 'Acc_pre')

    # plot_data(gyro_data, 'Gyro')
    plt.show()
