import numpy as np
import matplotlib.pyplot as plt
import plot
import data
from scipy import signal
from sklearn import preprocessing
from compose import compose

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

def preprocess(preprocessf, data):
    return np.apply_along_axis(preprocessf, 0, data)

if __name__ == '__main__':
    labels = data.get_labels()

    exp, user, start, stop = labels[1][0]

    acc_file, gyro_file = data.get_raw_data_files(exp, user)

    acc_data = data.get_data(acc_file, start, stop)
    # gyro_data = data.get_data(gyro_file, start, stop)

    plt.figure(facecolor="white", figsize=(15,7))

    original = preprocess(compose(), acc_data)

    plt.subplot(221)
    plot.plot_data(original, 'Acc')

    plt.subplot(222)
    plot.plot_freq_spec(original, 'Acc')

    preprocessed = preprocess(compose(standardize, hann, medfilt), acc_data)
    plt.subplot(223)
    plot.plot_data(preprocessed, 'Acc_pre')

    plt.subplot(224)
    plot.plot_freq_spec(preprocessed, 'Acc_pre')

    # plot_data(gyro_data, 'Gyro')
    plt.show()
