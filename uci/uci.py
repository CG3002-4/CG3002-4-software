import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fftpack
from sklearn import preprocessing
from collections import defaultdict
from compose import compose

def get_labels():
    labels_file = open('dataset/RawData/labels.txt')

    labels = defaultdict(list)

    for line in labels_file:
        line = map(int, line.split(' '))
        labels[line[2]].append((line[0], line[1], line[3], line[4]))

    labels = {label : np.array(values) for label, values in labels.iteritems()}

    # for label, windows in labels.items():
    #     print label,
    #     windows = np.array(windows)
    #     window_sizes = windows[:, 3] - windows[:, 2]
    #     print np.mean(window_sizes), np.std(window_sizes)

    return labels

def get_data(file, start, stop):
    data = open(file).readlines()[start : stop + 1]
    return np.array(map(lambda line: map(float, line.split(' ')), data))

def get_raw_data_files(exp, user):
    suffix = 'exp' + format(exp, '02') + '_user' + format(user, '02') + '.txt'
    prefix = 'dataset/RawData/'
    return prefix + 'acc_' + suffix, prefix + 'gyro_' + suffix

def plot_data(data, title):
    plt.title(title)
    plt.plot(range(len(data)), data[:, 0], 'r-', label='x')
    plt.plot(range(len(data)), data[:, 1], 'g-', label='y')
    plt.plot(range(len(data)), data[:, 2], 'b-', label='z')
    # plt.legend(loc='upper left')

def plot_freq_spec(data, title):
    plt.title(title)
    def plot_freq_spec(axis, line, label):
        n = len(axis)
        fft = fftpack.fft(axis) / n
        fft = fft[range(n / 2)]
        plt.plot(range(n / 2), abs(fft), line, label=label)
    plot_freq_spec(data[:, 0], 'r-', label='x')
    plot_freq_spec(data[:, 1], 'g-', label='y')
    plot_freq_spec(data[:, 2], 'b-', label='z')

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
    labels = get_labels()

    exp, user, start, stop = labels[1][0]

    acc_file, gyro_file = get_raw_data_files(exp, user)

    acc_data = get_data(acc_file, start, stop)
    gyro_data = get_data(gyro_file, start, stop)

    plt.figure(facecolor="white", figsize=(15,7))

    original = preprocess(compose(), acc_data)

    plt.subplot(221)
    plot_data(original, 'Acc')

    plt.subplot(222)
    plot_freq_spec(original, 'Acc')

    preprocessed = preprocess(compose(standardize, hann, medfilt), acc_data)
    plt.subplot(223)
    plot_data(preprocessed, 'Acc_pre')

    plt.subplot(224)
    plot_freq_spec(preprocessed, 'Acc_pre')

    # plot_data(gyro_data, 'Gyro')
    plt.show()
