import numpy as np
import pickle
import data
import compose
from scipy import signal
from sklearn import preprocessing


STATS_FILE = 'stats.txt'


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


def save_stats(preprocessfs, labels):
    all_acc_values = np.empty([0, 3])
    all_gyro_values = np.empty([0, 3])

    for file_ids, labelled_activities in labels.items():
        expr_id = file_ids[0]
        user_id = file_ids[1]

        acc_file, gyro_file = data.get_raw_acc_gyro(expr_id, user_id)
        acc_values = data.format_raw_data(acc_file)
        gyro_values = data.format_raw_data(gyro_file)

        for window, label in labelled_activities:
            start = window[0]
            end = window[1] + 1  # account for inclusive end

            all_acc_values = np.append(all_acc_values, acc_values[start: end], axis=0)
            all_gyro_values = np.append(all_gyro_values, gyro_values[start: end], axis=0)

    all_acc_values = preprocess(preprocessfs, all_acc_values)
    all_gyro_values = preprocess(preprocessfs, all_gyro_values)

    stats = {
        'means': {
            'acc': np.mean(all_acc_values, axis=0),
            'gyro': np.mean(all_gyro_values, axis=0)
        },
        'stdevs': {
            'acc': np.std(all_acc_values, axis=0),
            'gyro': np.std(all_gyro_values, axis=0)
        }
    }

    with open(STATS_FILE, 'wb') as stats_file:
        pickle.dump(stats, stats_file)


def load_stats():
    with open(STATS_FILE, 'rb') as stats_file:
        return pickle.load(stats_file)


def standardize_segment(segment):
    stats = load_stats()

    def standardize_sensor(sensor_name, axis):
        segment[sensor_name][:, axis] = (segment[sensor_name][:, axis] - stats['means'][sensor_name][axis])  # / stats['stdevs'][sensor_name][axis]

    for i in range(3):
        standardize_sensor('acc', i)
        standardize_sensor('gyro', i)

    return segment


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import plot

    labels = data.get_windows_per_label()

    exp, user, start, stop = labels[1][0]

    acc_file, gyro_file = data.get_raw_acc_gyro(exp, user)

    acc_data = data.format_raw_data(acc_file)[start: start + 200]
    # gyro_data = data.get_data(gyro_file, start, stop)

    plt.figure(facecolor="white", figsize=(15, 7))

    original = preprocess([], acc_data)

    plt.subplot(221)
    plot.plot_data(original, 'Acc')

    plt.subplot(222)
    plot.plot_freq_spec(original, 'Acc')

    # Standardize using global means
    preprocessed = preprocess([butter], acc_data)
    plt.subplot(223)
    plot.plot_data(preprocessed, 'Acc_pre')

    plt.subplot(224)
    plot.plot_freq_spec(preprocessed, 'Acc_pre')

    # plot_data(gyro_data, 'Gyro')
    plt.show()
