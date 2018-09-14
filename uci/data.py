import numpy as np
from collections import defaultdict


def get_labels():
    """
    For specified file, return a formatted dict of labelled activities with
    the experiment and user id, along with label start and stop.

    Format:
        {
            (experiment_id, user_id):
                [((label_start, label_stop_inclusive), label),
                 ...
                ],
            ...
        }
    """
    file_location = './dataset/RawData/labels.txt'

    with open(file_location) as labels_file:
        labels = defaultdict(list)

        for line in labels_file:
            line = list(map(int, line.split(' ')))
            labels[(line[0], line[1])].append(((line[3], line[4]), line[2]))

        return {file_ids: np.array(label_info) for file_ids, label_info in labels.items()}


def get_windows_per_label():
    """
    For every label, get a list of values detailing info about each occurrence
    in dataset.

    Format:
        {
            label:
                [(experiment_id, user_id, label_start, label_stop_inc)
                 ...
                ],
            ...
        }
    """
    with open('dataset/RawData/labels.txt') as labels_file:
        labels = defaultdict(list)

        for line in labels_file:
            line = list(map(int, line.split(' ')))
            labels[line[2]].append((line[0], line[1], line[3], line[4]))

        return {label: np.array(values) for label, values in labels.items()}

        # for label, windows in labels.items():
        #     print label,
        #     windows = np.array(windows)
        #     window_sizes = windows[:, 3] - windows[:, 2]
        #     print np.mean(window_sizes), np.std(window_sizes)


def format_raw_data(file_location):
    """
    Formats raw data from given file location into a numpy array of floats.

    Format:
        [[x, y, z], ...]
    """
    with open(file_location) as data:
        # For every line in data, parse data as floats into np array
        return np.array(list(map(lambda line: list(map(float, line.split(' '))), data)))


def get_raw_acc_gyro(expr_id, user_id):
    """Returns names of raw acc and gyro data files matching given experiment and user id"""
    suffix = 'exp' + format(expr_id, '02') + '_user' + \
        format(user_id, '02') + '.txt'
    prefix = 'dataset/RawData/'
    return prefix + 'acc_' + suffix, prefix + 'gyro_' + suffix
