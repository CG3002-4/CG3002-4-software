import numpy as np
from collections import defaultdict

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
