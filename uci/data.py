import numpy as np
from collections import defaultdict

def get_labels_per_file():
    """For every file, get a list of windows with labels"""
    labels_file = open('dataset/RawData/labels.txt')

    labels = defaultdict(list)

    for line in labels_file:
        line = map(int, line.split(' '))
        labels[(line[0], line[1])].append(((line[3], line[4]), line[2]))

    return {file : np.array(windows) for file, windows in labels.iteritems()}

def get_windows_per_label():
    """
    For every label, get a list of values detailing info about each occurence
    in dataset.
    """
    labels_file = open('dataset/RawData/labels.txt')

    labels = defaultdict(list)

    for line in labels_file:
        line = map(int, line.split(' '))
        labels[line[2]].append((line[0], line[1], line[3], line[4]))

    return {label : np.array(values) for label, values in labels.iteritems()}

    # for label, windows in labels.items():
    #     print label,
    #     windows = np.array(windows)
    #     window_sizes = windows[:, 3] - windows[:, 2]
    #     print np.mean(window_sizes), np.std(window_sizes)

def get_data(file, start, stop):
    """Read in raw data into a numpy array"""
    data = open(file).readlines()[start : stop + 1]
    return np.array(map(lambda line: map(float, line.split(' ')), data))

def get_raw_data_files(exp, user):
    """Get name of raw data file given experiment and user number"""
    suffix = 'exp' + format(exp, '02') + '_user' + format(user, '02') + '.txt'
    prefix = 'dataset/RawData/'
    return prefix + 'acc_' + suffix, prefix + 'gyro_' + suffix

def get_segment_labels(labels_per_file):
    """For every file, get a list of segments with labels"""
    segment_labels = defaultdict(list)

    for file, windows_with_labels in labels_per_file.iteritems():
        for window, label in windows_with_labels:
            num_segments = ((window[1] - window[0]) // 64) - 1

            start = window[0]
            for i in range(num_segments):
                segment_labels[file].append(((start, start + 128), label))
                start += 64

    return segment_labels

if __name__ == '__main__':
    segment_labels = get_segment_labels(get_labels_per_file())

    print sum([len(segments) for segments in segment_labels.values()])
