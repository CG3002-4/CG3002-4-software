import numpy as np
import data
import pickle
from collections import defaultdict

SEGMENT_SIZE = 128
OVERLAP_SIZE = SEGMENT_SIZE // 2

def get_segment_labels(labels_per_file):
    """
    For every file, get a list of segments with labels

    Returns:
        {
            (experiment_id, user_id):
                [((segment_start, segment_stop_exc), segment_label),
                 ...
                ],
            ...
        }
    """
    segment_labels = defaultdict(list)

    for file_ids, windows_with_labels in labels_per_file.iteritems():
        for window, label in windows_with_labels:
            num_segments = ((window[1] + 1 - window[0]) // OVERLAP_SIZE) - 1

            start = window[0]
            for i in range(num_segments):
                segment_labels[file_ids].append(((start, start + SEGMENT_SIZE), label))
                start += OVERLAP_SIZE

    return segment_labels

def get_raw_segments(labels_per_file):
    """
    Get a list of all the segments given a mapping from files to
    segment_labels.

    Returns:
        [
            {
                "acc": [[x, y, z], ...]
                "gyro": [[x, y, z], ...]
            },
            ...
        ]
    """
    segment_labels = get_segment_labels(labels_per_file)

    raw_segments = []

    for file_ids, segments in segment_labels.iteritems():
        acc_file, gyro_file = data.get_raw_data_files(file_ids[0], file_ids[1])
        acc_values, gyro_values = data.get_data(acc_file), data.get_data(gyro_file)

        for segment, label in segments:
            acc_seg = acc_values[segment[0] : segment[1]]
            gyro_seg = gyro_values[segment[0] : segment[1]]

            raw_segments.append(({
                "acc": acc_seg,
                "gyro": gyro_seg
            }, label))

    return np.array(raw_segments)

def save_segments(segments, filename):
    with open(filename, 'wb') as output_file:
        pickle.dump(segments, output_file)

def load_segments(filename):
    with open(filename, 'rb') as input_file:
        return pickle.load(input_file)

if __name__ == '__main__':
    print len(get_raw_segments(data.get_labels_per_file()))
