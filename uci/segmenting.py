import numpy as np
import data
import pickle
from collections import defaultdict

SEGMENT_SIZE = 128
OVERLAP_SIZE = SEGMENT_SIZE // 2


def get_segments(labelled_activities):
    """
    Returns a list of segments with labels from given 
    formatted dictionary of labelled activities.

    Each labelled activity is divided into overlapping segments


    Format:
        {
            (experiment_id, user_id):
                [((segment_start, segment_stop_exc), segment_label),
                 ...
                ],
            ...
        }
    """
    labelled_segments = defaultdict(list)

    for file_ids, labelled_windows in labelled_activities.items():
        for window, label in labelled_windows:
            start = window[0]
            end = window[1]
            # -1 is for the overrun segment
            num_segments = ((end + 1 - start) // OVERLAP_SIZE) - 1

            for i in range(num_segments):
                labelled_segments[file_ids].append(
                    ((start, start + SEGMENT_SIZE), label))
                start += OVERLAP_SIZE

    return labelled_segments


def segment_activities(labelled_activities):
    """
    Returns a numpy array of segmented data from given 
    formatted dictionary of labelled activities. 

    Format:
        [
            {
                "acc": [[x, y, z], ...]
                "gyro": [[x, y, z], ...]
            },
            ...
        ]
    """
    labelled_segments = get_segments(labelled_activities)

    segmented_data = []

    for file_ids, segments in labelled_segments.items():
        expr_id = file_ids[0]
        user_id = file_ids[1]

        acc_file, gyro_file = data.get_raw_acc_gyro(expr_id, user_id)
        acc_values = data.format_raw_data(acc_file)
        gyro_values = data.format_raw_data(gyro_file)

        for segment, label in segments:
            acc_seg = acc_values[segment[0]: segment[1]]
            gyro_seg = gyro_values[segment[0]: segment[1]]

            segmented_data.append((
                {
                    "acc": acc_seg,
                    "gyro": gyro_seg
                },
                label))

    return np.array(segmented_data)


def save_segments(segments, filename):
    with open(filename, 'wb') as output_file:
        pickle.dump(segments, output_file)


def load_segments(filename):
    with open(filename, 'rb') as input_file:
        return pickle.load(input_file)


if __name__ == '__main__':
    print(len(segment_activities(data.get_labels())))
