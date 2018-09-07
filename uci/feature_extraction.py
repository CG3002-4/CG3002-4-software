import numpy as np
import data
import segmenting

def extract_features_over_axes(segment, feature_func):
    return np.concatenate(
        [np.apply_along_axis(feature_func, 0, segment["acc"]).flatten(),
         np.apply_along_axis(feature_func, 0, segment["gyro"]).flatten()
        ]
    )

def freq_amps(segment):
    def freq_amps(axis):
        return np.abs(np.fft.rfft(axis))[:3]
    return extract_features_over_axes(segment, freq_amps)

def extract_features(segment, *feature_funcs):
    feature_lists = [feature_func(segment) for feature_func in feature_funcs]
    return np.concatenate(feature_lists)

if __name__ == '__main__':
    labels_per_file = data.get_labels_per_file()
    sample_labels_per_file = {(1, 1): [labels_per_file[1, 1][0]]}

    sample_segment = segmenting.get_raw_segments(sample_labels_per_file)[0][0]

    print extract_features(sample_segment, freq_amps)
