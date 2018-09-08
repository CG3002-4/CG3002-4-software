import numpy as np
import data
import segmenting
from scipy import stats

def extract_features_over_axes(segment, feature_func):
    return np.concatenate(
        [np.apply_along_axis(feature_func, 0, segment["acc"]).flatten(),
         np.apply_along_axis(feature_func, 0, segment["gyro"]).flatten()
        ]
    )

FFT_NUM_AMPS = 3

def freq_amps(segment):
    def freq_amps(axis):
        return np.abs(np.fft.rfft(axis))[:FFT_NUM_AMPS]
    return extract_features_over_axes(segment, freq_amps)

def energy(segment):
    def energy(axis):
        freq_components = np.abs(np.fft.rfft(axis))[1:]
        return np.sum(freq_components ** 2) / len(freq_components)
    return extract_features_over_axes(segment, energy)

def entropy(segment):
    def entropy(axis):
        freq_components = np.abs(np.fft.rfft(axis))[1:]
        return stats.entropy(freq_components, base=2)
    return extract_features_over_axes(segment, entropy)

def mean(segment):
    def mean(axis):
        return np.mean(axis)
    return extract_features_over_axes(segment, mean)

def stdev(segment):
    def stdev(axis):
        return np.std(axis)
    return extract_features_over_axes(segment, stdev)

def extract_features(segments, feature_funcs):
    def extract_features(segment):
        feature_lists = [feature_func(segment) for feature_func in feature_funcs]
        return np.concatenate(feature_lists)
    return np.array([extract_features(segment) for segment in segments])



if __name__ == '__main__':
    labels_per_file = data.get_labels_per_file()
    sample_labels_per_file = {(1, 1): [labels_per_file[1, 1][0]]}

    sample_segment = segmenting.get_raw_segments(sample_labels_per_file)[0][0]
