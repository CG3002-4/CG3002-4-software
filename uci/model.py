import data
import segmenting
import preprocess
import feature_extraction
import numpy as np

PREPROCESS_FUNCS = [preprocess.hann, preprocess.medfilt]

FEATURES = [feature_extraction.energy,
            feature_extraction.entropy,
            feature_extraction.mean,
            feature_extraction.stdev
            ]

def preprocess_segment(segment):
    segment['acc'] = preprocess.preprocess(PREPROCESS_FUNCS, segment['acc'])
    segment['gyro'] = preprocess.preprocess(PREPROCESS_FUNCS, segment['gyro'])

    return segment

def extract_features(segments):
    segments = np.array([preprocess_segment(segment) for segment in segments])
    return feature_extraction.extract_features(segments, FEATURES)

if __name__ == '__main__':
    # labels_per_file = data.get_labels_per_file()
    #
    # segments = segmenting.get_raw_segments(labels_per_file)
    #
    # segmenting.save_segments(segments, 'all_segments.txt')

    segments = segmenting.load_segments('all_segments.txt')

    print segments.shape

    features = extract_features(segments[:, 0])
    labels = segments[:, 1]

    print features.shape
