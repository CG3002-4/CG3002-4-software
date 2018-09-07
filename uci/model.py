import data
import segmenting
import feature_extraction

FEATURES = [feature_extraction.freq_amps]

if __name__ == '__main__':
    labels_per_file = data.get_labels_per_file()

    segments = segmenting.get_raw_segments(labels_per_file)

    print segments.shape

    features = feature_extraction.extract_features(segments[:, 0], FEATURES)
    labels = segments[:, 1]

    print features.shape
    print labels.shape
