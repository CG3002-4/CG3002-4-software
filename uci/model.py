import data
import segmenting
import feature_extraction

FEATURES = [
    feature_extraction.energy,
    feature_extraction.entropy,
    feature_extraction.mean,
    feature_extraction.stdev
]

if __name__ == '__main__':
    segments = segmenting.load_segments('all_segments.txt')

    print(segments.shape)  # Returns shape of nd-array as a tuple

    features = feature_extraction.extract_features(segments[:, 0], FEATURES)
    labels = segments[:, 1]

    print(features.shape)
